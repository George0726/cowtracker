#!/usr/bin/env python3
"""
CoWTracker Mask Propagation FastAPI Service

A REST API service for propagating masks through video frames using CoWTracker DENSE tracking.
The model is loaded once at startup and reused for all requests.

Usage:
    # Start the server
    python cowtracker_fastapi.py --host 0.0.0.0 --port 8000

    # Make requests using Python requests library
    import requests

    # Simple request
    with open("video.mp4", "rb") as video, open("mask.png", "rb") as mask:
        response = requests.post(
            "http://localhost:8000/propagate",
            files={"video": video, "mask": mask},
            data={"enable_smoothing": "true"}
        )
    output_video = response.content

    # With options
    with open("video.mp4", "rb") as video, open("mask.png", "rb") as mask:
        response = requests.post(
            "http://localhost:8000/propagate",
            files={"video": video, "mask": mask},
            data={
                "start_frame": 0,
                "enable_smoothing": "true",
                "close_radius": 51,
                "dilate_radius": 8,
                "visualize": "false"
            }
        )
    with open("output.mp4", "wb") as f:
        f.write(response.content)
"""
# import debugpy
# debugpy.listen(("0.0.0.0", 7780))
# print("Debugger attached, running…")
# debugpy.wait_for_client()  # 连接前阻塞，可选

import argparse
import io
import logging
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

# Configure logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("cowtracker_api")

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

# Import CoWTracker and utilities
from cowtracker import CoWTracker
from cowtracker.utils.padding import (
    apply_padding,
    compute_padding_params,
    remove_padding_and_scale_back,
)


# ============================================================
# ---------------------- Global State ------------------------
# ============================================================
# Global model instance (loaded once at startup)
GLOBAL_MODEL = None
GLOBAL_DEVICE = None
GLOBAL_DTYPE = None
MODEL_CHECKPOINT = None


def initialize_model(checkpoint_path: Optional[str] = None):
    """Initialize and load the CoWTracker model once at startup."""
    global GLOBAL_MODEL, GLOBAL_DEVICE, GLOBAL_DTYPE, MODEL_CHECKPOINT

    GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GLOBAL_DTYPE = torch.float16 if GLOBAL_DEVICE == "cuda" else torch.float32
    MODEL_CHECKPOINT = checkpoint_path

    logger.info(f"{'='*60}")
    logger.info(f"Initializing CoWTracker model on {GLOBAL_DEVICE}...")
    logger.info(f"{'='*60}")

    try:
        GLOBAL_MODEL = CoWTracker.from_checkpoint(
            MODEL_CHECKPOINT,  # Downloads from HuggingFace by default if None
            device=GLOBAL_DEVICE,
            dtype=GLOBAL_DTYPE,
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load CoWTracker: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting CoWTracker FastAPI Service...")
    initialize_model()
    yield
    # Shutdown
    logger.info("Shutting down CoWTracker FastAPI Service...")


# Create FastAPI app with lifespan
app = FastAPI(
    title="CoWTracker Mask Propagation API",
    description="REST API for propagating masks through videos using CoWTracker DENSE tracking",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# ---------------------- ffmpeg helpers ----------------------
# ============================================================
def run_cmd(cmd: List[str]) -> str:
    """Run a shell command and return stdout."""
    import subprocess
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(cmd)}\n\nstdout:\n{p.stdout}\n\nstderr:\n{p.stderr}"
        )
    return p.stdout


def probe_video(video_path: str):
    """Get video metadata: duration, fps, width, height."""
    out = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            video_path,
        ]
    )
    data = eval(out.replace("null", "None").replace("true", "True").replace("false", "False"))
    stream = data["streams"][0]
    w = int(stream["width"])
    h = int(stream["height"])
    duration = float(data["format"]["duration"])

    def parse_rate(s: str) -> float:
        if not s or s == "0/0":
            return 0.0
        num, den = s.split("/")
        return float(num) / float(den)

    fps = parse_rate(stream.get("avg_frame_rate", "")) or parse_rate(stream.get("r_frame_rate", "")) or 24.0
    return duration, float(fps), w, h


def extract_all_frames(video_path: str, fps: float, out_dir: str) -> None:
    """Extract all frames from video to a directory."""
    os.makedirs(out_dir, exist_ok=True)
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vf",
            f"fps={fps}",
            "-start_number",
            "0",
            os.path.join(out_dir, "%06d.png"),
        ]
    )


def encode_masks_to_video(
    masks: List[np.ndarray],
    fps: float,
    output_path: str,
) -> str:
    """Encode a list of mask arrays to a video file."""
    if not masks:
        raise ValueError("No masks to encode")

    h, w = masks[0].shape[:2]
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=5,
        pixelformat="yuv420p",
        ffmpeg_params=["-preset", "fast", "-crf", "23"]
    )

    try:
        for mask in masks:
            # Ensure RGB format for video encoding
            if mask.ndim == 2:
                mask_rgb = np.stack([mask, mask, mask], axis=-1)
            else:
                mask_rgb = mask
            writer.append_data(mask_rgb.astype(np.uint8))
    finally:
        writer.close()

    return output_path


# ============================================================
# ---------------------- Mask Post-Processing ----------------
# ============================================================
def close_then_dilate(mask, close_radius=6, dilate_radius=8):
    """Apply morphological closing followed by dilation to a mask."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8) * 255

    k1 = 2 * close_radius + 1
    k2 = 2 * dilate_radius + 1
    ker1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))
    ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker1)
    mask = cv2.dilate(mask, ker2, iterations=1)
    return mask


# ============================================================
# ---------------------- Core Tracking Function --------------
# ============================================================
def propagate_mask_with_cowtracker(
    frames: List[np.ndarray],
    initial_mask: np.ndarray,
    start_frame: int = 0,
    enable_temporal_smoothing: bool = True,
    visualize_tracks: bool = False,
    vis_output_path: Optional[str] = None,
    close_radius: int = 6,
    dilate_radius: int = 8,
) -> List[np.ndarray]:
    """
    Propagate a mask through video frames using CoWTracker DENSE tracking.
    Uses the global model instance.
    """
    if GLOBAL_MODEL is None:
        raise RuntimeError("Model not initialized. Call initialize_model() first.")

    num_frames = len(frames)
    H, W = frames[0].shape[:2]

    logger.info(f"{'='*60}")
    logger.info(f"CoWTracker DENSE Tracking: {num_frames} frames @ {H}x{W}")
    logger.info(f"Query frame: {start_frame}")
    logger.info(f"{'='*60}")

    # Prepare video tensor for CoWTracker
    frames_np = np.stack(frames)  # [T, H, W, 3]
    frames_tensor = torch.tensor(frames_np).unsqueeze(0).to(GLOBAL_DTYPE)  # [1, T, H, W, 3]

    # CoWTracker expects [B, T, C, H, W]
    video_tensor = frames_tensor.permute(0, 1, 4, 2, 3)  # [1, T, 3, H, W]
    T = video_tensor.shape[1]

    # Configure inference size and padding
    inf_H, inf_W = 336, 560
    skip_upscaling = True

    # Compute padding parameters
    padding_info = compute_padding_params(
        H, W, inf_H, inf_W, skip_upscaling=skip_upscaling
    )

    logger.info(f"Original size: {H}x{W}")
    logger.info(f"Inference size: {inf_H}x{inf_W}")
    logger.info(f"Scale factor: {padding_info['scale']:.4f}")

    # Initialize output tensors
    traj_maps_e = torch.zeros(
        (1, T, inf_H, inf_W, 2), dtype=torch.float32, device="cpu"
    )
    visconf_maps_e = torch.zeros(
        (1, T, inf_H, inf_W), dtype=torch.float32, device="cpu"
    )

    query_frame = start_frame

    start = time.time()
    with torch.no_grad():
        # Forward pass
        if query_frame < T - 1:
            logger.info(f"Forward pass: frames {query_frame} -> {T-1}")
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                forward_video = video_tensor[0, query_frame:]
                forward_video_padded = apply_padding(forward_video, padding_info).to(GLOBAL_DEVICE)

                predictions = GLOBAL_MODEL.forward(
                    video=forward_video_padded,
                    queries=None,
                )

                tracks_dense = predictions["track"][0]
                visibility_dense = predictions["vis"][0]
                confidence_dense = predictions["conf"][0]

                T_forward = tracks_dense.shape[0]
                traj_maps_e[0, query_frame : query_frame + T_forward] = tracks_dense.cpu()
                visconf_maps_e[0, query_frame : query_frame + T_forward] = (
                    visibility_dense * confidence_dense
                ).cpu()

        # Backward pass
        if query_frame > 0:
            logger.info(f"Backward pass: frames {query_frame} -> 0")
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                backward_video = video_tensor[0, : query_frame + 1].flip([0])
                backward_video_padded = apply_padding(backward_video, padding_info).to(GLOBAL_DEVICE)

                predictions = GLOBAL_MODEL.forward(
                    video=backward_video_padded,
                    queries=None,
                )

                tracks_dense = predictions["track"][0]
                visibility_dense = predictions["vis"][0]
                confidence_dense = predictions["conf"][0]

                backward_tracks = tracks_dense.flip([0]).cpu()
                backward_visconf = (visibility_dense * confidence_dense).flip([0]).cpu()

                end_idx = query_frame if query_frame < T - 1 else query_frame + 1
                traj_maps_e[0, :end_idx] = backward_tracks[:end_idx]
                visconf_maps_e[0, :end_idx] = backward_visconf[:end_idx]

    # Remove padding and scale back
    logger.info(f"Unpadding and scaling back to {H}x{W}")
    tracks_final, _, confidence_final = remove_padding_and_scale_back(
        traj_maps_e[0],
        torch.ones_like(visconf_maps_e[0]),
        visconf_maps_e[0],
        padding_info,
    )

    end = time.time()
    logger.info(f"Full tracking time: {end - start:.2f}s")

    # Convert to numpy
    tracks_np = tracks_final.permute(1, 2, 0, 3).numpy()  # [H, W, T, 2]
    conf_np = confidence_final.permute(1, 2, 0).numpy()  # [H, W, T]

    # Extract pixels from initial mask
    mask_pixels = np.argwhere(initial_mask > 127)
    logger.info(f"Initial mask has {len(mask_pixels)} pixels")

    if len(mask_pixels) == 0:
        logger.warning("Initial mask is empty!")
        return [np.zeros((H, W), dtype=np.uint8) for _ in range(num_frames)]

    # Propagate mask for each frame
    masks = []
    vis_frames = []

    for t in range(num_frames):
        mask_t = np.zeros((H, W), dtype=np.uint8)
        placed = 0

        # For frame 0, use initial mask directly
        if t == 0:
            mask_t = initial_mask.copy()
            placed = (mask_t > 127).sum()
            logger.info(f"Frame {t}: Using initial mask directly ({placed} pixels)")
        else:
            for my, mx in mask_pixels:
                tracked_x, tracked_y = tracks_np[my, mx, t]
                ix, iy = int(tracked_x), int(tracked_y)
                if 0 <= ix < W and 0 <= iy < H:
                    if conf_np[my, mx, t] > 0.1:
                        mask_t[iy, ix] = 255
                        placed += 1

        # Morphological smoothing (skip for frame 0)
        if enable_temporal_smoothing and placed > 0 and t != 0:
            mask_t = close_then_dilate(mask_t, close_radius=close_radius, dilate_radius=dilate_radius)

        masks.append(mask_t)

        # Create visualization if requested
        if visualize_tracks:
            vis_frame = frames[t].copy()
            try:
                import matplotlib.colormaps as cm
                cmap = cm["gist_rainbow"]
                for i, (my, mx) in enumerate(mask_pixels[::20]):
                    color = np.array(cmap(i / len(mask_pixels))[:3]) * 255
                    for ft in range(num_frames):
                        tx, ty = tracks_np[my, mx, ft]
                        if 0 <= int(tx) < W and 0 <= int(ty) < H:
                            cv2.circle(vis_frame, (int(tx), int(ty)), 1, color, -1)
            except ImportError:
                pass
            vis_frames.append(vis_frame)

    # Save visualization
    if visualize_tracks and vis_output_path and vis_frames:
        logger.info(f"Saving visualization to {vis_output_path}")
        writer = imageio.get_writer(vis_output_path, fps=15, codec="libx264", quality=5)
        for vf in vis_frames:
            writer.append_data(vf)
        writer.close()

    return masks


def load_mask_from_bytes(mask_bytes: bytes, target_size: Optional[tuple] = None) -> np.ndarray:
    """Load a mask from bytes and convert to grayscale."""
    mask = Image.open(io.BytesIO(mask_bytes))
    if target_size is not None:
        mask = mask.resize(target_size, resample=Image.NEAREST)
    mask = np.array(mask.convert("L"))
    return (mask > 127).astype(np.uint8) * 255


# ============================================================
# ---------------------- API Endpoints -----------------------
# ============================================================
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "CoWTracker Mask Propagation API",
        "version": "1.0.0",
        "status": "ready",
        "model_loaded": GLOBAL_MODEL is not None,
        "device": GLOBAL_DEVICE,
        "endpoints": {
            "/propagate": "POST - Propagate mask through video",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation (Swagger UI)"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": GLOBAL_MODEL is not None,
        "device": GLOBAL_DEVICE
    }


@app.post("/propagate")
async def propagate_mask_endpoint(
    video: UploadFile = File(..., description="Input video file (mp4, avi, mov, etc.)"),
    mask: UploadFile = File(..., description="Initial mask image (png, jpg, etc.)"),
    start_frame: int = Form(0, description="Frame index where mask is defined"),
    end_frame: Optional[int] = Form(None, description="End frame index (default: end of video)"),
    enable_smoothing: bool = Form(True, description="Apply morphological smoothing"),
    close_radius: int = Form(51, description="Morphological closing radius"),
    dilate_radius: int = Form(8, description="Morphological dilation radius"),
    visualize: bool = Form(False, description="Create track visualization"),
):
    """
    Propagate a mask through a video using CoWTracker.

    Returns the output mask video as a file download.
    """
    if GLOBAL_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # Create temporary directory for this request
    request_id = uuid.uuid4().hex[:8]
    tmp_dir = tempfile.mkdtemp(prefix=f"cowtracker_api_{request_id}_")

    try:
        # Save uploaded files
        video_path = os.path.join(tmp_dir, f"input_{video.filename}")
        mask_path = os.path.join(tmp_dir, f"mask_{mask.filename}")

        with open(video_path, "wb") as f:
            f.write(await video.read())
        with open(mask_path, "wb") as f:
            f.write(await mask.read())

        logger.info(f"[{request_id}] Processing request...")
        logger.info(f"[{request_id}] Video: {video.filename}")
        logger.info(f"[{request_id}] Mask: {mask.filename}")

        # Get video metadata
        duration, fps, width, height = probe_video(video_path)
        total_frames = int(duration * fps)

        if end_frame is None:
            end_frame = total_frames - 1

        logger.info(f"[{request_id}] Video: {total_frames} frames @ {fps} fps, {width}x{height}")
        logger.info(f"[{request_id}] Processing frames {start_frame} -> {end_frame}")

        # Extract frames
        frames_dir = os.path.join(tmp_dir, "frames")
        logger.info(f"[{request_id}] Extracting frames...")
        extract_all_frames(video_path, fps, frames_dir)

        # Load frames
        frames = []
        for i in range(start_frame, end_frame + 1):
            frame_path = os.path.join(frames_dir, f"{i:06d}.png")
            if os.path.exists(frame_path):
                frames.append(np.array(Image.open(frame_path).convert("RGB")))

        if not frames:
            raise HTTPException(
                status_code=400,
                detail=f"No frames found in range {start_frame}-{end_frame}"
            )

        logger.info(f"[{request_id}] Loaded {len(frames)} frames")

        # Load initial mask from file
        initial_mask = load_mask_from_bytes(
            await mask.read() if False else open(mask_path, "rb").read(),
            (width, height)
        )
        logger.info(f"[{request_id}] Loaded initial mask: {initial_mask.shape}")

        # Run CoWTracker tracking
        masks = propagate_mask_with_cowtracker(
            frames=frames,
            initial_mask=initial_mask,
            start_frame=0,
            enable_temporal_smoothing=enable_smoothing,
            visualize_tracks=visualize,
            vis_output_path=os.path.join(tmp_dir, "visualization.mp4") if visualize else None,
            close_radius=close_radius,
            dilate_radius=dilate_radius,
        )

        # Encode masks to video
        output_path = os.path.join(tmp_dir, "output.mp4")
        logger.info(f"[{request_id}] Encoding output video...")
        encode_masks_to_video(masks, fps, output_path)

        logger.info(f"[{request_id}] Done! Returning output video")

        # Return the video file
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"cowtracker_output_{request_id}.mp4",
            background=None  # Will be cleaned up after response
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Schedule cleanup (don't do it immediately as FileResponse needs the file)
        # The temp file will be cleaned up by the OS eventually, or you can implement
        # a background task to clean it up after a delay
        pass


# ============================================================
# ---------------------- Main Entry Point --------------------
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="CoWTracker Mask Propagation FastAPI Service"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    # Set global checkpoint path
    global MODEL_CHECKPOINT
    MODEL_CHECKPOINT = args.checkpoint

    # Run the server
    uvicorn.run(
        "cowtracker_fastapi:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
