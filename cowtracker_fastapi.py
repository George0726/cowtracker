#!/usr/bin/env python3
"""
CoWTracker Mask Propagation FastAPI Service (PATH INPUT VERSION)

- Input: server-local video_path + mask_path (no UploadFile)
- Output: mp4 mask video
- Optimizations:
  * json.loads for ffprobe
  * mask read once
  * CPU-safe autocast
  * GPU concurrency lock
  * BackgroundTasks cleanup temp dir

Run:
  python cowtracker_fastapi_paths.py --host 0.0.0.0 --port 8000

Request example:
  curl -X POST "http://localhost:8000/propagate_path" \
    -H "Content-Type: application/json" \
    -d '{
      "video_path": "/path/to/video.mp4",
      "mask_path": "/path/to/mask.png",
      "start_frame": 0,
      "end_frame": 120,
      "enable_smoothing": true,
      "close_radius": 51,
      "dilate_radius": 8,
      "visualize": false
    }' --output out.mp4
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import logging
import os
import time
from contextlib import asynccontextmanager, nullcontext
from typing import List, Optional, Tuple

import numpy as np
import torch
import uvicorn
import cv2
import imageio.v2 as imageio
import subprocess

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

# Import CoWTracker and utilities
from cowtracker import CoWTracker
from cowtracker.utils.padding import (
    apply_padding,
    compute_padding_params,
    remove_padding_and_scale_back,
)

# ---------------- logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cowtracker_api")

# ---------------- global model state ----------------
GLOBAL_MODEL = None
GLOBAL_DEVICE = None
GLOBAL_DTYPE = None
MODEL_CHECKPOINT = None

GPU_LOCK = asyncio.Lock()


def initialize_model(checkpoint_path: Optional[str] = None):
    """Load the CoWTracker model once at startup."""
    global GLOBAL_MODEL, GLOBAL_DEVICE, GLOBAL_DTYPE, MODEL_CHECKPOINT

    MODEL_CHECKPOINT = checkpoint_path
    GLOBAL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GLOBAL_DTYPE = torch.float16 if GLOBAL_DEVICE == "cuda" else torch.float32

    logger.info("=" * 60)
    logger.info(f"Initializing CoWTracker model on {GLOBAL_DEVICE} ...")
    logger.info("=" * 60)

    GLOBAL_MODEL = CoWTracker.from_checkpoint(
        MODEL_CHECKPOINT,  # None -> download default
        device=GLOBAL_DEVICE,
        dtype=GLOBAL_DTYPE,
    )
    GLOBAL_MODEL.eval()
    logger.info("Model loaded successfully!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting CoWTracker FastAPI Service...")
    initialize_model(MODEL_CHECKPOINT)
    yield
    logger.info("Shutting down CoWTracker FastAPI Service...")


app = FastAPI(
    title="CoWTracker Mask Propagation API (Path Input)",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# ---------------------- helpers -----------------------------
# ============================================================
def run_cmd(cmd: List[str]) -> str:
    """Run command and return stdout, raise on non-zero."""
    import subprocess

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(cmd)}\n\nstdout:\n{p.stdout}\n\nstderr:\n{p.stderr}"
        )
    return p.stdout


def probe_video(video_path):
    out = run_cmd([
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ])

    data = json.loads(out)

    stream = data["streams"][0]

    w = int(stream["width"])
    h = int(stream["height"])

    duration = float(data["format"]["duration"])

    def parse_rate(s):
        if not s or s == "0/0":
            return 0.0
        a, b = s.split("/")
        return float(a) / float(b)

    fps = parse_rate(stream.get("avg_frame_rate"))

    total_frames = int(stream.get("nb_read_frames", 0))

    return duration, fps, w, h, total_frames


def ffprobe_wh(path: str):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        path
    ]
    out = subprocess.check_output(cmd)
    info = json.loads(out.decode())
    st = info["streams"][0]
    return int(st["width"]), int(st["height"])

def read_rgb_frames(video_path: str, start_frame: int = 0, end_frame: int | None = None):
    """
    Read RGB frames from video using ffmpeg pipe.

    Returns:
        List[np.ndarray]  (H,W,3) uint8
    """

    w, h = ffprobe_wh(video_path)

    if end_frame is None:
        vf = f"select='gte(n\\,{start_frame})'"
    else:
        vf = f"select='between(n\\,{start_frame}\\,{end_frame})'"

    cmd = [
        "ffmpeg",
        "-v", "error",
        "-nostdin",
        "-i", video_path,
        "-vf", vf,
        "-vsync", "0",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_bytes = w * h * 3
    frames = []

    try:
        while True:
            buf = p.stdout.read(frame_bytes)
            if len(buf) != frame_bytes:
                break

            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            frames.append(frame)

    finally:
        p.stdout.close()
        stderr = p.stderr.read()
        p.stderr.close()
        ret = p.wait()

        if ret != 0:
            raise RuntimeError(
                f"ffmpeg failed:\n{stderr.decode('utf-8', errors='ignore')}"
            )

    return frames



def load_mask_gray(mask_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Load mask image -> grayscale uint8 {0,255}, resized to target_size (W,H)."""
    im = Image.open(mask_path)
    if im.size != target_size:
        im = im.resize(target_size, resample=Image.NEAREST)
    m = np.array(im.convert("L"), dtype=np.uint8)
    return (m > 127).astype(np.uint8) * 255


def encode_masks_to_video(
    masks: List[np.ndarray],
    fps: float,
    output_path: str,
    preset: str = "ultrafast",
    crf: int = 30,
    codec: str = "libx264",
    as_rgb: bool = True,
) -> str:
    """Encode masks to mp4 using ffmpeg rawvideo pipe."""
    if not masks:
        raise ValueError("No masks to encode")

    h, w = masks[0].shape[:2]
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    input_pix_fmt = "rgb24" if as_rgb else "gray"
    cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        "-f", "rawvideo",
        "-pix_fmt", input_pix_fmt,
        "-s", f"{w}x{h}",
        "-r", f"{float(fps):.6f}",
        "-i", "pipe:0",
        "-an",
        "-c:v", codec,
        "-preset", preset,
        "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for mask in masks:
            mask_gray = mask if mask.ndim == 2 else mask[..., 0]
            if as_rgb:
                mask_rgb = np.repeat(mask_gray[:, :, None], 3, axis=2)
                p.stdin.write(np.ascontiguousarray(mask_rgb, dtype=np.uint8).tobytes())
            else:
                p.stdin.write(np.ascontiguousarray(mask_gray, dtype=np.uint8).tobytes())
        p.stdin.close()
        stderr = p.stderr.read()
        ret = p.wait()
        if ret != 0:
            raise RuntimeError(
                f"ffmpeg encode failed:\n{stderr.decode('utf-8', errors='ignore')}"
            )
    finally:
        if p.stdin and not p.stdin.closed:
            p.stdin.close()
        if p.stderr:
            p.stderr.close()

    return output_path


def close_then_dilate(
    mask: np.ndarray,
    close_radius: int = 6,
    dilate_radius: int = 8,
    close_kernel: Optional[np.ndarray] = None,
    dilate_kernel: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Morph close then dilate (expects 2D mask)."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8) * 255

    if close_kernel is None and int(close_radius) > 0:
        k1 = 2 * int(close_radius) + 1
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))
    if dilate_kernel is None and int(dilate_radius) > 0:
        k2 = 2 * int(dilate_radius) + 1
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))

    if close_kernel is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    if dilate_kernel is not None:
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)
    return mask


def _propagate_mask_single_window(
    frames: List[np.ndarray],
    initial_mask: np.ndarray,
    start_frame: int = 0,
    enable_temporal_smoothing: bool = True,
    visualize_tracks: bool = False,
    vis_output_path: Optional[str] = None,
    close_radius: int = 6,
    dilate_radius: int = 8,
    parallel_smoothing: bool = False,
    smoothing_workers: int = 0,
    smoothing_chunk_size: int = 16,
) -> List[np.ndarray]:
    """
    Propagate mask through frames using GLOBAL_MODEL (CoWTracker).
    frames: list of RGB uint8 [H,W,3]
    initial_mask: uint8 [H,W] {0,255} at start_frame (usually 0 within the provided chunk)
    """
    if GLOBAL_MODEL is None:
        raise RuntimeError("Model not initialized")

    num_frames = len(frames)
    H, W = frames[0].shape[:2]
    logger.info(f"CoWTracker tracking chunk: {num_frames} frames @ {H}x{W}, query={start_frame}")

    # video tensor [1,T,3,H,W]
    frames_np = np.stack(frames)  # [T,H,W,3]
    video_tensor = torch.from_numpy(frames_np).unsqueeze(0).to(GLOBAL_DTYPE)  # [1,T,H,W,3]
    video_tensor = video_tensor.permute(0, 1, 4, 2, 3).contiguous()  # [1,T,3,H,W]
    T = video_tensor.shape[1]

    # padding setup
    inf_H, inf_W = 336, 560
    padding_info = compute_padding_params(H, W, inf_H, inf_W, skip_upscaling=True)

    # store device (keep on GPU if available to reduce transfers)
    store_device = GLOBAL_DEVICE if GLOBAL_DEVICE == "cuda" else "cpu"
    traj_maps_e = torch.zeros((1, T, inf_H, inf_W, 2), dtype=torch.float32, device=store_device)
    visconf_maps_e = torch.zeros((1, T, inf_H, inf_W), dtype=torch.float32, device=store_device)

    # autocast safe
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if GLOBAL_DEVICE == "cuda"
        else nullcontext()
    )

    query_frame = int(start_frame)

    t0 = time.time()
    with torch.no_grad():
        # forward
        if query_frame < T - 1:
            with autocast_ctx:
                forward_video = video_tensor[0, query_frame:]  # [Tf,3,H,W]
                forward_video_padded = apply_padding(forward_video, padding_info).to(GLOBAL_DEVICE)

                pred = GLOBAL_MODEL.forward(video=forward_video_padded, queries=None)
                tracks_dense = pred["track"][0]           # [Tf,inf_H,inf_W,2] (device=GLOBAL_DEVICE)
                visibility_dense = pred["vis"][0]
                confidence_dense = pred["conf"][0]

                Tf = tracks_dense.shape[0]
                traj_maps_e[0, query_frame: query_frame + Tf] = tracks_dense.to(store_device)
                visconf_maps_e[0, query_frame: query_frame + Tf] = (visibility_dense * confidence_dense).to(store_device)

        # backward
        if query_frame > 0:
            with autocast_ctx:
                backward_video = video_tensor[0, : query_frame + 1].flip([0])
                backward_video_padded = apply_padding(backward_video, padding_info).to(GLOBAL_DEVICE)

                pred = GLOBAL_MODEL.forward(video=backward_video_padded, queries=None)
                tracks_dense = pred["track"][0]
                visibility_dense = pred["vis"][0]
                confidence_dense = pred["conf"][0]

                backward_tracks = tracks_dense.flip([0]).to(store_device)
                backward_visconf = (visibility_dense * confidence_dense).flip([0]).to(store_device)

                end_idx = query_frame if query_frame < T - 1 else query_frame + 1
                traj_maps_e[0, :end_idx] = backward_tracks[:end_idx]
                visconf_maps_e[0, :end_idx] = backward_visconf[:end_idx]

    # remove padding and scale back (do on CPU to be safe)
    traj_cpu = traj_maps_e[0].to("cpu")
    visconf_cpu = visconf_maps_e[0].to("cpu")

    tracks_final, _, confidence_final = remove_padding_and_scale_back(
        traj_cpu,
        torch.ones_like(visconf_cpu),  # dummy vis
        visconf_cpu,
        padding_info,
    )

    logger.info(f"Tracking done in {time.time() - t0:.2f}s")

    # tracks_final: [T,H,W,2]?  (based on your original usage you permute later)
    tracks_np = tracks_final.permute(1, 2, 0, 3).numpy()   # [H,W,T,2]
    conf_np = confidence_final.permute(1, 2, 0).numpy()    # [H,W,T]

    mask_pixels = np.argwhere(initial_mask > 127)  # [N,2] (y,x)
    if len(mask_pixels) == 0:
        logger.warning("Initial mask empty")
        return [np.zeros((H, W), dtype=np.uint8) for _ in range(num_frames)]

    t1 = time.time()
    # Vectorize pixel projection to avoid Python loops over mask pixels.
    ys0 = mask_pixels[:, 0]
    xs0 = mask_pixels[:, 1]
    tracks_seed = tracks_np[ys0, xs0]  # [N,T,2]
    conf_seed = conf_np[ys0, xs0]      # [N,T]

    masks_out: List[np.ndarray] = [initial_mask.copy()]
    conf_thr = 0.1

    projection_s = 0.0
    morph_s = 0.0

    # First build raw projected masks; run smoothing as a separate phase.
    for t in range(1, num_frames):
        tp0 = time.time()
        xy_t = tracks_seed[:, t]  # [N,2]
        ix = xy_t[:, 0].astype(np.int32)
        iy = xy_t[:, 1].astype(np.int32)

        valid = (
            (ix >= 0)
            & (ix < W)
            & (iy >= 0)
            & (iy < H)
            & (conf_seed[:, t] > conf_thr)
        )

        mask_t = np.zeros((H, W), dtype=np.uint8)
        if np.any(valid):
            mask_t[iy[valid], ix[valid]] = 255
        projection_s += time.time() - tp0

        masks_out.append(mask_t)

    if enable_temporal_smoothing and num_frames > 1:
        tm0 = time.time()
        n_masks = num_frames - 1
        requested_workers = int(smoothing_workers)
        workers = requested_workers if requested_workers > 0 else min(os.cpu_count() or 1, 8)
        workers = max(1, min(workers, n_masks))
        chunk_size = max(1, int(smoothing_chunk_size))

        if parallel_smoothing and workers > 1:
            logger.info(
                f"Running parallel smoothing: workers={workers}, chunk_size={chunk_size}, frames={n_masks}"
            )
            close_kernel = None
            dilate_kernel = None
            if int(close_radius) > 0:
                k1 = 2 * int(close_radius) + 1
                close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))
            if int(dilate_radius) > 0:
                k2 = 2 * int(dilate_radius) + 1
                dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))

            chunks = []
            for start in range(1, num_frames, chunk_size):
                end = min(start + chunk_size, num_frames)
                chunks.append((start, end))

            def _smooth_chunk(start_idx: int, end_idx: int):
                out_chunk = []
                for i in range(start_idx, end_idx):
                    out_chunk.append(
                        close_then_dilate(
                            masks_out[i],
                            close_kernel=close_kernel,
                            dilate_kernel=dilate_kernel,
                        )
                    )
                return start_idx, out_chunk

            prev_cv_threads = cv2.getNumThreads()
            cv2.setNumThreads(1)
            try:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = [ex.submit(_smooth_chunk, s, e) for s, e in chunks]
                    for fut in futures:
                        start, out_chunk = fut.result()
                        masks_out[start:start + len(out_chunk)] = out_chunk
            except Exception as e:
                logger.warning(f"Parallel smoothing failed ({e}); falling back to serial smoothing")
                for i in range(1, num_frames):
                    masks_out[i] = close_then_dilate(
                        masks_out[i],
                        close_kernel=close_kernel,
                        dilate_kernel=dilate_kernel,
                    )
            finally:
                cv2.setNumThreads(prev_cv_threads)
        else:
            close_kernel = None
            dilate_kernel = None
            if int(close_radius) > 0:
                k1 = 2 * int(close_radius) + 1
                close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))
            if int(dilate_radius) > 0:
                k2 = 2 * int(dilate_radius) + 1
                dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
            for i in range(1, num_frames):
                masks_out[i] = close_then_dilate(
                    masks_out[i],
                    close_kernel=close_kernel,
                    dilate_kernel=dilate_kernel,
                )
        morph_s = time.time() - tm0

    logger.info(
        f"Post-Processing done in {time.time() - t1:.2f}s "
        f"(projection={projection_s:.2f}s, morph={morph_s:.2f}s)"
    )

    # optional visualization (kept minimal)
    if visualize_tracks and vis_output_path:
        try:
            writer = imageio.get_writer(vis_output_path, fps=15, codec="libx264", quality=5)
            for fr, m in zip(frames, masks_out):
                vis = fr.copy()
                ys, xs = np.where(m > 127)
                # draw a few points
                for y, x in zip(ys[:: max(1, len(ys)//800)], xs[:: max(1, len(xs)//800)]):
                    cv2.circle(vis, (int(x), int(y)), 1, (255, 0, 0), -1)
                writer.append_data(vis)
            writer.close()
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")

    return masks_out


def propagate_mask_with_cowtracker(
    frames: List[np.ndarray],
    initial_mask: np.ndarray,
    start_frame: int = 0,
    enable_temporal_smoothing: bool = True,
    visualize_tracks: bool = False,
    vis_output_path: Optional[str] = None,
    close_radius: int = 6,
    dilate_radius: int = 8,
    parallel_smoothing: bool = False,
    smoothing_workers: int = 0,
    smoothing_chunk_size: int = 16,
    tile_size: int = 0,
) -> List[np.ndarray]:
    """
    Propagate with optional recurrent sliding windows.

    If tile_size > 1 and tile_size < num_frames, run windows with 1-frame overlap:
    - first window: [0, tile_size)
    - next window starts at previous window's last frame
    - last output mask of previous window is reused as initial mask
    """
    num_frames = len(frames)
    if num_frames == 0:
        return []

    # Single-shot path (original behavior)
    if tile_size <= 1 or tile_size >= num_frames:
        return _propagate_mask_single_window(
            frames=frames,
            initial_mask=initial_mask,
            start_frame=start_frame,
            enable_temporal_smoothing=enable_temporal_smoothing,
            visualize_tracks=visualize_tracks,
            vis_output_path=vis_output_path,
            close_radius=close_radius,
            dilate_radius=dilate_radius,
            parallel_smoothing=parallel_smoothing,
            smoothing_workers=smoothing_workers,
            smoothing_chunk_size=smoothing_chunk_size,
        )

    window_size = int(tile_size)
    logger.info(f"Sliding-window propagation enabled: tile_size={window_size}, total_frames={num_frames}")

    all_masks: List[np.ndarray] = []
    window_start = 0
    current_init_mask = initial_mask
    window_idx = 0

    while window_start < num_frames:
        window_end = min(window_start + window_size, num_frames)
        window_frames = frames[window_start:window_end]
        if not window_frames:
            break

        logger.info(
            f"Window {window_idx}: frames [{window_start}, {window_end - 1}] "
            f"(len={len(window_frames)})"
        )

        win_masks = _propagate_mask_single_window(
            frames=window_frames,
            initial_mask=current_init_mask,
            start_frame=0,
            enable_temporal_smoothing=enable_temporal_smoothing,
            visualize_tracks=False,  # skip per-window vis to avoid repeated work
            vis_output_path=None,
            close_radius=close_radius,
            dilate_radius=dilate_radius,
            parallel_smoothing=parallel_smoothing,
            smoothing_workers=smoothing_workers,
            smoothing_chunk_size=smoothing_chunk_size,
        )

        if window_start == 0:
            all_masks.extend(win_masks)
        else:
            # Drop first frame because it overlaps previous window's last frame.
            all_masks.extend(win_masks[1:])

        if window_end >= num_frames:
            break

        current_init_mask = win_masks[-1]
        window_start = window_end - 1  # 1-frame overlap for recurrent input
        window_idx += 1

    return all_masks


# ============================================================
# ---------------------- API schema --------------------------
# ============================================================
class PropagatePathRequest(BaseModel):
    video_path: str
    mask_path: str
    output_path: str

    start_frame: int = 0
    end_frame: Optional[int] = None
    mask_idx: int = 0
    enable_smoothing: bool = True
    close_radius: int = 51
    dilate_radius: int = 8
    visualize: bool = False
    parallel_smoothing: bool = False
    smoothing_workers: int = 0
    smoothing_chunk_size: int = 16
    tile_size: int = 0
    clear_cuda_cache: bool = True
    encode_preset: str = "ultrafast"
    encode_crf: int = 30
    encode_codec: str = "libx264"
    encode_as_rgb: bool = True


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": GLOBAL_MODEL is not None,
        "device": GLOBAL_DEVICE,
        "dtype": str(GLOBAL_DTYPE),
    }


@app.post("/propagate_path")
async def propagate_mask_from_paths(req: PropagatePathRequest):
    req_t0 = time.time()
    decode_s = 0.0
    infer_s = 0.0
    save_s = 0.0

    if GLOBAL_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    if not os.path.exists(req.video_path):
        raise HTTPException(400, f"video_path not found: {req.video_path}")

    if not os.path.exists(req.mask_path):
        raise HTTPException(400, f"mask_path not found: {req.mask_path}")

    output_path = req.output_path
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    duration, fps, width, height, total_frames = probe_video(req.video_path)

    end_frame = req.end_frame if req.end_frame is not None else (total_frames - 1)

    logger.info(f"video={req.video_path}")
    logger.info(f"mask={req.mask_path}")
    logger.info(f"output={output_path}")

    # decode frames
    t_decode0 = time.time()
    frames = read_rgb_frames(req.video_path, req.start_frame, end_frame)
    decode_s = time.time() - t_decode0

    if not frames:
        raise HTTPException(400, "No frames decoded")

    initial_mask = load_mask_gray(req.mask_path, (width, height))
    mask_local_idx = int(req.mask_idx) - int(req.start_frame)
    if mask_local_idx < 0 or mask_local_idx >= len(frames):
        raise HTTPException(
            400,
            (
                f"mask_idx out of decoded range: mask_idx={req.mask_idx}, "
                f"decoded frame range=[{req.start_frame}, {req.start_frame + len(frames) - 1}]"
            ),
        )

    t_infer0 = time.time()
    async with GPU_LOCK:
        if mask_local_idx == 0:
            masks = propagate_mask_with_cowtracker(
                frames=frames,
                initial_mask=initial_mask,
                start_frame=0,
                enable_temporal_smoothing=req.enable_smoothing,
                visualize_tracks=req.visualize,
                vis_output_path=None,
                close_radius=req.close_radius,
                dilate_radius=req.dilate_radius,
                parallel_smoothing=req.parallel_smoothing,
                smoothing_workers=req.smoothing_workers,
                smoothing_chunk_size=req.smoothing_chunk_size,
                tile_size=req.tile_size,
            )
        else:
            logger.info(
                f"Using split propagation from mask_idx={req.mask_idx} "
                f"(local_idx={mask_local_idx} in decoded chunk)"
            )

            # Forward: [mask_idx .. end]
            forward_frames = frames[mask_local_idx:]
            forward_masks = propagate_mask_with_cowtracker(
                frames=forward_frames,
                initial_mask=initial_mask,
                start_frame=0,
                enable_temporal_smoothing=req.enable_smoothing,
                visualize_tracks=False,
                vis_output_path=None,
                close_radius=req.close_radius,
                dilate_radius=req.dilate_radius,
                parallel_smoothing=req.parallel_smoothing,
                smoothing_workers=req.smoothing_workers,
                smoothing_chunk_size=req.smoothing_chunk_size,
                tile_size=req.tile_size,
            )

            # Backward by reversing [0 .. mask_idx] -> [mask_idx .. 0].
            backward_input_frames = frames[: mask_local_idx + 1][::-1]
            backward_masks_reversed = propagate_mask_with_cowtracker(
                frames=backward_input_frames,
                initial_mask=initial_mask,
                start_frame=0,
                enable_temporal_smoothing=req.enable_smoothing,
                visualize_tracks=False,
                vis_output_path=None,
                close_radius=req.close_radius,
                dilate_radius=req.dilate_radius,
                parallel_smoothing=req.parallel_smoothing,
                smoothing_workers=req.smoothing_workers,
                smoothing_chunk_size=req.smoothing_chunk_size,
                tile_size=req.tile_size,
            )
            backward_masks = backward_masks_reversed[::-1]  # [0 .. mask_idx]

            # Merge without duplicating mask_idx frame.
            masks = backward_masks[:-1] + forward_masks
            if len(masks) != len(frames):
                raise HTTPException(
                    500,
                    f"Merged mask length mismatch: got {len(masks)} expected {len(frames)}",
                )
    infer_s = time.time() - t_infer0

    save_t0 = time.time()
    encode_masks_to_video(
        masks,
        fps,
        output_path,
        preset=req.encode_preset,
        crf=req.encode_crf,
        codec=req.encode_codec,
        as_rgb=req.encode_as_rgb,
    )
    save_s = time.time() - save_t0

    logger.info(f"Saved output to {output_path} in {save_s:.2f}s")

    if req.clear_cuda_cache and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        logger.info("CUDA cache cleanup completed")

    total_s = time.time() - req_t0
    logger.info(
        f"Request total time: {total_s:.2f}s "
        f"(decode={decode_s:.2f}s, infer+post={infer_s:.2f}s, save={save_s:.2f}s)"
    )

    return {
        "status": "success",
        "output_path": output_path,
        "frames": len(masks),
        "fps": fps
    }
# ============================================================
# ---------------------- main entry --------------------------
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CoWTracker FastAPI (path input)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument("--reload", action="store_true")

    args = parser.parse_args()
    global MODEL_CHECKPOINT
    MODEL_CHECKPOINT = args.checkpoint

    uvicorn.run(
        "cowtracker_fastapi:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
