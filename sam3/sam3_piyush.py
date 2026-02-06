
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import numpy as np
import argparse
import torch
import pandas as pd
import sys
import gc
from math import floor

import supervision as sv

# GPU Memory Monitoring Utilities
class TensorTracker:
    """Track tensor allocations to identify memory leaks"""
    def __init__(self):
        self.snapshots = []

    def snapshot(self, label=""):
        """Take a snapshot of all GPU tensors"""
        current_tensors = {}
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    tensor_id = id(obj)
                    size_mb = obj.element_size() * obj.nelement() / 1024**2
                    current_tensors[tensor_id] = {
                        'shape': tuple(obj.shape),
                        'dtype': str(obj.dtype),
                        'size_mb': size_mb,
                    }
            except:
                pass
        return current_tensors

    def compare_snapshots(self, before, after, label=""):
        """Compare two snapshots and show what leaked"""
        new_tensors = set(after.keys()) - set(before.keys())

        if new_tensors:
            print(f"\n{'='*70}")
            print(f"[MEMORY LEAK DETECTED - {label}]")
            print(f"{'='*70}")
            total_leaked = 0

            leaked_by_shape = {}
            for tid in new_tensors:
                info = after[tid]
                size_mb = info['size_mb']
                if size_mb > 0.1:  # Only show tensors > 100KB
                    total_leaked += size_mb
                    shape_key = (info['shape'], info['dtype'])
                    if shape_key not in leaked_by_shape:
                        leaked_by_shape[shape_key] = {'count': 0, 'total_mb': 0}
                    leaked_by_shape[shape_key]['count'] += 1
                    leaked_by_shape[shape_key]['total_mb'] += size_mb

            # Sort by total memory
            sorted_leaks = sorted(leaked_by_shape.items(),
                                 key=lambda x: x[1]['total_mb'],
                                 reverse=True)

            print(f"\nNEW TENSORS (not freed from previous iteration):")
            for (shape, dtype), info in sorted_leaks:
                print(f"  Shape: {shape}, dtype: {dtype}")
                print(f"    Count: {info['count']}, Total: {info['total_mb']:.2f} MB")

            print(f"\nTOTAL LEAKED: {total_leaked:.2f} MB")
            print(f"{'='*70}\n")
            return total_leaked
        return 0

def log_variable_memory(variables_dict, label=""):
    """Log memory usage of specific variables"""
    print(f"\n[VARIABLE MEMORY - {label}]")
    for var_name, var in variables_dict.items():
        if var is None:
            continue

        if torch.is_tensor(var):
            size_mb = var.element_size() * var.nelement() / 1024**2
            device = 'GPU' if var.is_cuda else 'CPU'
            print(f"  {var_name}: Tensor {var.shape} on {device}, {size_mb:.2f} MB")
        elif isinstance(var, (list, tuple)):
            for i, item in enumerate(var):
                if torch.is_tensor(item):
                    item_size = item.element_size() * item.nelement() / 1024**2
                    device = 'GPU' if item.is_cuda else 'CPU'
                    if item_size > 0.1:
                        print(f"  {var_name}[{i}]: Tensor {item.shape} on {device}, {item_size:.2f} MB")
        elif isinstance(var, dict):
            for key, item in var.items():
                if torch.is_tensor(item):
                    item_size = item.element_size() * item.nelement() / 1024**2
                    device = 'GPU' if item.is_cuda else 'CPU'
                    if item_size > 0.1:
                        print(f"  {var_name}[{key}]: Tensor {item.shape} on {device}, {item_size:.2f} MB")

def log_gpu_memory(label=""):
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\n[GPU MEMORY - {label}] Allocated: {allocated:.3f} GB, Reserved: {reserved:.3f} GB")
        return allocated, reserved
    return 0, 0

def log_tensor_memory():
    """Log all tensors in GPU memory"""
    if torch.cuda.is_available():
        print("\n[ALL GPU TENSORS]")
        total_size = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    size_mb = obj.element_size() * obj.nelement() / 1024**2
                    if size_mb > 1:
                        print(f"  Tensor: shape={obj.shape}, dtype={obj.dtype}, size={size_mb:.2f} MB")
                        total_size += size_mb
            except:
                pass
        print(f"  Total: {total_size:.2f} MB\n")

try:
    from sam3.model_builder import build_sam3_video_predictor
except ImportError:
    print("CRITICAL ERROR: 'sam3' library not found.")
    sys.exit(1)

# --- CONFIGURATION ---
DEFAULT_SAM_CONFIG = "sam3_hiera_large.yaml"
DEFAULT_SAM_CHECKPOINT = "checkpoints/sam3_hiera_large.pt"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_PROMPT = "person"

def get_id_suffix(n):
    n = int(n) - 1
    result = ""
    while n >= 0:
        result = chr(ord('a') + (n % 26)) + result
        n = floor(n / 26) - 1
    return result

# --- CORE LOGIC (UNCHANGED) ---
def process_single_video(input_path, output_path, args):
    """
    Process a single video. Logic preserved exactly as requested.
    """
    print(f"[INFO] Processing: {os.path.basename(input_path)}")

    log_gpu_memory("START - Before predictor")

    predictor = build_sam3_video_predictor(
        async_loading_frames=True
    )

    log_gpu_memory("After predictor initialization")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): return {}, 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    palette = sv.ColorPalette.from_hex(["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFD700", "#FFA07A"])
    mask_annotator = sv.MaskAnnotator(color=palette, color_lookup=sv.ColorLookup.TRACK)
    box_annotator = sv.BoxAnnotator(color=palette, color_lookup=sv.ColorLookup.TRACK, thickness=2)
    label_annotator = sv.LabelAnnotator(color=palette, text_position=sv.Position.TOP_CENTER, color_lookup=sv.ColorLookup.TRACK)

    state_output = predictor.start_session(input_path)#1
    log_gpu_memory("After start_session")

    # Handle API variations
    if isinstance(state_output, dict):
        session_id = state_output.get("session_id")
        inference_state = state_output.get("inference_state")
    else:
        inference_state = state_output
        session_id = None

    predictor.add_prompt(session_id=session_id, frame_idx=0, text=TEXT_PROMPT)#2
    log_gpu_memory("After add_prompt")

    generator = predictor.propagate_in_video(
        session_id if session_id else inference_state,
        propagation_direction="forward",
        start_frame_idx=0,
        max_frame_num_to_track=None
    )#3
    log_gpu_memory("After propagate_in_video generator creation")

    tracking_history = {}

    # Initialize tensor tracker
    tracker = TensorTracker()
    snapshot_before_loop = tracker.snapshot("before_loop")
    prev_snapshot = snapshot_before_loop

    frame_count = 0
    for frame_out in generator:
        sam_frame_idx = frame_out.get("frame_index")
        outputs = frame_out.get("outputs")

        if outputs is None: continue
        obj_ids = outputs.get("out_obj_ids")
        video_res_masks = outputs.get("out_binary_masks")
        if obj_ids is None or video_res_masks is None: continue

        # Track memory leaks every 5 frames
        if frame_count % 5 == 0 and frame_count > 0:
            current_snapshot = tracker.snapshot(f"frame_{sam_frame_idx}")
            tracker.compare_snapshots(prev_snapshot, current_snapshot, f"Frame {sam_frame_idx}")

            # Log specific variables that might be leaking
            log_variable_memory({
                'frame_out': frame_out,
                'outputs': outputs,
                'obj_ids': obj_ids,
                'video_res_masks': video_res_masks,
            }, f"Frame {sam_frame_idx} variables")

            prev_snapshot = current_snapshot

        frame_count += 1

        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_pos != sam_frame_idx:
            if 0 < (sam_frame_idx - current_pos) < 5:
                 for _ in range(sam_frame_idx - current_pos): cap.grab()
            else:
                 cap.set(cv2.CAP_PROP_POS_FRAMES, sam_frame_idx)
            
        ret, frame_bgr = cap.read()
        if not ret: break

        if len(obj_ids) > 0:
            # CRITICAL: Convert to numpy immediately to free GPU tensor
            if torch.is_tensor(video_res_masks):
                masks_tensor = video_res_masks.cpu().numpy()
            else:
                masks_tensor = np.asarray(video_res_masks)

            if masks_tensor.ndim == 4 and masks_tensor.shape[1] == 1:
                masks_tensor = masks_tensor.squeeze(1)

            has_pixels = masks_tensor.reshape(masks_tensor.shape[0], -1).any(axis=1)
            valid_indices = np.where(has_pixels)[0]

            if len(valid_indices) > 0:
                current_masks = masks_tensor[valid_indices]
                current_ids = [int(obj_ids[i]) for i in valid_indices]
                
                for oid in current_ids:
                    if oid not in tracking_history:
                        tracking_history[oid] = {'start_frame': sam_frame_idx, 'end_frame': sam_frame_idx}
                    else:
                        tracking_history[oid]['end_frame'] = sam_frame_idx

                detections = sv.Detections(
                    mask=current_masks, xyxy=sv.mask_to_xyxy(current_masks), tracker_id=np.array(current_ids)
                )
                labels = [f"ID: {tid}" for tid in detections.tracker_id]
                annotated_frame = mask_annotator.annotate(frame_bgr.copy(), detections)
                annotated_frame = box_annotator.annotate(annotated_frame, detections)
                annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=labels)
                out.write(annotated_frame)
            else:
                out.write(frame_bgr)
        else:
            out.write(frame_bgr)

        # MEMORY FIX: Clear variables that may hold GPU references
        del frame_out, outputs, video_res_masks
        if 'masks_tensor' in locals():
            del masks_tensor
        if 'current_masks' in locals():
            del current_masks

        # Periodic cleanup
        if frame_count % 50 == 0:
            print(f"   Frame {sam_frame_idx}/{total_frames}", end='\r')
            gc.collect()

    log_gpu_memory("After processing loop - before cleanup")

    cap.release()
    out.release()

    # Explicit cleanup with logging
    del generator
    log_gpu_memory("After deleting generator")

    del predictor
    log_gpu_memory("After deleting predictor")

    if 'inference_state' in locals():
        del inference_state
    if 'session_id' in locals():
        del session_id

    gc.collect()
    torch.cuda.empty_cache()
    log_gpu_memory("After gc.collect() and empty_cache()")

    # Log all remaining tensors
    log_tensor_memory()

    return tracking_history, total_frames, fps

# --- MAIN (ADAPTED FOR SINGLE RUN) ---
def main():
    parser = argparse.ArgumentParser()
    # Arguments simplified for single file processing
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_video", type=str, required=True)
    parser.add_argument("--output_csv_fragment", type=str, required=True)
    parser.add_argument("--clip_name", type=str, required=True, help="Used for ID generation")
    
    parser.add_argument("--sam_config", type=str, default=DEFAULT_SAM_CONFIG)
    parser.add_argument("--sam_checkpoint", type=str, default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    
    args = parser.parse_args()
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    with torch.inference_mode(), torch.autocast(device_type=args.device, dtype=dtype):
        try:
            video_stats, total_frames, fps = process_single_video(
                args.input_video, 
                args.output_video,
                args
            )
            
            # Format Data
            all_tracking_data = []
            objects_list = [{'original_id': k, 'start': v['start_frame'], 'end': v['end_frame']} 
                          for k, v in video_stats.items()]
            objects_list.sort(key=lambda x: (x['start'], x['end']))

            for i, obj in enumerate(objects_list):
                all_tracking_data.append({
                    "clip_name": args.clip_name,
                    "unique_people_id": f"{args.clip_name}{get_id_suffix(i+1)}",
                    "start_frame": obj['start'],
                    "end_frame": obj['end'],
                    "total_frames": total_frames,
                    "fps": fps
                })
            
            # Save Fragment CSV (Headers included, Bash script will handle merging)
            if all_tracking_data:
                pd.DataFrame(all_tracking_data).to_csv(args.output_csv_fragment, index=False)
                print(f"[SUCCESS] Saved CSV fragment to {args.output_csv_fragment}")
            else:
                # Create empty file to signal completion but no data
                open(args.output_csv_fragment, 'a').close()

        except Exception as e:
            print(f"[ERROR] Failed to process {args.input_video}: {e}")
            sys.exit(1) # Exit with error for Bash to catch

if __name__ == "__main__":
    main()


