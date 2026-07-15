import os
import glob
import numpy as np
import pandas as pd
import cv2
import json
import hashlib
from datetime import datetime

try:
    import mediapipe as mp
except ImportError:
    pass

CACHE_FILE = "dataset_cache.json"

def auto_search_dataset():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
            if 'ubfc_root' in cache and os.path.exists(cache['ubfc_root']):
                print(f"Loaded dataset path from cache: {cache['ubfc_root']}")
                return cache['ubfc_root']
                
    search_paths = [
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Documents"),
        os.path.expanduser("~/Desktop"),
        "D:\\",
        "C:\\"
    ]
    
    print("Initiating automatic dataset discovery...")
    for root_path in search_paths:
        if not os.path.exists(root_path):
            continue
        print(f"Searching in {root_path}...")
        for dirpath, dirnames, filenames in os.walk(root_path):
            if any(skip in dirpath.lower() for skip in ['windows', 'program files', 'appdata']):
                continue
            if 'ground_truth.txt' in filenames and 'vid.avi' in filenames:
                parent_dir = os.path.dirname(dirpath)
                print(f"Found dataset root: {parent_dir}")
                with open(CACHE_FILE, "w") as f:
                    json.dump({'ubfc_root': parent_dir}, f)
                return parent_dir
    return None

def detect_dataset_structure(dataset_path: str) -> list:
    subjects = sorted(glob.glob(os.path.join(dataset_path, "subject*")))
    if not subjects:
        dirs = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        for d in dirs:
            if os.path.exists(os.path.join(d, "vid.avi")) and os.path.exists(os.path.join(d, "ground_truth.txt")):
                subjects.append(d)
    return subjects

def compute_dataset_fingerprint(subjects):
    hasher = hashlib.sha256()
    for subj in subjects[:5]:
        gt_path = os.path.join(subj, "ground_truth.txt")
        if os.path.exists(gt_path):
            with open(gt_path, "rb") as f:
                hasher.update(f.read())
    return hasher.hexdigest()[:16]

def check_roi_quality(video_path: str, max_frames=100):
    try:
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    except:
        return 100.0, 0
        
    cap = cv2.VideoCapture(video_path)
    success_count = 0
    frame_count = 0
    empty_frames = 0
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        
        if np.mean(frame) < 5.0:  # Detect completely black/empty frames
            empty_frames += 1
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            success_count += 1
        frame_count += 1
        
    cap.release()
    if frame_count == 0: return 0.0, 0
    return (success_count / frame_count) * 100.0, empty_frames

def generate_dataset_report(dataset_path: str, report_dir: str):
    subjects = detect_dataset_structure(dataset_path)
    if not subjects:
        print("No valid subjects found. Aborting validation.")
        return
        
    total_subjects = len(subjects)
    fingerprint = compute_dataset_fingerprint(subjects)
    print(f"Scanning dataset at {dataset_path} (Fingerprint: {fingerprint})")
    
    os.makedirs(report_dir, exist_ok=True)
    stats, all_hrs, face_success_rates = [], [], []
    warnings_list = []
    
    for subj in subjects:
        subj_id = os.path.basename(subj)
        video_path = os.path.join(subj, "vid.avi")
        gt_path = os.path.join(subj, "ground_truth.txt")
        
        has_video = os.path.exists(video_path)
        has_gt = os.path.exists(gt_path)
        
        missing_files = []
        if not has_video: missing_files.append("vid.avi")
        if not has_gt: missing_files.append("ground_truth.txt")
        
        frames, fps, duration = 0, 0.0, 0.0
        roi_success, empty_frames = 0.0, 0
        decode_status = "Fail"
        
        if has_video:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    duration = frames / fps
                decode_status = "Pass" if frames > 0 else "Fail (Empty)"
            else:
                decode_status = "Fail (Cannot Open)"
            cap.release()
            
            if decode_status == "Pass":
                roi_success, empty_frames = check_roi_quality(video_path)
                face_success_rates.append(roi_success)
                if empty_frames > 0:
                    warnings_list.append(f"[{subj_id}] Detected {empty_frames} empty/black frames in video.")
            else:
                warnings_list.append(f"[{subj_id}] Video decoding failed.")
                
        gt_len = 0
        nan_count = 0
        if has_gt:
            try:
                gt_data = np.loadtxt(gt_path)
                if gt_data.ndim == 2 and gt_data.shape[0] >= 3:
                    hr_trace = gt_data[1, :]
                    nan_count = np.isnan(hr_trace).sum()
                    gt_len = len(hr_trace)
                    all_hrs.extend(hr_trace[~np.isnan(hr_trace)].tolist())
                    if nan_count > 0:
                        warnings_list.append(f"[{subj_id}] Ground truth contains {nan_count} NaN values.")
            except Exception as e:
                warnings_list.append(f"[{subj_id}] Error reading ground truth: {str(e)}")

        alignment_diff = abs(frames - gt_len)
        if alignment_diff > fps and fps > 0:
            warnings_list.append(f"[{subj_id}] Severe GT/Video misalignment (Diff: {alignment_diff} frames).")

        corrupted = (decode_status != "Pass" or gt_len == 0 or len(missing_files) > 0)
        usable = not corrupted and alignment_diff <= max(fps * 2, 30) # within 2 seconds
        
        if not usable and not corrupted:
            warnings_list.append(f"[{subj_id}] Subject intact but marked unusable due to extreme misalignment.")

        stats.append({
            'Subject': subj_id,
            'Missing_Files': ", ".join(missing_files) if missing_files else "None",
            'Decode_Status': decode_status,
            'FPS': round(fps, 2),
            'Frames': frames,
            'Duration_Sec': round(duration, 2),
            'GT_Length': gt_len,
            'Alignment_Diff': alignment_diff,
            'NaN_Count': nan_count,
            'Empty_Frames': empty_frames,
            'Corrupted': corrupted,
            'Usable': usable
        })
        
    df = pd.DataFrame(stats)
    
    usable_count = df['Usable'].sum()
    corrupted_count = df['Corrupted'].sum()
    total_missing_files = sum(1 for x in df['Missing_Files'] if x != "None")
    total_nans = df['NaN_Count'].sum()
    total_empty_frames = df['Empty_Frames'].sum()
    
    md_content = f"""# Dataset Validation Report (Phase 1)

## Global Metrics
- **Total Detected Subjects**: {total_subjects}
- **Usable Subjects**: {usable_count}
- **Corrupted Subjects**: {corrupted_count}
- **Subjects with Missing Files**: {total_missing_files}
- **Total Empty/Black Frames Detected**: {total_empty_frames}
- **Total NaNs in Ground Truth**: {total_nans}
- **Average FPS**: {df['FPS'].mean():.2f}
- **Average Video Duration**: {df['Duration_Sec'].mean():.2f} seconds

## Warnings
"""
    if warnings_list:
        for w in warnings_list:
            md_content += f"- ⚠️ {w}\n"
    else:
        md_content += "- ✅ No critical warnings detected.\n"

    md_content += "\n## Subject Detail Matrix\n"
    md_content += df.drop(columns=['Corrupted', 'Usable']).to_markdown(index=False)
    
    report_path = os.path.join(report_dir, "dataset_validation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"\nPhase 1 Report generated at {report_path}")

if __name__ == "__main__":
    target_path = auto_search_dataset()
    if target_path:
        generate_dataset_report(target_path, ".")
    else:
        print("FAIL: Dataset not found.")
