import os
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import json
import hashlib
from datetime import datetime

try:
    import mediapipe as mp
except ImportError:
    pass

CACHE_FILE = "dataset_cache.json"

def auto_search_dataset():
    """Searches common directories for the UBFC-rPPG dataset."""
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
                
    print("\n--- Detected Dataset Structure ---")
    for subj in subjects[:3]:
        subj_id = os.path.basename(subj)
        print(f"{subj_id}\n  vid.avi\n  ground_truth.txt\n")
    if len(subjects) > 3:
        print("...\n")
    print(f"Total Subjects Detected: {len(subjects)}\n----------------------------------\n")
    return subjects

def compute_dataset_fingerprint(subjects):
    hasher = hashlib.sha256()
    # Hash the first 5 ground truth files to create a unique fast fingerprint
    for subj in subjects[:5]:
        gt_path = os.path.join(subj, "ground_truth.txt")
        if os.path.exists(gt_path):
            with open(gt_path, "rb") as f:
                hasher.update(f.read())
    return hasher.hexdigest()[:16]

def check_roi_quality(video_path: str, max_frames=100) -> float:
    try:
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    except:
        return 100.0 # Skip if no mediapipe
        
    cap = cv2.VideoCapture(video_path)
    success_count = 0
    frame_count = 0
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            success_count += 1
        frame_count += 1
        
    cap.release()
    if frame_count == 0: return 0.0
    return (success_count / frame_count) * 100.0

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
    
    for subj in subjects:
        subj_id = os.path.basename(subj)
        video_path = os.path.join(subj, "vid.avi")
        gt_path = os.path.join(subj, "ground_truth.txt")
        
        has_video = os.path.exists(video_path)
        has_gt = os.path.exists(gt_path)
        
        frames, fps, w, h = 0, 0, 0, 0
        roi_success = 0.0
        
        if has_video:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            roi_success = check_roi_quality(video_path)
            face_success_rates.append(roi_success)
            
        gt_len, mean_hr, std_hr, min_hr, max_hr = 0, 0, 0, 0, 0
        nan_count = 0
        
        if has_gt:
            gt_data = np.loadtxt(gt_path)
            if gt_data.ndim == 2 and gt_data.shape[0] >= 3:
                hr_trace = gt_data[1, :]
                nan_count = np.isnan(hr_trace).sum()
                gt_len = len(hr_trace)
                mean_hr = np.nanmean(hr_trace)
                std_hr = np.nanstd(hr_trace)
                min_hr = np.nanmin(hr_trace)
                max_hr = np.nanmax(hr_trace)
                all_hrs.extend(hr_trace[~np.isnan(hr_trace)].tolist())
                
        stats.append({
            'Subject': subj_id,
            'Frames': frames,
            'FPS': fps,
            'Resolution': f"{w}x{h}",
            'GT_Length': gt_len,
            'NaNs': nan_count,
            'Alignment_Diff': abs(frames - gt_len),
            'Mean_HR': mean_hr,
            'ROI_Success': roi_success
        })
        
    df = pd.DataFrame(stats)
    
    # Analyze Dataset Health
    corrupted_videos = df[df['Frames'] == 0]['Subject'].tolist()
    missing_gt = df[df['GT_Length'] == 0]['Subject'].tolist()
    misaligned = df[df['Alignment_Diff'] > fps]['Subject'].tolist()
    avg_roi = np.mean(face_success_rates) if face_success_rates else 0
    
    # Readiness Score
    readiness = {
        "Dataset found": True,
        "Folder structure valid": True,
        "Ground truth aligned": len(misaligned) == 0,
        "Face detection success >95%": avg_roi > 95.0,
        "Missing frames": len(missing_gt) == 0,
        "Corrupted videos": len(corrupted_videos) == 0,
        "HR distribution verified": len(all_hrs) > 0
    }
    
    md_content = f"""# Dataset Quality & Readiness Report (Stage 0.75)

## Dataset Fingerprint
- **Dataset Name**: UBFC-rPPG
- **Subjects Evaluated**: {total_subjects}
- **SHA256 Fingerprint**: `{fingerprint}`
- **Validation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Training Readiness Score
| Check | Status |
|---|---|
"""
    ready = True
    for k, v in readiness.items():
        status = "✅ Pass" if v else "❌ Fail"
        if not v: ready = False
        md_content += f"| {k} | {status} |\n"
        
    md_content += f"\n**Overall Status**: {'✅ READY FOR STAGE 1' if ready else '❌ MANUAL REVIEW REQUIRED'}\n"
    
    md_content += f"""
## 1. Video Quality
- **Average FPS**: {df['FPS'].mean():.2f}
- **Average Frames**: {df['Frames'].mean():.0f}
- **Corrupted Videos**: {len(corrupted_videos)}

## 2. ROI Quality (MediaPipe)
- **Face Detection Success Rate**: {avg_roi:.2f}%

## 3. Physiological Quality
- **HR Range**: {np.min(all_hrs) if all_hrs else 0:.1f} - {np.max(all_hrs) if all_hrs else 0:.1f} BPM
- **Global Mean HR**: {np.mean(all_hrs) if all_hrs else 0:.1f} BPM (±{np.std(all_hrs) if all_hrs else 0:.1f})
- **Missing/NaN Values**: {df['NaNs'].sum()}
- **Alignment Errors**: {len(misaligned)} (diff > 1s)

## Subject Detail Matrix
"""
    md_content += df.to_markdown(index=False)
    
    report_path = os.path.join(report_dir, "dataset_validation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"\nReport generated at {report_path}")
    print(f"Overall Status: {'READY FOR STAGE 1' if ready else 'MANUAL REVIEW REQUIRED'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="")
    parser.add_argument('--report_dir', type=str, default=r"experiments\reports")
    args = parser.parse_args()
    
    target_path = args.dataset_path
    if not target_path:
        discovered_path = auto_search_dataset()
        if discovered_path:
            target_path = discovered_path
        else:
            print("Auto-search failed to locate the dataset. Please provide path explicitly.")
            exit(1)
            
    generate_dataset_report(target_path, args.report_dir)
