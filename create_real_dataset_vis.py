import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Set paths
vid_path = r"C:\Users\MNIT USER\Downloads\UBFC-rPPG Dataset\subject1\vid.avi"
gt_path = r"C:\Users\MNIT USER\Downloads\UBFC-rPPG Dataset\subject1\ground_truth.txt"
output_path = r"paper\figures\dataset_visualization.png"

# Read video frames
cap = cv2.VideoCapture(vid_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame1 = cap.read()
ret, frame2 = cap.read()
cap.release()

# Convert BGR to RGB
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

# Define face ROI (rough approximation for UBFC center)
h, w, _ = frame1.shape
roi_y1, roi_y2 = int(h*0.2), int(h*0.8)
roi_x1, roi_x2 = int(w*0.3), int(w*0.7)

# Extract ROI
face_roi1 = frame1[roi_y1:roi_y2, roi_x1:roi_x2]
face_roi2 = frame2[roi_y1:roi_y2, roi_x1:roi_x2]

# Compute Temporal Difference
diff = cv2.absdiff(face_roi1, face_roi2)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
# Enhance difference for visualization
diff_vis = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)

# Read Ground Truth
with open(gt_path, 'r') as f:
    lines = f.readlines()
ppg = np.fromstring(lines[0].strip(), sep=' ')
time = np.fromstring(lines[2].strip(), sep=' ')

# Select a 10 second window
start_idx = 100
end_idx = start_idx + 300 # 30 fps * 10 sec
ppg_window = ppg[start_idx:end_idx]
time_window = time[start_idx:end_idx] - time[start_idx]

# Create Figure
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 3)

# Plot Raw Frame
ax1 = fig.add_subplot(gs[:, 0])
ax1.imshow(frame1)
# Draw bounding box
rect = plt.Rectangle((roi_x1, roi_y1), roi_x2-roi_x1, roi_y2-roi_y1, fill=False, edgecolor='red', linewidth=2)
ax1.add_patch(rect)
ax1.set_title("1. Raw Video Frame")
ax1.axis('off')

# Plot ROI
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(face_roi1)
ax2.set_title("2. Cropped Face ROI")
ax2.axis('off')

# Plot Diff
ax3 = fig.add_subplot(gs[1, 1])
ax3.imshow(diff_vis, cmap='magma')
ax3.set_title("3. Temporal Difference (ΔI)")
ax3.axis('off')

# Plot PPG
ax4 = fig.add_subplot(gs[:, 2])
ax4.plot(time_window, ppg_window, color='green', linewidth=1.5)
ax4.set_title("4. Ground Truth PPG Waveform")
ax4.set_xlabel("Time (seconds)")
ax4.set_ylabel("Amplitude")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved real dataset visualization to {output_path}")
