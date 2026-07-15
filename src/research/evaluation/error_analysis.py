import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def generate_error_analysis(exp_name: str, report_dir: str):
    """
    Analyzes predictions to find the worst failure cases and generates a report.
    """
    os.makedirs(report_dir, exist_ok=True)
    
    preds_file = f"experiments/{exp_name}/hr_preds.pt"
    targets_file = f"experiments/{exp_name}/hr_targets.pt"
    subject_file = f"experiments/{exp_name}/latents.pt"
    
    if not (os.path.exists(preds_file) and os.path.exists(targets_file)):
        print("Predictions not found!")
        return
        
    preds = torch.load(preds_file).numpy().flatten()
    targets = torch.load(targets_file).numpy().flatten()
    
    try:
        latents_data = torch.load(subject_file)
        subject_ids = latents_data['subject_ids']
    except Exception:
        subject_ids = ["Unknown"] * len(preds)
        
    errors = np.abs(preds - targets)
    
    df = pd.DataFrame({
        'Subject': subject_ids,
        'Prediction': preds,
        'GroundTruth': targets,
        'AbsoluteError': errors
    })
    
    # Sort to find worst 20
    worst_20 = df.sort_values(by='AbsoluteError', ascending=False).head(20)
    best_20 = df.sort_values(by='AbsoluteError', ascending=True).head(20)
    
    # Subject-wise MAE
    subject_mae = df.groupby('Subject')['AbsoluteError'].mean().reset_index()
    subject_mae = subject_mae.sort_values(by='AbsoluteError', ascending=False)
    
    # Generate Markdown Report
    md_content = f"# Error Analysis Report: {exp_name}\n\n"
    
    md_content += "## 1. Subject-Wise MAE Distribution\n"
    md_content += subject_mae.to_markdown(index=False) + "\n\n"
    
    md_content += "## 2. Worst 20 Predictions (Failure Cases)\n"
    md_content += "These cases require manual inspection of ROI, motion, and lighting.\n\n"
    md_content += worst_20.to_markdown(index=False) + "\n\n"
    
    md_content += "## 3. Best 20 Predictions\n"
    md_content += best_20.to_markdown(index=False) + "\n\n"
    
    # Error vs HR Range (Calibration)
    hr_bins = [0, 60, 80, 100, 200]
    labels = ['<60 (Bradycardia)', '60-80 (Normal)', '80-100 (Elevated)', '>100 (Tachycardia)']
    df['HR_Range'] = pd.cut(df['GroundTruth'], bins=hr_bins, labels=labels)
    
    range_mae = df.groupby('HR_Range', observed=False)['AbsoluteError'].mean().reset_index()
    
    md_content += "## 4. Error by Heart Rate Range\n"
    md_content += range_mae.to_markdown(index=False) + "\n"
    
    report_path = os.path.join(report_dir, f"{exp_name}_error_analysis.md")
    with open(report_path, "w") as f:
        f.write(md_content)
        
    print(f"Error analysis generated at {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--report_dir', type=str, default=r"experiments\reports")
    args = parser.parse_args()
    
    generate_error_analysis(args.exp_name, args.report_dir)
