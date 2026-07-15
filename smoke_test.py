import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.research.preprocessing.ubfc_dataset import UBFCDataset
from src.research.models.physio_fm import PhysioFM

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_smoke_test():
    report = []
    status = {"all_passed": True}

    def verify(name, condition, details=""):
        res = "PASS" if condition else "FAIL"
        if not condition:
            status["all_passed"] = False
        report.append(f"| {name} | {res} | {details} |")
        print(f"[{res}] {name}: {details}")

    report.append("# Smoke Test Report (Phase 2)")
    report.append("Configuration: 1 Subject, 1 Epoch, Seed=42, No Mixed Precision\n")
    report.append("| Verification Check | Status | Details |")
    report.append("|---|---|---|")

    start_time = time.time()
    set_seed(42)

    try:
        # 1. Dataset Loading
        with open("dataset_cache.json", "r") as f:
            cache = json.load(f)
            data_path = cache.get('ubfc_root')
            
        dataset = UBFCDataset(root_dir=data_path, split='train')
        dataset.samples = dataset.samples[:1]  # Limit to 1 subject
        verify("Dataset Loading", len(dataset) > 0, f"Found {len(dataset)} sequence(s)")

        # 2. DataLoader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        batch = next(iter(dataloader))
        videos, targets = batch
        verify("DataLoader", videos is not None, f"Batch shape: {videos.shape}")

        # 3. CUDA Allocation
        has_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if has_cuda else 'cpu')
        verify("CUDA Allocation", has_cuda, f"Device: {device}")

        videos = videos.to(device)
        target_hr = targets['hr'].view(-1, 1).to(device)

        model = PhysioFM(embed_dim=256, seq_len=videos.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        # 4. Forward Pass
        outputs = model(videos, thm_video=None)
        verify("Forward Pass", 'hr' in outputs, f"Output HR shape: {outputs['hr'].shape}")

        # 5. Finite Loss
        loss = criterion(outputs['hr'], target_hr)
        is_finite_loss = torch.isfinite(loss).item()
        verify("Finite Loss", is_finite_loss, f"Loss: {loss.item():.4f}")

        # 6. Backward Pass
        loss.backward()
        verify("Backward Pass", True, "Successfully computed gradients")

        # 7. Finite Gradients
        has_nans = False
        for param in model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                has_nans = True
                break
        verify("Finite Gradients", not has_nans, "No NaNs or Infs in gradients")

        # 8. Optimizer Update
        optimizer.step()
        verify("Optimizer Update", True, "Successfully updated weights")

        # 9. Checkpoint Save
        ckpt_path = "smoke_ckpt.pt"
        torch.save(model.state_dict(), ckpt_path)
        verify("Checkpoint Save", os.path.exists(ckpt_path), "Saved smoke_ckpt.pt")

        # 10. Checkpoint Reload
        new_model = PhysioFM(embed_dim=256, seq_len=videos.shape[1]).to(device)
        new_model.load_state_dict(torch.load(ckpt_path))
        verify("Checkpoint Reload", True, "Successfully loaded state dict")

        # 11. Inference Pass
        new_model.eval()
        with torch.no_grad():
            infer_out = new_model(videos, thm_video=None)
            infer_loss = criterion(infer_out['hr'], target_hr)
        verify("Inference Pass", True, f"Inference Loss: {infer_loss.item():.4f}")

    except Exception as e:
        status["all_passed"] = False
        report.append(f"| Critical Failure | FAIL | Exception: {str(e)} |")
        print(f"CRITICAL FAILURE: {e}")

    # Metrics
    end_time = time.time()
    training_time = end_time - start_time
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

    report.append("\n## Performance Metrics")
    report.append(f"- **Training Time**: {training_time:.2f} seconds")
    report.append(f"- **Peak GPU Memory**: {peak_mem:.2f} MB")

    final_decision = "PASS" if status["all_passed"] else "FAIL"
    report.append(f"\n**FINAL DECISION: {final_decision}**")

    with open("smoke_test_report.md", "w") as f:
        f.write("\n".join(report))
        
    print(f"\nSmoke Test Report generated. Final Decision: {final_decision}")

if __name__ == "__main__":
    run_smoke_test()
