import torch
import csv
import os
import numpy as np

class DiagnosticMonitor:
    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.grad_file = os.path.join(output_dir, "gradient_dynamics.csv")
        self.act_file = os.path.join(output_dir, "activation_statistics.csv")
        self.collapse_file = os.path.join(output_dir, "collapse_dynamics.csv")
        self.comp_file = os.path.join(output_dir, "comprehensive_metrics.csv")
        
        # Initialize CSVs
        with open(self.grad_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Swin_Stage1', 'Swin_Stage2', 'Swin_Stage3', 'Swin_Stage4', 'STDS', 'Temporal', 'Fusion', 'Head', 'Alpha_Val', 'Alpha_Grad'])
            
        with open(self.act_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Module', 'Mean_Std'])
            
        with open(self.collapse_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Pred_Mean', 'Pred_Std', 'Pred_Entropy', 'Target_Std', 'Latent_Std'])

        with open(self.comp_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Loss', 'ValLoss', 'LR', 
                             'HR_Loss', 'WV_Loss',
                             'PredMean', 'PredStd', 'TargetStd', 'PredictionEntropy', 
                             'Alpha', 'GradAlpha', 'EffectiveAlpha', 
                             'GradHead', 'GradTemporal', 'GradSwin1', 'GradSwin2', 'GradSwin3', 'GradSwin4', 'GradSTDS', 
                             'Head/Swin Ratio', 'Head/STDS Ratio', 
                             'LatentStd', 'TemporalVar', 'MAE', 'RMSE', 'Pearson', 'CCC', 'GPU Memory'])
            
        self.activation_stats = {}
        
        # Register hooks
        # Tracking Dicts
        self.epoch_grad_norms = {
            'Swin_Stage1': [], 'Swin_Stage2': [], 'Swin_Stage3': [], 'Swin_Stage4': [], 'STDS': [], 'STDS_Proj': [],
            'Temporal': [], 'Fusion': [], 'Head': []
        }
        
        self._register_hooks()
        
    def _register_hooks(self):
        self.hooks = []
        modules_to_monitor = {
            'Swin_Stage1': self.model.rgb_spatial.backbone.features[1] if hasattr(self.model.rgb_spatial, 'backbone') else None,
            'STDS': getattr(self.model, 'rgb_stds', None),
            'Temporal': self.model.rgb_temporal,
            'Head': self.model.hr_head
        }
        
        for name, module in modules_to_monitor.items():
            if module is not None:
                hook = module.register_forward_hook(self._get_activation_hook(name))
                self.hooks.append(hook)

    def _get_activation_hook(self, name):
        def hook(module, input, output):
            if name not in self.activation_stats:
                self.activation_stats[name] = []
            if not hasattr(self, 'epoch_activation_means'):
                self.epoch_activation_means = {}
            if name not in self.epoch_activation_means:
                self.epoch_activation_means[name] = []
                
            out_tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(out_tensor, torch.Tensor):
                self.activation_stats[name].append(out_tensor.detach().std().item())
                self.epoch_activation_means[name].append(out_tensor.detach().abs().mean().item())
        return hook
        
    def record_batch_gradients(self):
        modules = {
            'Swin_Stage1': self.model.rgb_spatial.backbone.features[1] if hasattr(self.model.rgb_spatial, 'backbone') else None,
            'Swin_Stage2': self.model.rgb_spatial.backbone.features[3] if hasattr(self.model.rgb_spatial, 'backbone') else None,
            'Swin_Stage3': self.model.rgb_spatial.backbone.features[5] if hasattr(self.model.rgb_spatial, 'backbone') else None,
            'Swin_Stage4': self.model.rgb_spatial.backbone.features[7] if hasattr(self.model.rgb_spatial, 'backbone') else None,
            'STDS': getattr(self.model, 'rgb_stds', None),
            'STDS_Proj': getattr(self.model, 'stds_proj', None),
            'Temporal': self.model.rgb_temporal,
            'Fusion': self.model.fusion,
            'Head': self.model.hr_head
        }
        for name, module in modules.items():
            norm = 0.0
            if module is not None:
                for p in module.parameters():
                    if p.grad is not None:
                        norm += p.grad.data.norm(2).item() ** 2
            self.epoch_grad_norms[name].append(norm ** 0.5)

        # Log alpha specifically
        if hasattr(self.model, 'stds_alpha'):
            alpha_val = self.model.stds_alpha.data.item()
            alpha_grad = self.model.stds_alpha.grad.data.item() if self.model.stds_alpha.grad is not None else 0.0
            if 'Alpha_Val' not in self.epoch_grad_norms:
                self.epoch_grad_norms['Alpha_Val'] = []
                self.epoch_grad_norms['Alpha_Grad'] = []
            self.epoch_grad_norms['Alpha_Val'].append(alpha_val)
            self.epoch_grad_norms['Alpha_Grad'].append(alpha_grad)

    def log_epoch_stats(self, epoch):
        # Log Gradients
        avg_grads = {}
        for name, norms in self.epoch_grad_norms.items():
            avg_grads[name] = np.mean(norms) if len(norms) > 0 else 0.0
            self.epoch_grad_norms[name] = []
        self.last_avg_grads = avg_grads
            
        with open(self.grad_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, 
                             f"{avg_grads.get('Swin_Stage1', 0.0):.6e}", f"{avg_grads.get('Swin_Stage2', 0.0):.6e}", 
                             f"{avg_grads.get('Swin_Stage3', 0.0):.6e}", f"{avg_grads.get('Swin_Stage4', 0.0):.6e}",
                             f"{avg_grads.get('STDS', 0.0):.6e}",
                             f"{avg_grads.get('Temporal', 0.0):.6e}", f"{avg_grads.get('Fusion', 0.0):.6e}", 
                             f"{avg_grads.get('Head', 0.0):.6e}", 
                             f"{avg_grads.get('Alpha_Val', 0.0):.6e}", f"{avg_grads.get('Alpha_Grad', 0.0):.6e}"])
                             
        # Log Activations
        with open(self.act_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for name, stats_list in self.activation_stats.items():
                if len(stats_list) > 0:
                    avg_std = np.mean(stats_list)
                    writer.writerow([epoch, name, avg_std])
        
        # Reset stats for next epoch
        for name in self.activation_stats:
            self.activation_stats[name] = []
            
    def log_validation_stats(self, epoch, preds, targets, latents, mae=0.0, rmse=0.0, pearson=0.0, lr=0.0, **kwargs):
        if len(preds) > 0:
            pred_mean = np.mean(preds)
            pred_std = np.std(preds)
            target_std = np.std(targets)
            latent_std = np.std(latents) if len(latents) > 0 else 0.0
            
            # Histogram-based entropy for predictions
            if pred_std > 1e-6:
                try:
                    hist, _ = np.histogram(preds, bins=20, density=True)
                    hist = hist[hist > 0]
                    pred_entropy = -np.sum(hist * np.log(hist + 1e-9))
                except ValueError:
                    pred_entropy = 0.0
            else:
                pred_entropy = 0.0
                
            with open(self.collapse_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, pred_mean, pred_std, pred_entropy, target_std, latent_std])
                
            # Compute effective alpha
            alpha_val = self.epoch_grad_norms.get('Alpha_Val', [0.0])[-1] if hasattr(self, 'epoch_grad_norms') and 'Alpha_Val' in self.epoch_grad_norms and len(self.epoch_grad_norms['Alpha_Val'])>0 else 0.0
            grad_alpha = self.epoch_grad_norms.get('Alpha_Grad', [0.0])[-1] if hasattr(self, 'epoch_grad_norms') and 'Alpha_Grad' in self.epoch_grad_norms and len(self.epoch_grad_norms['Alpha_Grad'])>0 else 0.0
            
            mean_stds = np.mean(self.epoch_activation_means['STDS']) if hasattr(self, 'epoch_activation_means') and 'STDS' in self.epoch_activation_means and len(self.epoch_activation_means['STDS'])>0 else 0.0
            effective_alpha = alpha_val * mean_stds
            
            grad_stds = self.last_avg_grads.get('STDS', 0.0) if hasattr(self, 'last_avg_grads') else 0.0
            grad_swin1 = self.last_avg_grads.get('Swin_Stage1', 0.0) if hasattr(self, 'last_avg_grads') else 0.0
            grad_swin2 = self.last_avg_grads.get('Swin_Stage2', 0.0) if hasattr(self, 'last_avg_grads') else 0.0
            grad_swin3 = self.last_avg_grads.get('Swin_Stage3', 0.0) if hasattr(self, 'last_avg_grads') else 0.0
            grad_swin4 = self.last_avg_grads.get('Swin_Stage4', 0.0) if hasattr(self, 'last_avg_grads') else 0.0
            grad_head = self.last_avg_grads.get('Head', 0.0) if hasattr(self, 'last_avg_grads') else 0.0
            grad_temporal = self.last_avg_grads.get('Temporal', 0.0) if hasattr(self, 'last_avg_grads') else 0.0
            
            # Using Swin1 for the head/swin ratio
            head_swin_ratio = grad_head / grad_swin1 if grad_swin1 > 1e-12 else 0.0
            head_stds_ratio = grad_head / grad_stds if grad_stds > 1e-12 else 0.0
            
            # Assuming loss, val_loss, and gpu_mem are not available here, they will need to be passed from train.py, or we just write N/A for now if they aren't passed.
            # I will just write what we have since train.py doesn't pass loss, val_loss, gpu_mem. Wait, I should update train.py to pass them!
            # I will pass them as kwargs: loss=0.0, val_loss=0.0, gpu_mem=0.0
            # Wait, since the signature doesn't have them yet, I'll access them via kwargs if I add them.
            loss_val = kwargs.get('loss', 0.0)
            val_loss_val = kwargs.get('val_loss', 0.0)
            gpu_mem = kwargs.get('gpu_mem', 0.0)
            
            hr_loss_val = kwargs.get('hr_loss', 0.0)
            wv_loss_val = kwargs.get('wv_loss', 0.0)
            temporal_var = kwargs.get('temporal_var', 0.0)
            
            with open(self.comp_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, loss_val, val_loss_val, f"{lr:.6e}",
                    f"{hr_loss_val:.4f}", f"{wv_loss_val:.4f}",
                    f"{pred_mean:.4f}", f"{pred_std:.6f}", f"{target_std:.4f}", f"{pred_entropy:.4f}",
                    f"{alpha_val:.6e}", f"{grad_alpha:.6e}", f"{effective_alpha:.6e}",
                    f"{grad_head:.6e}", f"{grad_temporal:.6e}", f"{grad_swin1:.6e}", f"{grad_swin2:.6e}", f"{grad_swin3:.6e}", f"{grad_swin4:.6e}", f"{grad_stds:.6e}",
                    f"{head_swin_ratio:.2f}", f"{head_stds_ratio:.2f}",
                    f"{latent_std:.4f}", f"{temporal_var:.4f}", f"{mae:.4f}", f"{rmse:.4f}", f"{pearson:.4f}", "0.0", f"{gpu_mem:.1f}"
                ])
