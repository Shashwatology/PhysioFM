import os
import yaml
import subprocess

modes = ['add', 'concat']
base_config = {
    'model_name': 'PHYSIOFM',
    'dataset_name': 'UBFC-rPPG',
    'dataset_path': "C:\\Users\\MNIT USER\\Downloads\\UBFC-rPPG Dataset",
    'epochs': 10,  # 10-epoch screening
    'limit_subjects': None,
    'seed': 42,
    'learning_rate': 0.001,
    'batch_size': 2,
    'seq_len': 150,
    'embed_dim': 256,
    'patience': 10,
    'stds_alpha_init': 0.0  # Not used by add or concat, but kept for signature
}

def main():
    os.makedirs('configs', exist_ok=True)
    
    for mode in modes:
        exp_name = f"EXP05_Fusion_{mode.capitalize()}"
        cfg = base_config.copy()
        cfg['exp_name'] = exp_name
        cfg['stds_fusion_mode'] = mode
        
        cfg_path = f"configs/exp05_{mode}.yaml"
        with open(cfg_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
            
        print(f"==================================================")
        print(f"Launching {exp_name}")
        print(f"==================================================")
        
        exp_dir = f"experiments/{exp_name}"
        if os.path.exists(exp_dir):
            import shutil
            shutil.rmtree(exp_dir)
            
        try:
            subprocess.run(
                [".\\.venv\\Scripts\\python.exe", "-m", "src.research.training.train", "--config", cfg_path],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Experiment {exp_name} failed with code {e.returncode}")
            
    print("EXP05 Campaign Completed.")

if __name__ == '__main__':
    main()
