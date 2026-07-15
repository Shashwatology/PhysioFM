import os
import yaml
import subprocess

alphas = [0.0, 0.1, 0.5, 1.0]
base_config = {
    'model_name': 'PHYSIOFM',
    'dataset_name': 'UBFC-rPPG',
    'dataset_path': "C:\\Users\\MNIT USER\\Downloads\\UBFC-rPPG Dataset",
    'epochs': 10,
    'limit_subjects': None,
    'seed': 42,
    'learning_rate': 0.001,
    'batch_size': 2,
    'seq_len': 150,
    'embed_dim': 256,
    'patience': 10
}

def main():
    os.makedirs('configs', exist_ok=True)
    
    for a in alphas:
        exp_name = f"EXP04A_Alpha_{str(a).replace('.', '_')}"
        cfg = base_config.copy()
        cfg['exp_name'] = exp_name
        cfg['stds_alpha_init'] = a
        
        cfg_path = f"configs/exp04a_alpha_{str(a).replace('.', '_')}.yaml"
        with open(cfg_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
            
        print(f"==================================================")
        print(f"Launching {exp_name} with alpha={a}")
        print(f"==================================================")
        
        # Remove old directory if exists
        exp_dir = f"experiments/{exp_name}"
        if os.path.exists(exp_dir):
            import shutil
            shutil.rmtree(exp_dir)
            
        # Run training
        try:
            subprocess.run(
                [".\\.venv\\Scripts\\python.exe", "-m", "src.research.training.train", "--config", cfg_path],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Experiment {exp_name} failed with code {e.returncode}")
            
    print("EXP04A Campaign Completed.")

if __name__ == '__main__':
    main()
