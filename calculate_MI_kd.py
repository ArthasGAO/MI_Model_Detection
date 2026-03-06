import os
import glob

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # needed for full CUDA determinism
from pathlib import Path
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import csv
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import torchvision
import torchvision.transforms as transforms
from util import create_or_load_group_A, process_yaml_file, evaluate2,\
                 process_experiment_kd_setup, load_best_checkpoint


# =====================================================
# 1. Global setup
# =====================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_worker(worker_id):
    # Worker gets a different, but deterministic, seed derived from the main seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # all GPUs

    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
        # Raises error if a non-deterministic op is used; great for debugging reproducibility
        torch.use_deterministic_algorithms(True)
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

# ---------------- CONFIGURATION ----------------

# =====================================================
# 3. Main training logic
# =====================================================
def main(seed, r, yaml_file_path): 
    print(device)

    g = torch.Generator()
    g.manual_seed(seed)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_kd_setup(exp_yaml) 

    print('==> Preparing data..')

    in_sample_set = exp_setup["Dataset"].in_sample_set # No data augmentation here
    out_sample_set =  exp_setup["Dataset"].test_set

    group_A = create_or_load_group_A(dataset=in_sample_set, save_dir=f'./Indices/{exp_yaml['Dataset']['name']}/',
                                                  group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
    in_sample_subset = exp_setup["Dataset"].subset("train", group_A, clean=True)

    in_sample_loader = DataLoader(in_sample_subset,batch_size=128,shuffle=False,num_workers=8, # in sample without augmentation
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)
    
    out_sample_loader = DataLoader(out_sample_set,batch_size=128,shuffle=False,num_workers=8, # out sample
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)
    
    reserved_keys = ["Dataset", "NumClasses", "Epochs", "GroupSize"]
    kd_methods = [k for k in exp_setup.keys() if k not in reserved_keys]

    for method_name in kd_methods:
        print(f"\n{'='*50}")
        print(f"==> Starting Experiment for Method: {method_name}")
        print(f"{'='*50}")

        print('==> Loading model..')
        model_name = exp_yaml["Scenario_Name"] + f"_{method_name}_{seed}_{r}"
        model_dir = './saved_models/kd_vanilla/'
        model_folder = Path(model_dir)/model_name

        ckpt_path = load_best_checkpoint(model_folder)
        if ckpt_path is None:
            print(f"[⚠️] No .pth files found in {model_folder}")

        student_net = exp_setup[method_name]["Distiller"].student.to(device)
        state = torch.load(ckpt_path, map_location=device)
        student_net.load_state_dict(state)

        log_dir = f'./saved_logs/kd_vanilla/MI'
        os.makedirs(log_dir, exist_ok=True)
        log_name = model_name
        log_file = os.path.join(log_dir, f"training_log_{log_name}_MI.csv")

        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Scenario','Epoch','I(X;T)-InAug', 'I(T;Y)-InAug', 'I(X;T)-In', 'I(T;Y)-In','I(X;T)-Out', 'I(T;Y)-Out']) 
        
        print(f"This is the {method_name} {seed} round on student base 99 epoch!")
        value_xt_in, value_ty_in = evaluate2(student_net, in_sample_loader, device)
        value_xt_out, value_ty_out = evaluate2(student_net, out_sample_loader, device)

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([log_name, 99,
                                0, 0,
                                value_xt_in, value_ty_in,
                                value_xt_out, value_ty_out
                                ])



# =====================================================
# 4. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Folder containing all YAML experiment plans
    exp_dir = "./saved_exp_plan/kd_plan"
    yaml_files = sorted(glob.glob(os.path.join(exp_dir, "*.yaml")))
    
    if not yaml_files:
            print(f"No YAML files found in {exp_dir}")
    else:
        print(f"Found {len(yaml_files)} experiment plan(s):")
        for f in yaml_files:
            print(" -", f)

    # Iterate over YAML files and seeds
    for yaml_path in yaml_files:
        print(f"\n========== Starting experiments from {yaml_path} ==========")
        for seed in range(42, 43): # entry point for main_nega
            print(f"\n>>> Running seed {seed} for {os.path.basename(yaml_path)}")
            set_seed(seed)
        
            main(seed, 1.0, yaml_path)
