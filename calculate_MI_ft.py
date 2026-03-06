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
from util import process_yaml_file, process_experiment_ft_setup, create_train_subset, evaluate2, prepare_group_subset,\
                 load_last_checkpoint, load_checkpoint_from_epoch, build_epoch_to_ckpt_map, create_or_load_group_A

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
STRATEGY = ['FT-LL', 'FT-AL']#, 'RT-AL']

# =====================================================
# 3. Main training logic
# =====================================================
def main(seed1, seed2, yaml_file_path): 
    print(device)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_ft_setup(exp_yaml)

    g = torch.Generator()
    g.manual_seed(seed2)

    print('==> Preparing data..')
    train_set = exp_setup["Dataset"].train_set
    test_set =  exp_setup["Dataset"].test_set
    in_sample_set = exp_setup["Dataset"].in_sample_set # No data augmentation here
    # each run is saved in its own folder, e.g. ./saved_models/CIFAR-10_ResNet-18_45/

    # 40000 vs. 10000 （50000）
    train_subset, _ = create_train_subset(train_set, 40000, seed2) # with data augmentation
    in_sample_subset, _  = create_train_subset(in_sample_set, 40000, seed2)

    trainloader = DataLoader(train_subset, batch_size=128, shuffle=False, num_workers=8, # in sample with augmentation
                    worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                    pin_memory=True)

    testloader = DataLoader(test_set,batch_size=128,shuffle=False,num_workers=8, # out sample
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)
    
    in_sample_loader = DataLoader(in_sample_subset,batch_size=128,shuffle=False,num_workers=8, # in sample without augmentation
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    for strategy in STRATEGY:
        model_name = f"{exp_yaml['Dataset']}_{exp_yaml['Model']}_{seed1}_{strategy}"
        print(model_name)
        model_dir = './saved_models/fine_tune_new/'
        model_folder = Path(model_dir)/model_name

        ckpt_path = load_last_checkpoint(model_folder)
        if ckpt_path is None:
            print(f"[⚠️] No .pth files found in {model_folder}")

        net = exp_setup["Model"].to(device) # load model
        state = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(state)

        log_dir = './saved_logs/fine_tune_new/MI'
        os.makedirs(log_dir, exist_ok=True)
        log_name = model_name + f"_{seed2}"
        log_file = os.path.join(log_dir, f"training_log_{log_name}_MI.csv")

        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Scenario','Epoch','I(X;T)-InAug', 'I(T;Y)-InAug', 'I(X;T)-In', 'I(T;Y)-In','I(X;T)-Out', 'I(T;Y)-Out']) #,

        # Evaluation loop

        print(f"This is the {seed1} round on {seed2} base 99 epoch!") # here we only consider calculating the last epoch's metric
        value_xt_inAug, value_ty_inAug = evaluate2(net, trainloader, device)
        value_xt_in, value_ty_in = evaluate2(net, in_sample_loader, device)
        value_xt_out, value_ty_out = evaluate2(net, testloader, device)
        #scheduler.step()

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([log_name, 99,
                                value_xt_inAug, value_ty_inAug,
                                value_xt_in, value_ty_in,
                                value_xt_out, value_ty_out
                                ])

def main_new_one_epoch(seed, r, yaml_file_path): # use last epoch from fine-tuning
    print(device)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_ft_setup(exp_yaml)

    g = torch.Generator()
    g.manual_seed(seed)

    print('==> Preparing data..')

    in_sample_set = exp_setup["Dataset"].in_sample_set # No data augmentation here
    out_sample_set =  exp_setup["Dataset"].test_set

    # Important: the in sample evaluation set should be the same each round
    group_A = create_or_load_group_A(dataset=in_sample_set, save_dir=f'./Indices/{exp_yaml['Dataset']['name']}/',
                                                  group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
    in_sample_subset = exp_setup["Dataset"].subset("train", group_A, clean=True)

    in_sample_loader = DataLoader(in_sample_subset,batch_size=128,shuffle=False,num_workers=8, # in sample without augmentation
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)
    
    out_sample_loader = DataLoader(out_sample_set,batch_size=128,shuffle=False,num_workers=8, # out sample
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    for strategy in STRATEGY:
        model_name = exp_yaml["Scenario_Name"] + f"_{seed}_{r}_{strategy}"
        print(model_name)
        model_dir = './saved_models/fine_tune_vanilla/'
        model_folder = Path(model_dir)/model_name

        ckpt_path = load_last_checkpoint(model_folder)
        if ckpt_path is None:
            print(f"[⚠️] No .pth files found in {model_folder}")

        net = exp_setup["Model"].to(device) # load model
        state = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(state)

        log_dir = './saved_logs/fine_tune_vanilla/MI'
        os.makedirs(log_dir, exist_ok=True)
        log_name = model_name
        log_file = os.path.join(log_dir, f"training_log_{log_name}_MI.csv")

        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Scenario','Epoch','I(X;T)-InAug', 'I(T;Y)-InAug', 'I(X;T)-In', 'I(T;Y)-In','I(X;T)-Out', 'I(T;Y)-Out']) #,

        # Evaluation loop

        print(f"This is the {seed} round on {strategy} base 99 epoch!") # here we only consider calculating the last epoch's metric
        value_xt_in, value_ty_in = evaluate2(net, in_sample_loader, device)
        value_xt_out, value_ty_out = evaluate2(net, out_sample_loader, device)
        #scheduler.step()

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([log_name, 99,
                                0, 0,
                                value_xt_in, value_ty_in,
                                value_xt_out, value_ty_out
                                ])
            

def main_new_multiple_epochs(seed, r, yaml_file_path): 
    print(device)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_ft_setup(exp_yaml)

    g = torch.Generator()
    g.manual_seed(seed)

    print('==> Preparing data..')

    in_sample_set = exp_setup["Dataset"].in_sample_set # No data augmentation here
    out_sample_set =  exp_setup["Dataset"].test_set

    # Important: the in sample evaluation set should be the same each round
    group_A = create_or_load_group_A(dataset=in_sample_set, save_dir=f'./Indices/{exp_yaml['Dataset']['name']}/',
                                                  group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
    in_sample_subset = exp_setup["Dataset"].subset("train", group_A, clean=True)
    
    in_sample_loader = DataLoader(in_sample_subset,batch_size=128,shuffle=False,num_workers=8, # in sample without augmentation
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)
    
    out_sample_loader = DataLoader(out_sample_set,batch_size=128,shuffle=False,num_workers=8, # out sample
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    for strategy in STRATEGY:
        model_name = exp_yaml["Scenario_Name"] + f"_{seed}_{r}_{strategy}"
        print(model_name)
        model_dir = './saved_models/fine_tune_vanilla/'
        model_folder = Path(model_dir)/model_name
        
        ckpt_map = build_epoch_to_ckpt_map(model_dir=model_folder, gap=1) # this map contains all possible epochs .pth file in this folder.

        log_dir = './saved_logs/fine_tune_vanilla/MI'
        os.makedirs(log_dir, exist_ok=True)
        log_name = model_name
        log_file = os.path.join(log_dir, f"training_log_{log_name}_MI.csv")

        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Scenario','Epoch','I(X;T)-In', 'I(T;Y)-In','I(X;T)-Out', 'I(T;Y)-Out'])
        
        for epoch in sorted(ckpt_map.keys()):
            ckpt_path = ckpt_map[epoch] # loading the current epoch's pth file path through key

            net = exp_setup["Model"].to(device) # load model
            state = torch.load(ckpt_path, map_location=device)
            net.load_state_dict(state)

            # Evaluation loop
            print(f"This is the {seed} round on {strategy} base {epoch} epoch!") # here we only consider calculating the last epoch's metric
            value_xt_in, value_ty_in = evaluate2(net, in_sample_loader, device)
            value_xt_out, value_ty_out = evaluate2(net, out_sample_loader, device)
            #scheduler.step()

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([log_name, epoch,
                                value_xt_in, value_ty_in,
                                value_xt_out, value_ty_out
                                ])

# =====================================================
# 4. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Folder containing all YAML experiment plans
    exp_dir = "./saved_exp_plan/ft_plan"
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

        '''for sus_seed in range(42, 43):
            for vic_seed in range(42, 43):
                print(f"\n>>> Running suscipious seed {sus_seed} on {vic_seed} for {os.path.basename(yaml_path)}")
                set_seed(vic_seed)
                main(seed1 = sus_seed, seed2 = vic_seed, yaml_file_path = yaml_path)'''
        
        for seed in range(42, 43): # entry point for main_nega
            print(f"\n>>> Running seed {seed} for {os.path.basename(yaml_path)}")
            set_seed(seed)
        
            main_new_multiple_epochs(seed, 1.0, yaml_path)
