import os
import glob
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # needed for full CUDA determinism
from pathlib import Path
import torch
import torch.nn as nn
from Model.ResNet_18 import ResNet18
from Model.VGG16 import ModifiedVGG16
import torch.backends.cudnn as cudnn
import random
import numpy as np
from Dataset.CIFAR_10 import CIFAR10Dataset, CIFAR10PseudoLabelDataset
from Dataset.CIFAR_100 import CIFAR100Dataset
from torch.utils.data import Subset, DataLoader
from util import setup_finetune
import torch.optim as optim
import csv
from util import process_yaml_file, process_experiment_ft_setup, train_one_epoch, evaluate1, create_train_subset\
,ft_one_epoch, extract_ft_balanced_train_val, load_pickel_dataset, setup_finetune_deit, evaluate_deit, create_or_load_group_B\
,prepare_group_subset


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

# ---------------- HELPER: LOAD LAST CHECKPOINT ----------------
def load_last_checkpoint(model_dir: Path):
    """
    Load the most recent .pth checkpoint in a model directory.
    Returns the checkpoint path or None if not found.
    """
    ckpt_files = sorted(model_dir.glob("*.pth"), key=os.path.getmtime)
    if not ckpt_files:
        return None
    last_ckpt = ckpt_files[-1]  # the newest one
    print(f"[📦] Loading last checkpoint: {last_ckpt.name}")
    return last_ckpt

def load_best_checkpoint(model_dir: Path, filename: str = "best_epoch.pth") -> Path | None:
    """
    Return the path to the best checkpoint file (default: best_epoch.pth) if it exists.
    """
    model_dir = Path(model_dir)
    best_ckpt = model_dir / filename

    if best_ckpt.exists() and best_ckpt.is_file():
        print(f"[📦] Loading best checkpoint: {best_ckpt.name}")
        return best_ckpt

    return None

    
# ---------------- CONFIGURATION ----------------

STRATEGY = ['FT-LL', 'FT-AL']#, 'RT-AL']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------------------------------------
# ---------------- MAIN EXECUTION ----------------
def main(seed, yaml_file_path):
    print(device)
    start_epoch = 0

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_ft_setup(exp_yaml) 

    scenario_name = f"{exp_yaml['Dataset']}_{exp_yaml['Model']}_{seed}"#_{exp_yaml['Strategy']}"

    model_dir = './saved_models/normal_new/'
    model_folder = Path(model_dir)/scenario_name
    ckpt_path = load_last_checkpoint(model_folder) # find last checkpoint
    if ckpt_path is None:
        print(f"[⚠️] No .pth files found in {model_folder}")

    print('==> Preparing data..')
    g = torch.Generator()
    g.manual_seed(seed)

    # Dataset Preparation
    train_set = exp_setup["Dataset"].train_set
    test_set =  exp_setup["Dataset"].test_set

    # 40000 vs. 10000 （50000）
    train_subset1, train_subset2 = create_train_subset(train_set, 40000, seed) # here, train_subset1 is used to train, and train_subset2 used to fine-tune

    trainloader = DataLoader(train_subset2, batch_size=64, shuffle=True, num_workers=8,
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    testloader = DataLoader(test_set,batch_size=128,shuffle=False,num_workers=8,
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    print('==> Building model..')
    # load experiment information
    net = exp_setup["Model"].to(device) # load model 
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)

    print(f"[✅] Loaded model for ({ckpt_path.name})")

    criterion = nn.CrossEntropyLoss()

    # Fine Tune Training loop
    for strategy in STRATEGY:

        net = setup_finetune(model=net, strategy=strategy, device=device)

        scenario_name = f"{exp_yaml['Dataset']}_{exp_yaml['Model']}_{seed}_{strategy}"
        print(scenario_name)
        log_dir = './saved_logs/fine_tune_new/Performance'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")

        # Make sure log file has header
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Scenario','Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall','Train_F1'
                                , 'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall','Test_F1'])
        
        optimizer = exp_setup[strategy]["Optimizer"]
        scheduler = exp_setup[strategy]["Scheduler"]
        epochs = exp_setup[strategy]["Epochs"]

        for epoch in range(start_epoch, start_epoch + epochs): 
            print(f"This is the {seed} round, {epoch} epoch!")
            train_result = ft_one_epoch(net, trainloader, optimizer, criterion, epoch, device, strategy)       
            # ---- scheduler step (per epoch) ----
            if scheduler is not None:
                scheduler.step()
            print(f"Epoch {epoch} | LR = {optimizer.param_groups[0]['lr']}")

            test_result = evaluate1(net,testloader,criterion,device)

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([scenario_name, epoch, 
                                train_result["train_loss"], train_result["train_acc"], train_result["train_precision"], train_result["train_recall"],train_result["train_f1"], 
                                test_result["test_loss"], test_result["test_acc"], test_result["test_precision"], test_result["test_recall"],test_result["test_f1"], 
                                ])
        
        os.makedirs(f'./saved_models/fine_tune_new/{scenario_name}', exist_ok=True)
        torch.save(net.state_dict(), f'./saved_models/fine_tune_new/{scenario_name}/epoch_{epoch}.pth') 


def main_new1(seed, r, yaml_file_path):
    print(device)
    start_epoch = 0

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_ft_setup(exp_yaml) 

    model_name = exp_yaml["Model_Name"] + f"_{seed}_{r}"

    model_dir = './saved_models/vanilla/'
    model_folder = Path(model_dir)/model_name
    #ckpt_path = load_last_checkpoint(model_folder) # find last checkpoint
    ckpt_path = load_best_checkpoint(model_folder) # find best checkpoint
    if ckpt_path is None:
        print(f"[⚠️] No .pth files found in {model_folder}")

    print('==> Preparing data..')
    g = torch.Generator()
    g.manual_seed(seed)

    # Dataset Preparation
    if exp_yaml["FT_Dataset"]["note"] != "same":
        pkl_path = "E:/Experiment/data/ti_500K_pseudo_labeled.pickle"
        X, Y = load_pickel_dataset(pkl_path)
        (X_tr, Y_tr), (X_va, Y_va), (idx_tr, idx_va) = extract_ft_balanced_train_val(X, Y, total_size=exp_setup["FT_GroupSize"], seed=0)

        exp_setup["FT_Dataset"].set_data(X, Y)
        ft_data_train = exp_setup["FT_Dataset"].subset("train", idx_tr, clean=False)
        ft_data_val = exp_setup["FT_Dataset"].subset("train", idx_va, clean=True)
    else:
        group_B = create_or_load_group_B(group_size=exp_setup["FT_GroupSize"], overlap_rate=0.0, 
                                         save_dir=f'./Indices/{exp_yaml['Dataset']['name']}/',
                                         seed=42)
        ft_data_train = exp_setup["Dataset"].subset("train", group_B, clean=False)
        ft_data_val = exp_setup["Dataset"].test_set

    trainloader = DataLoader(ft_data_train, batch_size=128, shuffle=True, num_workers=4,
                worker_init_fn=seed_worker,generator=g, persistent_workers=True,
                pin_memory=True)
    
    testloader = DataLoader(ft_data_val, batch_size=128, shuffle=False, num_workers=4,
                worker_init_fn=seed_worker,generator=g, persistent_workers=True,
                pin_memory=True)

    print('==> Building model..')
    # load experiment information
    net = exp_setup["Model"].to(device) # load model 
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)

    print(f"[✅] Loaded model for ({ckpt_path.name})")

    criterion = nn.CrossEntropyLoss()

    # Fine Tune Training loop
    for strategy in STRATEGY:

        net = setup_finetune(model=net, strategy=strategy, device=device)

        scenario_name = exp_yaml["Scenario_Name"] + f"_{seed}_{r}_{strategy}"
        print(scenario_name)
        log_dir = './saved_logs/fine_tune_vanilla/Performance'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")

        # Make sure log file has header
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Scenario','Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall','Train_F1'
                                , 'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall','Test_F1'])
        
        pre_test_result = evaluate1(net,testloader,criterion,device) # this part is the metrics before fine tuning.
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([scenario_name, -1, 
                             0, 0, 0, 0, 0, 
                             pre_test_result["test_loss"], pre_test_result["test_acc"], pre_test_result["test_precision"], pre_test_result["test_recall"],pre_test_result["test_f1"], 
                            ])
        
        optimizer = exp_setup[strategy]["Optimizer"]
        scheduler = exp_setup[strategy]["Scheduler"]
        epochs = exp_setup[strategy]["Epochs"]

        for epoch in range(start_epoch, start_epoch + epochs): 
            print(f"This is the {seed} round, {epoch} epoch!")
            train_result = ft_one_epoch(net, trainloader, optimizer, criterion, epoch, device, strategy)       
            # ---- scheduler step (per epoch) ----
            if scheduler is not None:
                scheduler.step()
            print(f"Epoch {epoch} | LR = {optimizer.param_groups[0]['lr']}")

            test_result = evaluate1(net,testloader,criterion,device)

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([scenario_name, epoch, 
                                train_result["train_loss"], train_result["train_acc"], train_result["train_precision"], train_result["train_recall"],train_result["train_f1"], 
                                test_result["test_loss"], test_result["test_acc"], test_result["test_precision"], test_result["test_recall"],test_result["test_f1"], 
                                ])
        
            os.makedirs(f'./saved_models/fine_tune_vanilla/{scenario_name}', exist_ok=True)
            torch.save(net.state_dict(), f'./saved_models/fine_tune_vanilla/{scenario_name}/epoch_{epoch}.pth') 


def main_new_deit(seed, yaml_file_path):
    print(device)
    start_epoch = 0

    g = torch.Generator()
    g.manual_seed(seed)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_ft_setup(exp_yaml) 

    model_name = exp_yaml["Scenario_Name"]

    model_dir = './saved_models/normal_nega/'
    model_folder = Path(model_dir)/model_name
    ckpt_path = load_last_checkpoint(model_folder) # find last checkpoint
    if ckpt_path is None:
        print(f"[⚠️] No .pth files found in {model_folder}")

    print('==> Preparing data..')
    # Dataset Preparation
    # 25000 original training vs. 10000 fromm 50000
    pkl_path = "E:/Experiment/data/ti_500K_pseudo_labeled.pickle"
    X, Y = load_pickel_dataset(pkl_path)
    (X_tr, Y_tr), (X_va, Y_va), (idx_tr, idx_va) = extract_ft_balanced_train_val(X, Y, total_size=10000, seed=0)

    ft_data_train = CIFAR10PseudoLabelDataset(X=X_tr, Y=Y_tr, normalization="cifar10")
    ft_data_val = CIFAR10PseudoLabelDataset(X=X_va, Y=Y_va, normalization="cifar10")

    trainloader = DataLoader(ft_data_train.train_set, batch_size=128, shuffle=True, num_workers=8,
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)
    
    testloader = DataLoader(ft_data_val.in_sample_set, batch_size=128, shuffle=False, num_workers=8,
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    print('==> Building model..')
    # load experiment information
    net = exp_setup["Student Model"].to(device) # load model
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)

    print(f"[✅] Loaded model for ({ckpt_path.name})")

    eval_criterion = nn.CrossEntropyLoss()

    # Fine Tune Training loop
    for strategy in STRATEGY:

        net = setup_finetune_deit(model=net, strategy=strategy, device=device)

        scenario_name = model_name + f"_{strategy}_{seed}"
        print(scenario_name)
        log_dir = './saved_logs/fine_tune_new1/Performance'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")

        # Make sure log file has header
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Scenario','Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall','Train_F1'
                                , 'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall','Test_F1'])
        
        pre_test_result = evaluate_deit(net, testloader, eval_criterion, device) # this part is the performance before fine tuning.
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([scenario_name, -1, 
                             0, 0, 0, 0, 0, 
                             pre_test_result["test_loss"], pre_test_result["test_acc"], pre_test_result["test_precision"], pre_test_result["test_recall"],pre_test_result["test_f1"], 
                            ])
        
        optimizer = exp_setup[strategy]["Optimizer"]
        scheduler = exp_setup[strategy]["Scheduler"]
        epochs = exp_setup[strategy]["Epochs"]

        criterion = nn.CrossEntropyLoss()

        for epoch in range(start_epoch, start_epoch + epochs): 
            print(f"This is the {seed} round, {epoch} epoch!")
            train_result = ft_one_epoch(net, trainloader, optimizer, criterion, epoch, device, strategy)       
            # ---- scheduler step (per epoch) ----
            if scheduler is not None:
                scheduler.step()
            print(f"Epoch {epoch} | LR = {optimizer.param_groups[0]['lr']}")

            test_result = evaluate1(net,testloader,criterion,device)

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([scenario_name, epoch, 
                                train_result["train_loss"], train_result["train_acc"], train_result["train_precision"], train_result["train_recall"],train_result["train_f1"], 
                                test_result["test_loss"], test_result["test_acc"], test_result["test_precision"], test_result["test_recall"],test_result["test_f1"], 
                                ])
        
            os.makedirs(f'./saved_models/fine_tune_new1/{scenario_name}', exist_ok=True)
            torch.save(net.state_dict(), f'./saved_models/fine_tune_new1/{scenario_name}/epoch_{epoch}.pth') 


# =====================================================
# 4. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Folder containing all YAML experiment plans
    exp_folder = "./saved_exp_plan/ft_plan"
    yaml_files = sorted(glob.glob(os.path.join(exp_folder, "*.yaml")))
    
    if not yaml_files:
            print(f"No YAML files found in {exp_folder}")
    else:
        print(f"Found {len(yaml_files)} experiment plan(s):")
        for f in yaml_files:
            print(" -", f)

    # Iterate over YAML files and seeds
    for yaml_path in yaml_files:
        print(f"\n========== Starting experiments from {yaml_path} ==========")

        for seed in range(42, 43):
            print(f"\n>>> Running seed {seed} for {os.path.basename(yaml_path)}")
            set_seed(seed)
            main_new1(seed, 1.0, yaml_path)


