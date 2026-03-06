import os
import glob
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # needed for full CUDA determinism

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
from util import process_yaml_file, process_experiment_setup, train_one_epoch, evaluate1, create_train_subset

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

# =====================================================
# 3. Main training logic
# =====================================================
def main(seed, yaml_file_path): 
    print(device)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    g = torch.Generator()
    g.manual_seed(seed)

    print('==> Preparing data..')

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup(exp_yaml) 

    scenario_name = f"{exp_yaml['Dataset']}_{exp_yaml['Model']}_{seed}" 

    #os.makedirs(f'./saved_logs/normal', exist_ok=True)
    log_file = f'./saved_logs/normal_new/performance/training_log_{scenario_name}.csv'

    train_set = exp_setup["Dataset"].train_set
    test_set =  exp_setup["Dataset"].test_set

    # 40000 vs. 10000 （50000）
    train_subset1, train_subset2 = create_train_subset(train_set, 40000, seed) # here, train_subset1 is used to train, and train_subset2 used to fine-tune

    trainloader = DataLoader(train_subset1, batch_size=128, shuffle=True, num_workers=4,
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    testloader = DataLoader(test_set,batch_size=128,shuffle=False,num_workers=4,
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    print('==> Building model..')
    # load experiment information
    net = exp_setup["Model"].to(device) # load model 
    criterion = nn.CrossEntropyLoss()
    optimizer = exp_setup["Optimizer"]
    scheduler = exp_setup["Scheduler"]
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Make sure log file has header
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario','Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall','Train_F1'
                             , 'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall','Test_F1'])

    # Training loop
    for epoch in range(start_epoch, start_epoch + exp_yaml["Epochs"]): 
        print(f"This is the {seed} round, {epoch} epoch!")
        train_result = train_one_epoch(net, trainloader, optimizer, criterion, epoch, device)       
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
            
    os.makedirs(f'./saved_models/normal_new/{scenario_name}', exist_ok=True)
    torch.save(net.state_dict(), f'./saved_models/normal_new/{scenario_name}/epoch_{epoch}.pth') 


# =====================================================
# 4. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Folder containing all YAML experiment plans
    exp_folder = "./saved_exp_plan/train_plan"
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
            main(seed, yaml_path)
