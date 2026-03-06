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
from util import process_yaml_file, process_experiment_setup, train_one_epoch, evaluate1, create_train_subset, \
        create_or_load_group_A, create_or_load_group_B, prepare_group_subset, train_one_epoch_mix, process_experiment_setup_deit, \
        train_one_epoch_deit, evaluate_deit, process_experiment_kd_setup, train_one_epoch_kd
from torchvision.transforms import v2
from torch.utils.data.dataloader import default_collate
from Model.DeiT import DistillationLoss

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

mixup_cutmix = v2.RandomChoice([
    v2.MixUp(num_classes=100),
    v2.CutMix(num_classes=100),
])

def collate_fn(batch):
    inputs, targets = default_collate(batch)   # inputs: [B,3,32,32], targets: [B] ints
    inputs, targets = mixup_cutmix(inputs, targets)
    return inputs, targets


# =====================================================
# 3. Main training logic
# =====================================================
def main(seed, r, yaml_file_path): # this method is made for CNN normal training with full samples （no overlapping)
    print(device)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    g = torch.Generator()
    g.manual_seed(seed)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_kd_setup(exp_yaml) 

    print('==> Preparing data..')

    train_set = exp_setup["Dataset"].train_set
    test_set =  exp_setup["Dataset"].test_set

    # 25000 vs. 250000 （50000）
    group_A = create_or_load_group_A(dataset=train_set, save_dir=f'./Indices/{exp_yaml['Dataset']['name']}/',
                                                  group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
    train_subset1 =  exp_setup["Dataset"].subset("train", group_A, clean=False) # with augmentation
    
    trainloader = DataLoader(train_subset1, batch_size=128, shuffle=True, num_workers=4,
                    worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                    pin_memory=True)

    testloader = DataLoader(test_set,batch_size=128,shuffle=False,num_workers=4,
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)
    

    reserved_keys = ["Dataset", "NumClasses", "Epochs", "GroupSize"]
    kd_methods = [k for k in exp_setup.keys() if k not in reserved_keys]

    for method_name in kd_methods:
        print(f"\n{'='*50}")
        print(f"==> Starting Experiment for Method: {method_name}")
        print(f"{'='*50}")

        # Unique naming and logging for this specific method
        scenario_name = exp_yaml["Scenario_Name"] + f"_{method_name}_{seed}_{r}"
        
        log_dir = f'./saved_logs/kd_vanilla/Performance'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")

        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Scenario','Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall','Train_F1', 
                                 'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall','Test_F1'])
                
        # Load the specific distiller, optimizer, and scheduler for this method
        distiller = exp_setup[method_name]["Distiller"].to(device)
        optimizer = exp_setup[method_name]["Optimizer"]
        scheduler = exp_setup[method_name]["Scheduler"]
        
        # Note: criterion is handled internally by KD, but evaluate1 might still need it
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_test_acc = -1.0
        os.makedirs(f'./saved_models/kd_vanilla/{scenario_name}', exist_ok=True)

        for epoch in range(start_epoch, start_epoch + exp_yaml["Epochs"]): 
            print(f"This is the {method_name} {seed} round, {epoch} epoch!")
            train_result = train_one_epoch_kd(distiller, trainloader, optimizer, epoch, device)       
            # ---- scheduler step (per epoch) ----
            if scheduler is not None:
                scheduler.step()
            print(f"Epoch {epoch} | LR = {optimizer.param_groups[0]['lr']}")

            test_result = evaluate1(distiller.student, testloader, criterion, device)

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([scenario_name, epoch, 
                                train_result["train_loss"], train_result["train_acc"], train_result["train_precision"], train_result["train_recall"],train_result["train_f1"], 
                                test_result["test_loss"], test_result["test_acc"], test_result["test_precision"], test_result["test_recall"],test_result["test_f1"], 
                                ])
                
            current_acc = test_result["test_acc"]
            if current_acc > best_test_acc:
                best_test_acc = current_acc
                torch.save(distiller.student.state_dict(), f'./saved_models/kd_vanilla/{scenario_name}/best_epoch.pth') 
        
        # Save final epoch
        torch.save(distiller.student.state_dict(), f'./saved_models/kd_vanilla/{scenario_name}/epoch_{epoch}.pth') 
        print(f"==> Finished {method_name}. Best Acc: {best_test_acc:.2f}%\n")


def main_deit(seed, r, yaml_file_path): # this method is made for the DeiT Model(Transformer) normal training.
    print(device)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    g = torch.Generator()
    g.manual_seed(seed)

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_setup_deit(exp_yaml) 

    scenario_name = exp_yaml["Scenario_Name"] + f"_{seed}_{r}"

    log_dir = './saved_logs/vanilla/Performance'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")

    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario','Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall','Train_F1'
                             , 'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall','Test_F1'])

    print('==> Preparing data..')

    train_set = exp_setup["Dataset"].train_set
    test_set =  exp_setup["Dataset"].test_set

    # Prepare class balanced training subset
    group_A = create_or_load_group_A(dataset=train_set, save_dir=f'./Indices/{exp_yaml['Dataset']['name']}/',
                                                  group_size=exp_setup["GroupSize"], num_classes=exp_setup["NumClasses"], seed=42, force_rebuild=False)
    train_subset1 =  exp_setup["Dataset"].subset("train", group_A, clean=False)
    
    trainloader = DataLoader(train_subset1, batch_size=128, shuffle=True, num_workers=4,
                    worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                    pin_memory=True)

    testloader = DataLoader(test_set,batch_size=128,shuffle=False,num_workers=4,
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    print('==> Building model..')
    # load experiment information
    student_net = exp_setup["Student Model"].to(device) # load model
    teacher_net = exp_setup["Teacher Model"].to(device)

    optimizer = exp_setup["Optimizer"]
    scheduler = exp_setup["Scheduler"]
    mixup_fn = exp_setup.get("MixupFn", None)

    # load distillaiton information
    criterion = exp_setup["Distillation"]
    eval_criterion = nn.CrossEntropyLoss()

    # Training loop
    best_test_acc = -1.0
    os.makedirs(f'./saved_models/vanilla/{scenario_name}', exist_ok=True)

    for epoch in range(start_epoch, start_epoch + exp_yaml["Epochs"]): 
        print(f"This is the {seed} round, {epoch} epoch!")
        train_result = train_one_epoch_deit(student_net, trainloader, optimizer, criterion, epoch, device, mixup_fn=mixup_fn)      
        # ---- scheduler step (per epoch) ----
        if scheduler is not None:
            scheduler.step()
        print(f"Epoch {epoch} | LR = {optimizer.param_groups[0]['lr']}")

        test_result = evaluate_deit(student_net,testloader,eval_criterion,device)

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([scenario_name, epoch, 
                             train_result["train_loss"], train_result["train_acc"], train_result["train_precision"], train_result["train_recall"],train_result["train_f1"], 
                             test_result["test_loss"], test_result["test_acc"], test_result["test_precision"], test_result["test_recall"],test_result["test_f1"], 
                            ])
            
        current_acc = test_result["test_acc"]
        if current_acc > best_test_acc:
            best_test_acc = current_acc
            torch.save(student_net.state_dict(), f'./saved_models/vanilla/{scenario_name}/best_epoch.pth') 
            
    torch.save(student_net.state_dict(), f'./saved_models/vanilla/{scenario_name}/epoch_{epoch}.pth')


# =====================================================
# 4. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Folder containing all YAML experiment plans
    exp_folder = "./saved_exp_plan/kd_plan"
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

            main(seed, 1.0, yaml_path)    

            