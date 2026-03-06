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
from util import process_yaml_file, process_experiment_setup, train_one_epoch, evaluate1, create_train_subset,\
                 prune_model_global, check_pruned_weights, ft_one_epoch, load_pickel_dataset, extract_ft_balanced_train_val,\
                 prepare_group_subset, load_checkpoint_from_epoch, remove_prune_mask, process_experiment_prune_setup
import torch.nn.utils.prune as prune           

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
STRATEGY = ["FT-AL"] #"RT-AL", "FT-LL", In pruning task, we only consider fine tune all layers now.
                     # Since the model layer architecture has been destroyed, so it is recommended to train all together.
SPARSITY = [0.2, 0.6, 0.8] #


# ------------------------------------------------
# ---------------- MAIN EXECUTION ----------------
def main(seed, yaml_file_path):
    print(device)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    exp_yaml = process_yaml_file(yaml_file_path)
    exp_setup = process_experiment_prune_setup(exp_yaml) 

    model_name = f"{exp_yaml['Dataset']}_{exp_yaml['Model']}_42_{1.0}"
    model_dir = './saved_models/normal_nega/'
    model_folder = Path(model_dir)/model_name

    ckpt_path = load_checkpoint_from_epoch(model_folder, "latest") # find last checkpoint
    if ckpt_path is None:
        print(f"[⚠️] No .pth files found in {model_folder}")

    print('==> Building model..')
    # load experiment information
    base_state = torch.load(ckpt_path, map_location=device)
    print(f"[✅] Loaded model for ({ckpt_path.name})")

    print('==> Preparing data..')
    g = torch.Generator()
    g.manual_seed(seed)
    # Dataset Preparation
    print("Case 1️⃣: Assuming the adversary has no access to the original training data domain")
    # 25000 original training vs. 10000 from 50000 (This case represents the stealer has no access to the original training data
    # domain so they try to use one fine-tuning dataset)
    pkl_path = "E:/Experiment/data/ti_500K_pseudo_labeled.pickle"
    X, Y = load_pickel_dataset(pkl_path)
    (X_tr, Y_tr), (X_va, Y_va), (idx_tr, idx_va) = extract_ft_balanced_train_val(X, Y, total_size=10000, seed=0)

    ft_data_train = CIFAR10PseudoLabelDataset(X=X_tr, Y=Y_tr, normalization="cifar10")
    ft_data_val = CIFAR10PseudoLabelDataset(X=X_va, Y=Y_va, normalization="cifar10")

    trainloader_ft = DataLoader(ft_data_train.train_set, batch_size=128, shuffle=True, num_workers=8,
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)
    
    testloader_ft = DataLoader(ft_data_val.in_sample_set, batch_size=128, shuffle=False, num_workers=8,
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    print("Case 2️⃣: Assuming the adversary has access to the original training data domain")
    # 25000 original training vs. 10000 from test 10000(This case represents the stealer has access to the original training data
    # domain so they try to use the same domain test data)
    in_sample_set = exp_setup["Dataset"].in_sample_set # here the training portion are only used to evaluate
    out_sample_set =  exp_setup["Dataset"].test_set

    # 25000 vs. 250000 （50000）
    in_sample_subset = prepare_group_subset(in_sample_set, 1, f"./Indices/{exp_yaml['Dataset']}")

    in_sample_loader = DataLoader(in_sample_subset,batch_size=128,shuffle=False,num_workers=8, # in sample without augmentation
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)
    
    out_sample_loader = DataLoader(out_sample_set,batch_size=128,shuffle=False,num_workers=8, # out sample
                worker_init_fn=seed_worker,generator=g, persistent_workers=True, 
                pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    # Pruning Loop:
    for s in SPARSITY:
        net = exp_setup["Model"].to(device) # load model 
        net.load_state_dict(base_state)
        check_pruned_weights(net)

        pruned_net = prune_model_global(model=net, amount=s) 
        print(f"[✅] Prune model for {model_name} (s = {s}) sparsity:")
        check_pruned_weights(pruned_net)

        scenario_name = model_name + f"_{s}"
        print(scenario_name)
        log_dir = './saved_logs/prune_new/Performance'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_log_{scenario_name}.csv")

        # Make sure log file has header
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Scenario','Epoch', 'Train_Loss', 'Train_Acc', 'Train_Precision', 'Train_Recall','Train_F1'
                                , 'Test_Loss', 'Test_Acc', 'Test_Precision', 'Test_Recall','Test_F1'])

        result_pre_ft_train = evaluate1(pruned_net, in_sample_loader, criterion, device)
        result_pre_ft_test1 = evaluate1(pruned_net, testloader_ft, criterion, device)
        result_pre_ft_test2 = evaluate1(pruned_net, out_sample_loader, criterion, device)

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([scenario_name, -1,
                            result_pre_ft_train["test_loss"], result_pre_ft_train["test_acc"],
                            result_pre_ft_train["test_precision"], result_pre_ft_train["test_recall"], result_pre_ft_train["test_f1"],   
                            result_pre_ft_test1["test_loss"], result_pre_ft_test1["test_acc"],
                            result_pre_ft_test1["test_precision"], result_pre_ft_test1["test_recall"], result_pre_ft_test1["test_f1"]      
                ])
            
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([scenario_name, -2,
                            result_pre_ft_train["test_loss"], result_pre_ft_train["test_acc"],
                            result_pre_ft_train["test_precision"], result_pre_ft_train["test_recall"], result_pre_ft_train["test_f1"],   
                            result_pre_ft_test2["test_loss"], result_pre_ft_test2["test_acc"],
                            result_pre_ft_test2["test_precision"], result_pre_ft_test2["test_recall"], result_pre_ft_test2["test_f1"]      
                ])
        
        os.makedirs(f'./saved_models/pruning_new/{scenario_name}', exist_ok=True)
        torch.save(pruned_net.state_dict(), f'./saved_models/pruning_new/{scenario_name}/epoch_{-1}.pth')

        if exp_setup["DoRestoreFT"]: # Based on the experiment setup, deciding whether do the restoring ft.
            # Restoring Fine-tuning Phase: 
            for strategy in STRATEGY:
                print(f"\n[🔧] Starting fine-tuning strategy: {strategy} with sparsity {s}.")
                scenario_name = scenario_name + f"_{strategy}"
                        
                # ⬇️ Begin post fine tuning       
                pruned_net = setup_finetune(model=pruned_net, strategy=strategy, device=device)
                check_pruned_weights(pruned_net)

                optimizer = exp_setup[s]["Optimizer"]
                scheduler = exp_setup[s]["Scheduler"]
                epochs = exp_setup[s]["Epochs"]

                # Training loop
                for epoch in range(start_epoch, start_epoch + epochs): 
                    print(f"This is the {seed} round, {epoch} epoch!")
                    train_result = ft_one_epoch(pruned_net, trainloader_ft, optimizer, criterion, epoch, device, strategy)       
                    # ---- scheduler step (per epoch) ----
                    if scheduler is not None:
                        scheduler.step()
                    print(f"Epoch {epoch} | LR = {optimizer.param_groups[0]['lr']}")

                    test_result = evaluate1(pruned_net,testloader_ft,criterion,device)

                    with open(log_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([scenario_name, epoch, 
                                        train_result["train_loss"], train_result["train_acc"], train_result["train_precision"], train_result["train_recall"],train_result["train_f1"], 
                                        test_result["test_loss"], test_result["test_acc"], test_result["test_precision"], test_result["test_recall"],test_result["test_f1"], 
                                        ])
                
                    os.makedirs(f'./saved_models/prune_new/{scenario_name}', exist_ok=True)
                    torch.save(pruned_net.state_dict(), f'./saved_models/prune_new/{scenario_name}/epoch_{epoch}.pth') 
                
                    check_pruned_weights(pruned_net)

        remove_prune_mask(pruned_net) # used to erase the masks for next round loading
        


# =====================================================
# 4. Entry point
# =====================================================
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Folder containing all YAML experiment plans
    exp_folder = "./saved_exp_plan/prune_plan"
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