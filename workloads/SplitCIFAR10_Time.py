import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets

def get_split_cifar10_phases_loaders(num_clients: int, batch_size: int):
    """
    Returns a list (per client) of 5 DataLoaders (one for each phase),
    where each phase gets a random 80/20 train/test split. 80 goes to important
    train data if clients and 20 goes to client eval , global eval test loaders
    are un affected
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 1. Load Standard Full CIFAR10
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    
    # 2. Define the 5 phases (2 classes each)
    class_phases = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    
    # 3. Create 5 Global Test Loaders 
    global_test_loaders_list = []
    for phase_classes in class_phases:
        phase_indices = []
        for idx in range(len(test_ds)):
            _, label = test_ds[idx]
            if label in phase_classes:
                phase_indices.append(idx)
        
        phase_subset = Subset(test_ds, phase_indices)
        loader = DataLoader(phase_subset, batch_size=batch_size, shuffle=False)
        global_test_loaders_list.append(loader)
    
    # 4. Distribute Training Data (80% Train / 20% Train-Test Split per client per phase)
    all_clients_phase_loaders = []
    all_clients_test_loaders = []
    
    # Pre-calculate phase indices entirely once
    train_phase_indices = []
    for phase_classes in class_phases:
        indices = [idx for idx, (_, label) in enumerate(train_ds) if label in phase_classes]
        train_phase_indices.append(indices)
        
    for client_idx in range(num_clients):
        client_train_phases = []
        client_test_phases = []
        
        for phase_idx, phase_classes in enumerate(class_phases):
            # Reuse the single list of indices computed above
            phase_subset = Subset(train_ds, train_phase_indices[phase_idx])
            
            # Random 80-20 split for this client's phase
            train_len = int(0.8 * len(phase_subset))
            test_len = len(phase_subset) - train_len
            
            client_phase_train, client_phase_test = random_split(phase_subset, [train_len, test_len])
            
            train_loader = DataLoader(client_phase_train, batch_size=batch_size, shuffle=True) if train_len > 0 else []
            test_loader = DataLoader(client_phase_test, batch_size=batch_size, shuffle=False) if test_len > 0 else []
            
            client_train_phases.append(train_loader)
            client_test_phases.append(test_loader)
            
        all_clients_phase_loaders.append(client_train_phases)
        all_clients_test_loaders.append(client_test_phases)
        
    return all_clients_phase_loaders, all_clients_test_loaders, global_test_loaders_list