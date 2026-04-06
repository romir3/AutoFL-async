import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets

def get_split_cifar100_phases_loaders(num_clients: int, batch_size: int):
    """
    Returns a list (per client) of 5 DataLoaders (one for each phase),
    where each phase gets a random 80/20 train/test split. 80 goes to important
    train data for clients and 20 goes to client eval. Global eval test loaders
    are unaffected and span the 5 tasks.
    """
    # CIFAR100 standard normalization values
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    
    # 1. Load Standard Full CIFAR100
    train_ds = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    
    # 2. Define the 5 phases (20 classes each for CIFAR100)
    # This generates: [[0..19], [20..39], [40..59], [60..79], [80..99]]
    class_phases = [list(range(i * 20, (i + 1) * 20)) for i in range(5)]
    
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
    
    # Pre-calculate phase indices entirely once (Fast loading)
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
