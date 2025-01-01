from torch_geometric.loader import HGTLoader

# Assume 'data' is your heterogeneous graph dataset with node and edge types
# and it has attributes 'train_mask', 'val_mask', and 'test_mask' for node indices
# If you don't have masks, you'll need to create them based on your dataset
# For example, you can create masks by randomly sampling node indices for each type

# Define the parameters for neighbor sampling
# num_samples = [10, 10]  # Number of neighbors to sample per edge type for each layer
# batch_size = 32


def build_dataloaders(data, task, num_samples=[10,10], batch_size=32, loader=HGTLoader):
    # Create HGTLoader for the training dataset
    train_loader = loader(
        data,
        num_samples,
        input_nodes=(task, data[task].train_mask),  
        batch_size=batch_size,
        shuffle=True
    )

    # Create HGTLoader for the validation dataset
    val_loader = loader(
        data,
        num_samples,
        input_nodes=(task, data[task].val_mask),  
        batch_size=batch_size,
        shuffle=False
    )

    # Create HGTLoader for the test dataset
    test_loader = loader(
        data,
        num_samples,
        input_nodes=(task, data[task].test_mask), 
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader
