from torch import nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def count_precise_macs(model, input_tensor):
    """
    More precise estimation of MAC operations that aligns with model complexity.
    
    Args:
        model (nn.Module): The neural network model
        input_tensor (torch.Tensor): Input tensor to the model
    
    Returns:
        int: More accurate estimation of MAC operations
    """
    total_macs = 0
    
    def mac_counter_hook(module, input, output):
        nonlocal total_macs
        
        # Linear layer MACs: input_features * output_features
        if isinstance(module, nn.Linear):
            macs = input[0].size(-1) * output.size(-1)  # more robust to different dim orders
            if module.bias is not None:
                macs += output.size(-1)  # bias addition
            total_macs += macs * output.numel() // output.size(-1)  # account for batch
        
        # GRU layer MACs
        elif isinstance(module, nn.GRU):
            batch_size = module.batch_first and input[0].size(0) or input[0].size(1)
            seq_len = input[0].size(module.batch_first and 1 or 0)
            input_size = input[0].size(2)
            hidden_size = module.hidden_size
            
            # MACs per time step
            # 3 gates: (input_size + hidden_size) * hidden_size * 3
            # plus hidden state update: hidden_size
            macs_per_step = (input_size + hidden_size) * hidden_size * 3 + hidden_size
            total_macs += batch_size * seq_len * macs_per_step
        
        # Convolutional layer MACs
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            # Get dimensions
            out_channels = module.out_channels
            in_channels = module.in_channels
            kernel_size = module.kernel_size
            groups = module.groups
            
            # Calculate output spatial size
            output_spatial_size = output.size()[2:]
            output_elements = np.prod(output_spatial_size)
            
            # MACs calculation accounting for groups
            kernel_ops = in_channels // groups * np.prod(kernel_size)
            layer_macs = kernel_ops * out_channels * output_elements
            total_macs += layer_macs
            
            # Bias
            if module.bias is not None:
                total_macs += out_channels * output_elements
        
        # Add other layer types as needed...
        else:
            # Optionally warn about unhandled layers
            pass

    # Register hooks to compute MACs
    handles = []
    for module in model.modules():
        handles.append(module.register_forward_hook(mac_counter_hook))
    
    # Run forward pass to trigger MAC calculation
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    return total_macs

# 1. Verify data loading and class distribution
def check_dataset(dataset, ALLOWED_CLASSES):
    print(f"Dataset size: {len(dataset)}")
    
    # Check class distribution
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = {cls: labels.count(i) for i, cls in enumerate(ALLOWED_CLASSES)}
    print("Class distribution:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    
    # Check sample shapes
    sample, _ = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Sample min: {sample.min()}, max: {sample.max()}, mean: {sample.mean()}")
    
    # Check for NaN or inf values
    if torch.isnan(sample).any() or torch.isinf(sample).any():
        print("Warning: Sample contains NaN or inf values")

# 2. Verify model initialization
def check_model(model):
    print(f"Model architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Check for NaN or inf in parameters
    has_nan = False
    for name, param in tqdm(model.named_parameters()):
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"Warning: Parameter {name} contains NaN or inf values")
            has_nan = True
    
    if not has_nan:
        print("All parameters are valid (no NaN/inf)")

# 3. Verify forward pass
def check_forward_pass(model, dataset, device):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=64)
    inputs, labels = next(iter(dataloader))
    inputs = inputs.to(device)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Input min: {inputs.min()}, max: {inputs.max()}, mean: {inputs.mean()}")
    
    with torch.no_grad():
        try:
            outputs = model(inputs)
            print(f"Output shape: {outputs.shape}")
            print(f"Output min: {outputs.min()}, max: {outputs.max()}, mean: {outputs.mean()}")
            
            # Check predictions
            _, predicted = torch.max(outputs, 1)
            print(f"Predictions: {predicted}")
            
            # Check for NaN or inf in outputs
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("Warning: Output contains NaN or inf values")
            else:
                print("Output contains valid values (no NaN/inf)")
                
        except Exception as e:
            print(f"Error during forward pass: {e}")