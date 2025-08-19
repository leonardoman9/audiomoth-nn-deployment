import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import os


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    Export PyTorch model to ONNX format.

    Command-line overrides:
    - model_path: Path to the trained PyTorch model (.pth file)
    - output_path: Path to save the exported ONNX model

    Example usage:
    python export_to_onnx.py model_path=/path/to/model.pth output_path=/path/to/output.onnx
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check for required parameters
    if not hasattr(cfg, 'model_path'):
        raise ValueError("Missing required parameter: model_path")
    if not hasattr(cfg, 'output_path'):
        raise ValueError("Missing required parameter: output_path")

    # Access model path and output path from hydra config
    model_path = cfg.model_path
    output_path = cfg.output_path

    # Determine number of classes based on allowed_classes in config
    num_classes = len(cfg.dataset.allowed_classes)

    # Re-instantiate model in export mode
    export_model = instantiate(cfg.model, num_classes=num_classes, export_mode=True).to(device)
    export_model.load_state_dict(torch.load(model_path, map_location=device))
    export_model.eval()

    # Generate random tensor with correct shape for ONNX export
    # Model expects [batch_size, n_mel_bins, time]
    # Use a reasonable time length for the spectrogram (e.g., 100 frames)
    time_frames = 100
    dummy_input = torch.randn(1, cfg.model.n_mel_bins, time_frames).to(device)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export model to ONNX with dynamic time axis
    torch.onnx.export(
        export_model,
        dummy_input,  # input tensor
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {2: 'time'},  # dynamic time dimension
            'output': {1: 'class'}  # optional
        }
    )

    print(f"ONNX model exported to: {output_path}")


if __name__ == "__main__":
    main()
