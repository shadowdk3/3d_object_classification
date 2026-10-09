"""
1. Input: .pth checkpoint (trained PointNet weights)
2. Load and prepare model: create architecture and load weights
3. Export to ONNX: Trace model using dummy input
4. Output: .onnx model (validated ONNX file for inference)

usage: specify the input checkpoint and optionally an output filename

python -m utils.export_onnx models/pointnet_model_best.pth
or 
python -m utils.export_onnx models/pointnet_model_best.pth models/my_pointnet.onnx
"""

import argparse
import torch
import onnx
from src.model import SmallPointNet
from pathlib import Path
import utils.paths

# define no. of class for the model
NUM_CLASSES = 40

def main():
    # read command-line arguments
    # allows specify the input checkpoint and optionally an output filename
    parser = argparse.ArgumentParser(
        description="Export a PointNet .pth checkpoint to ONNX."
    )
    parser.add_argument("checkpoint", help="Input .pth checkpoint")
    parser.add_argument(
        "output",
        nargs="?",
        help="Output .onnx filename (default: same name as input)"
    )   
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    file_name = checkpoint_path.stem

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(utils.paths.MODEL_PATH) / f"{file_name}.onnx"
        
    # CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallPointNet(num_classes=NUM_CLASSES).to(device)
    
    # load model
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,            # places the loaded tensors on the selected device
        weights_only=True,              # restricts loading to weights-oriented data 
                                        # supported by PyTorch's safe loading mechanism
    )

    # support common checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)    # loads the trained parameters into SmallPointNet
    model.eval()

    # create dummy input, used to demonstrate the model's input and computation structure during export
    # input shape: [batch_size, 1024, 3]
    # 1: batch size — one object.
    # 1024: points per object.
    # 3: XYZ coordinates for each point.
    dummy_input = torch.randn(1, 1024, 3, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # export the model to ONNX
    # The output contains 40 scores for each input object. 
    # The highest-scoring class can be selected with argmax() during inference.
    # The dynamic batch axis means the model can accept inputs such as:
    # [1, 1024, 3] — one object.
    # [8, 1024, 3] — eight objects.
    # [32, 1024, 3] — 32 objects.
    torch.onnx.export(
        model,                              # trained PointNet model
        (dummy_input,),                     # example input used for export
        str(output_path),                   # destination ONNX filename
        input_names=["points"],             # names the input tensor points
        output_names=["logits"],            # names the output tensor logits
        dynamic_axes={                      # allows variable batch sizes
            "points": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,                   # ONNX operator set version 17
        dynamo=False,                       # uses PyTorch's legacy TorchScript-based ONNX export path
    )

    # validate the ONNX file
    onnx.checker.check_model(str(output_path))
    print("")
    print(f"Export successful: {checkpoint_path} -> {output_path}")
    
if __name__ == "__main__":
    main()