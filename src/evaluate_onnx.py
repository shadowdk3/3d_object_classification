from src.modelnet40_dataset import ModelNet40Dataset
import utils.paths
import onnxruntime as ort
import numpy as np
from pathlib import Path

# Configuration
NUM_CLASSES = 40
TEST_SAMPLE_NUMBER = -1
SAMPLE_INDEX = 500

# Device / ONNX Runtime
# This checks whether ONNX Runtime has GPU support available
# If CUDA is available, it prefers GPU execution and allows CPU fallback. Otherwise, it uses the CPU
providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers()
    else ["CPUExecutionProvider"]
)

# Creates an inference session using your exported ONNX model
session = ort.InferenceSession(
    str(Path(utils.paths.MODEL_PATH) / "pointnet_model_best.onnx"),
    providers=providers,
)
print("Providers:", session.get_providers())

# Dataset
test_dataset = ModelNet40Dataset(
    utils.paths.DATA_PATH,
    split="test",
    max_samples=TEST_SAMPLE_NUMBER
)

points, label = test_dataset[SAMPLE_INDEX]

# ONNX inference

# Converts the point cloud into the format expected by ONNX model: 1024 points, [X, Y, Z] 1024 x3
# Adds a batch dimension for one object : 1 × 1024 × 3
# Converts the tensor into a NumPy array that ONNX Runtime can accept
points_input = points.unsqueeze(0).cpu().numpy().astype(np.float32)

# This sends the point cloud to the ONNX model
# The model returns 40 scores, one for each object category
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

output = session.run(
    [output_name],
    {input_name: points_input}
)[0]

# Find the index of the highest score
prediction = int(np.argmax(output, axis=1)[0])

print("Actual:", test_dataset.classes[label])
print("Predicted:", test_dataset.classes[prediction])