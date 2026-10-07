import torch
from src.modelnet40_dataset import ModelNet40Dataset
from src.model import SmallPointNet
import utils.paths

# Configuration
NUM_CLASSES = 40
TEST_SAMPLE_NUMBER = -1
SAMPLE_INDEX = 1000

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = SmallPointNet(num_classes=NUM_CLASSES).to(device)

# Load model
model.load_state_dict(
    torch.load(
        utils.paths.BEST_MODEL_PATH,
        map_location=device
    )
)

model.eval()

# Dataset
test_dataset = ModelNet40Dataset(
    utils.paths.DATA_PATH,
    split="test",
    max_samples=TEST_SAMPLE_NUMBER
)

points, label = test_dataset[SAMPLE_INDEX]

with torch.no_grad():

    points_input = points.unsqueeze(0).to(device)

    output = model(points_input)

    prediction = output.argmax(dim=1).item()

print("Actual:", test_dataset.classes[label])
print("Predicted:", test_dataset.classes[prediction])