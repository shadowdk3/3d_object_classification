import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.modelnet40_dataset import ModelNet40Dataset
from src.model import SmallPointNet
import utils.paths
from tqdm.auto import tqdm

# Configuration
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
NUM_CLASSES = 40
NUM_WORKERS = 0
TRAIN_SAMPLE_NUMBER = -1
TEST_SAMPLE_NUMBER = -1

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Dataset
train_dataset = ModelNet40Dataset(
    utils.paths.DATA_PATH,
    split="train",
    max_samples=TRAIN_SAMPLE_NUMBER
)

test_dataset = ModelNet40Dataset(
    utils.paths.DATA_PATH,
    split="test",
    max_samples=TEST_SAMPLE_NUMBER
)

print("Train Samples:", len(train_dataset))
print("Test Samples:", len(test_dataset))

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# Model
model = SmallPointNet(num_classes=NUM_CLASSES).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# Training Function
def train_one_epoch(epoch):
    # train
    model.train()

    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(
        train_loader,
        desc=f"TRAIN {epoch+1}/{NUM_EPOCHS}"
    )

    for points, labels in progress_bar:

        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(points)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        # Statistics
        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        train_accuracy = correct / total

        # Update progress bar
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{train_accuracy:.3f}"
        )

    epoch_loss = total_loss / len(train_loader)
    train_accuracy = correct / total
    
    return epoch_loss, train_accuracy
    
# Evaluation Function    
def evaluate(epoch):
    # Eval
    model.eval()

    total_loss = 0.0
    test_correct = 0
    test_total = 0

    test_progress = tqdm(
        test_loader,
        desc=f"Eval {epoch+1}/{NUM_EPOCHS}"
    )
    
    with torch.no_grad():

        for points, labels in test_progress:

            points = points.to(device)
            labels = labels.to(device)

            outputs = model(points)

            loss = criterion(
                outputs,
                labels
            )
            
            total_loss += loss.item()
            
            predictions = outputs.argmax(dim=1)

            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)

            test_accuracy = test_correct / test_total

            test_progress.set_postfix(
                accuracy=f"{test_accuracy:.3f}"
            )
                
    epoch_loss = total_loss / len(train_loader)
    test_accuracy = test_correct / test_total
    
    return epoch_loss, test_accuracy
    
# Training Loop
best_test_accuracy = 0.0

for epoch in range(NUM_EPOCHS):
    
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    
    train_loss, train_accuracy = train_one_epoch(epoch)
    test_loss, test_accuracy = evaluate(epoch)

    # Save best model
    if test_accuracy > best_test_accuracy:

        best_test_accuracy = test_accuracy

        torch.save(
            model.state_dict(),
            utils.paths.BEST_MODEL_PATH
        )

        print("Best model saved!")

    # result
    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.3f} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Accuracy: {test_accuracy:.3f}"
    )

# save final
torch.save(
    model.state_dict(),
    utils.paths.FINAL_MODEL_PATH
)

print("Final model saved!")