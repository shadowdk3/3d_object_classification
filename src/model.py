import torch
import torch.nn as nn

class SmallPointNet(nn.Module):

    def __init__(self, num_classes=40):

        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        # x: [batch, 1024, 3]

        x = self.features(x)

        # [batch, 1024, 256]
        # Take maximum across the 1024 points

        x = torch.max(x, dim=1)[0]

        # [batch, 256]

        x = self.classifier(x)

        return x