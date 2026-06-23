"""AlexNet implemented from scratch in PyTorch.

Reference:
    A. Krizhevsky, I. Sutskever, G. E. Hinton.
    "ImageNet Classification with Deep Convolutional Neural Networks."
    NeurIPS 2012.
    https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html

This follows the single-tower form of the architecture (the original paper
split the network across two GPUs). Expects 227x227 RGB input.
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """AlexNet convolutional network for image classification.

    Five convolutional stages (local response normalization after the first
    two, max-pooling after stages 1, 2, and 5) followed by three
    fully-connected layers with dropout on the first two.

    Args:
        num_classes: Number of output classes. Defaults to 1000 (ImageNet).

    Shape:
        - Input:  (N, 3, 227, 227)
        - Output: (N, num_classes)
    """

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()

        # input: (3, 227, 227)  (227x227 required: conv1 has no padding)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4),  # -> (96, 55, 55)
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2),
            nn.MaxPool2d(3, 2)  # -> (96, 27, 27)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),  # -> (256, 27, 27)
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2),
            nn.MaxPool2d(3, 2)  # -> (256, 13, 13)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),  # -> (384, 13, 13)
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),  # -> (384, 13, 13)
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),  # -> (256, 13, 13)
            nn.ReLU(),
            nn.MaxPool2d(3, 2)  # -> (256, 6, 6)
        )

        # flatten: (256, 6, 6) -> 9216
        self.fc1 = nn.Sequential(
            nn.Linear((256 * 6 * 6), 4096),  # -> 4096
            nn.ReLU(),
            nn.Dropout(0.50)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),  # -> 4096
            nn.ReLU(),
            nn.Dropout(0.50)
        )

        self.fc3 = nn.Linear(4096, num_classes)  # -> num_classes

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
            
        for module in (self.conv2[0], self.conv4[0], self.conv5[0], self.fc1[0], self.fc2[0]):
            nn.init.constant_(module.bias, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, 227, 227)
        x = self.conv1(x)  # (N, 96, 27, 27)
        x = self.conv2(x)  # (N, 256, 13, 13)
        x = self.conv3(x)  # (N, 384, 13, 13)
        x = self.conv4(x)  # (N, 384, 13, 13)
        x = self.conv5(x)  # (N, 256, 6, 6)

        x = torch.flatten(x, 1)  # (N, 9216)

        x = self.fc1(x)  # (N, 4096)
        x = self.fc2(x)  # (N, 4096)
        x = self.fc3(x)  # (N, num_classes)

        return x


if __name__ == "__main__":
    model = AlexNet(num_classes=1000)
    x = torch.randn(2, 3, 227, 227)
    print(model(x).shape)
    print(f"{sum(p.numel() for p in model.parameters())} params")
