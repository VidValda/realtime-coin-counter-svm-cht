"""
Export coin_cnn.pt and coin_resnet18.pt to TorchScript for C++ (LibTorch) inference.
Run from project root: python export_torchscript.py
Output: coin_cnn_traced.pt, coin_resnet18_traced.pt (load with torch::jit::load in C++).
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18

NUM_CLASSES = 6
CROP_SIZE = 150


class SmallCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def main():
    device = torch.device("cpu")
    example = torch.rand(1, 3, CROP_SIZE, CROP_SIZE, device=device)

    load_kw = {"map_location": device}

    # Export CNN
    ckpt = torch.load("coin_cnn.pt", **load_kw)
    model_cnn = SmallCNN(num_classes=NUM_CLASSES)
    model_cnn.load_state_dict(ckpt["model_state"], strict=True)
    model_cnn.eval()
    traced_cnn = torch.jit.trace(model_cnn, example)
    traced_cnn.save("coin_cnn_traced.pt")
    print("Saved coin_cnn_traced.pt")

    # Export ResNet18
    ckpt = torch.load("coin_resnet18.pt", **load_kw)
    model_resnet = resnet18(weights=None)
    model_resnet.fc = nn.Linear(model_resnet.fc.in_features, NUM_CLASSES)
    model_resnet.load_state_dict(ckpt["model_state"], strict=True)
    model_resnet.eval()
    traced_resnet = torch.jit.trace(model_resnet, example)
    traced_resnet.save("coin_resnet18_traced.pt")
    print("Saved coin_resnet18_traced.pt")


if __name__ == "__main__":
    main()
