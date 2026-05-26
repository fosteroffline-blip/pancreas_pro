import torch
import torch.nn as nn
from torchvision import models

# ----------------------------------
# DOUBLE CONV BLOCK
# ----------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),   # 0
            nn.BatchNorm2d(out_c),                  # 1
            nn.ReLU(inplace=True),                 # 2
            nn.Conv2d(out_c, out_c, 3, padding=1), # 3
            nn.BatchNorm2d(out_c),                 # 4
            nn.ReLU(inplace=True)                  # 5
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------------
# REAL ATTENTION UNET MATCHING WEIGHTS
# ----------------------------------
class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.conv1 = DoubleConv(1,64)
        self.conv2 = DoubleConv(64,128)
        self.conv3 = DoubleConv(128,256)

        self.up1 = nn.ConvTranspose2d(256,128,2,2)
        self.up2 = nn.ConvTranspose2d(128,64,2,2)

        self.final = nn.Conv2d(64,1,1)

    def forward(self,x):

        e1 = self.conv1(x)
        e2 = self.conv2(self.pool(e1))
        e3 = self.conv3(self.pool(e2))

        d2 = self.up1(e3)
        d1 = self.up2(d2)

        out = self.final(d1)

        return torch.sigmoid(out)

# ----------------------------------
# CLASSIFIER
# ----------------------------------
def get_classifier():
    model = models.efficientnet_b4(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2
    )

    return model