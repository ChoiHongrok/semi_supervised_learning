import timm
import torch
import torch.nn as nn
import torchvision

class ResNet50(nn.Module):
    def __init__(self, device, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.model = torchvision.models.resnet50(weights=None)
        self.model.layer4 = torch.nn.Identity()
        self.model.fc = torch.nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        self.model.to(device)
    def forward(self, x):
        return self.model(x)
    
    
class ResNet18(nn.Module):
    def __init__(self, device, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.model = torchvision.models.resnet18(weights=None)
        self.model.layer4 = torch.nn.Identity()
        self.model.fc = torch.nn.Linear(in_features=256, out_features=num_classes, bias=True)
        self.model.to(device)
    def forward(self, x):
        return self.model(x)
    
    
class ResNet18_cifar10(nn.Module):
    def __init__(self, device, pretrained=False, num_classes=10):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.model = timm.create_model("resnet18", 
                                       num_classes=num_classes,
                                       pretrained=False)
        if pretrained:
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.model.maxpool = nn.Identity()  # type: ignore
            self.model.fc = nn.Linear(512,  num_classes)

            self.model.load_state_dict(
                torch.hub.load_state_dict_from_url(
                      "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
                       map_location="cpu", 
                       file_name="resnet18_cifar10.pth",
                       )
                )
        self.model.to(device)
        
    def forward(self, x):
        return self.model(x)
    
    
class AlexNet_cifar10(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.model = nn.Sequential(self.features, self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x