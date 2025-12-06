import torch
import torch.nn as nn
from torch.utils.data import Dataset

class CharacterDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        
        image_tensor = torch.tensor(image, dtype = torch.float32)
        label_tensor = torch.tensor(label, dtype = torch.long)
        
        return image_tensor, label_tensor

class CharacterRecognizer(nn.Module):
    def __init__(self, num_classes, input_channels = 1):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels = input_channels, 
                out_channels = 32, 
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 2, 
                stride = 2
            ),
            nn.Conv2d(
                in_channels = 32, 
                out_channels = 64, 
                kernel_size = 3, 
                padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 2, 
                stride = 2
            ),
            nn.Conv2d(
                in_channels = 64, 
                out_channels = 128, 
                kernel_size = 3, 
                padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 2, 
                stride = 2
            ),
            nn.Flatten(),
            nn.Linear(
                in_features = 128 * 8 * 8, 
                out_features = 512
            ),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(
                in_features = 512, 
                out_features = 128
            ),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(
                in_features = 128, 
                out_features = num_classes
            )
        )

    def forward(self, x):
        return self.model(x)