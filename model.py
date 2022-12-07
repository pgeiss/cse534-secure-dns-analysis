import torch
import torch.nn as nn

class NeuralNetworkClassifier(nn.Module):
    def __init__(self, intermediate_size=20, num_classes=11, device='cpu'):
        super(NeuralNetworkClassifier, self).__init__()
        
        self.feature_size = 42
        self.intermediate_size = intermediate_size
        self.num_classes = num_classes
        self.device = device

        self.L = nn.CrossEntropyLoss()

        self.feed_forward = nn.Sequential(
            nn.Linear(self.feature_size, self.intermediate_size),
            nn.ReLU(),
            nn.Linear(self.intermediate_size, self.num_classes)
        )
    
    def forward(self, batch):
        return self.feed_forward(batch)
    
    def loss(self, logits, labels):
        return self.L(logits, labels)