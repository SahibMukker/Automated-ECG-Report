import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class ECGClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ECGClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(12, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        '''
        Forward pass through the ECGClassifier model.

        Parameters:
            x (torch tensor): Batch of ECG signals with shape (batch_size, 12, 5000)

        Returns:
            torch tensor: Batch of output labels with shape (batch_size, num_classes)
        '''
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train_model(model, dataset, device, num_epochs=50, batch_size=64, learning_rate=0.001):
    '''
    Train the ECGClassifier model.

    Args:
        model (nn.Module): Initialized model
        dataset (torch.utils.data.Dataset): Dataset with ECG signals and labels
        device (torch.device): Training device (CPU/GPU)
        num_epochs (int): Number of epochs
        batch_size (int): Size of training batches
        learning_rate (float): Learning rate for optimizer
    '''
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), 'ecg_classifier.pth')
    
    
print("new_ecg_model loaded")
