import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.global_feat = global_feat

    def forward(self, x):
        # Input shape: [B, 3, N]
        n_pts = x.size()[2]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Max pooling over points
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        return x

class PointNetClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(PointNetClassifier, self).__init__()
        self.encoder = PointNetEncoder(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # Input shape: [B, N, 3]
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

if __name__ == '__main__':
    # Test model shape
    model = PointNetClassifier(num_classes=5)
    sim_data = torch.rand(4, 3, 512)
    out = model(sim_data)
    print("Output shape:", out.shape)
