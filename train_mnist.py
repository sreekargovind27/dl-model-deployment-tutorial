# train_mnist.py
import torch, os, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1, self.conv2 = nn.Conv2d(1, 10, 5), nn.Conv2d(10, 20, 5)
        self.conv2_drop, self.fc1, self.fc2 = nn.Dropout2d(), nn.Linear(320, 50), nn.Linear(50, 10)
    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = torch.dropout(x, training=self.training)
        return torch.log_softmax(self.fc2(x), dim=1)

def train_model():
    print("Starting MNIST model training...")
    if not os.path.exists('models'): os.makedirs('models')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), 64, True)
    model, optimizer, criterion = MNIST_CNN(), optim.SGD(MNIST_CNN().parameters(), 0.01, 0.5), nn.NLLLoss()
    model.train()
    for epoch in range(1, 4):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad(); output = model(data); loss = criterion(output, target)
            loss.backward(); optimizer.step()
            if batch_idx % 200 == 0: print(f'Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
    torch.save(model.state_dict(), "models/mnist_cnn.pth")
    print("\nTraining complete. Model saved to models/mnist_cnn.pth")

if __name__ == '__main__': train_model()