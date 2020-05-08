#!/usr/bin/env python

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from model import Net

def train(model, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        #%% INSERT EVALUTION CODE HERE
        data = data.view(data.shape[0], -1)
        output = model(data)
        loss = F.nll_loss(output, target)
        #%%

        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.shape[0], -1)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    BATCH_SIZE = 100
    TEST_BATCH_SIZE = 100
    LR = 0.01
    EPOCHS = 1
    LOG_INTERVAL = 10

    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307, ), (0.3081, ))
                    ])),
                batch_size=BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, 
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307, ), (0.3081, ))
                    ])),
                batch_size=TEST_BATCH_SIZE, shuffle=True)

    model = Net()
    optimizer = optim.Adadelta(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        # Training step
        train(model, train_loader, optimizer, epoch, LOG_INTERVAL)
        # Validation step
        test(model, test_loader)

    # Test
    test(model, test_loader)

    # Save Model
    torch.save(model.state_dict(), 'saved-models/pure-torch/model.pt')

if __name__ == '__main__':
    main()
