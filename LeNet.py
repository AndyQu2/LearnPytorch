import torch
from torch import nn
from torch.utils.data import DataLoader
from ImageDataset import ImageDataset
from d2l.torch import Accumulator, accuracy
import time

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def evaluate_model(para_net, data_iter):
    if isinstance(para_net, torch.nn.Module):
        para_net.eval()
    metric = Accumulator(2)
    device_model = next(para_net.parameters()).device
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device_model), y.to(device_model)
            metric.add(accuracy(para_net(X), y), y.numel())
    return metric[0] / metric[1]


def train_model(para_model, para_train_iter, para_test_iter, num_epochs, lr, para_device):
    if not torch.cuda.is_available() and para_device.type == 'cuda':
        print("Warning: CUDA disabled, using CPU instead.")
        para_device = torch.device('cpu')

    para_model.apply(init_weights)
    print('training on', para_device)
    para_model.to(para_device)

    optimizer = torch.optim.SGD(para_model.parameters(), lr=lr, momentum=0.9)
    loss = nn.CrossEntropyLoss(reduction='mean')

    best_test_acc = 0.0
    best_model_state = None

    start_time = time.time()

    for epoch in range(num_epochs):
        metric = Accumulator(3)
        para_model.train()

        num_batches = len(para_train_iter)

        for i, (X, y) in enumerate(para_train_iter):
            optimizer.zero_grad()
            X, y = X.to(para_device), y.to(para_device)
            y_hat = para_model(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            if (i + 1) % 10 == 0 or i == num_batches - 1:
                train_loss = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]

        test_acc = evaluate_model(para_model, para_test_iter)
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, '
              f'Test Acc: {test_acc:.3f}')

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = para_model.state_dict().copy()

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Duration: {training_time:.2f}秒")
    print(f"Best Accuracy: {best_test_acc:.3f}")

    if best_model_state is not None:
        para_model.load_state_dict(best_model_state)
    return para_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Use device: {device}")

batch_size = 256
dropout1, dropout2 = 0.2, 0.5
train_dataset = ImageDataset('datasets/train')
test_dataset = ImageDataset('datasets/test')

print(f"Length of training dataset: {len(train_dataset)}")
print(f"Length of testing dataset: {len(test_dataset)}")

train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

net = nn.Sequential(
    nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5, stride=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 14 * 14, 120),
    nn.Dropout(dropout1),
    nn.ReLU(inplace=True),
    nn.Linear(120, 84),
    nn.Dropout(dropout2),
    nn.ReLU(inplace=True),
    nn.Linear(84, 3)).to(device)

X_test = torch.randn(2, 3, 64, 64).to(device)
output = net(X_test)
print(f"In Shape: {X_test.shape}, Out shape: {output.shape}")

trained_net = train_model(net, train_iter, test_iter, 200, lr=0.001, para_device=device)

torch.save(trained_net.state_dict(), "model.pth")
print("Saving model to model.pth!")