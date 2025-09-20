import torch
from torch import nn
from d2l import torch as d2l
from d2l.torch import Accumulator, Animator

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(para_net, data_iter):
    if isinstance(para_net, torch.nn.Module):
        para_net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(para_net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(para_net, para_train_iter, para_loss, para_updater):
    if isinstance(para_net, torch.nn.Module):
        para_net.train()
    metric = Accumulator(3)
    for X, y in para_train_iter:
        y_hat = para_net(X)
        l = para_loss(y_hat, y)
        if isinstance(para_updater, torch.optim.Optimizer):
            para_updater.zero_grad()
            l.mean().backward()
            para_updater.step()
        else:
            l.sum().backward()
            para_updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(para_net, para_train_iter, para_test_iter, para_loss, para_num_epochs, para_updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(para_num_epochs):
        train_metrics = train_epoch_ch3(para_net, para_train_iter, para_loss, para_updater)
        test_acc = evaluate_accuracy(para_net, para_test_iter)
        train_loss, train_acc = train_metrics
        animator.add(epoch + 1, train_metrics + (test_acc,))
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")

def predict_ch3(model, para_test_iter, n=6):
    for X, y in para_test_iter:
        x_label = d2l.get_fashion_mnist_labels(y)
        y_label = d2l.get_fashion_mnist_labels(model(X).argmax(axis=1))  # 使用参数 model
        titles = [true + '\n' + pred for true, pred in zip(x_label, y_label)]
        print(x_label)
        print(" ")
        print(y_label)
        print(" ")
        print(titles)
        print("\n")
        break

lr = 0.1
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net.apply(init_weights)
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
predict_ch3(net, test_iter)