import random
import torch
import matplotlib.pyplot as plt

def synthetic_data(w_args, b_args, num_examples):
    x = torch.normal(0, 1, (num_examples, len(w_args)))
    y = torch.matmul(x, w_args) + b_args
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))

def data_iter(batch_sizes, features_array, labels_array):
    num_examples = len(features_array)
    indices = list(range(num_examples))

    random.shuffle(indices)
    for i in range(0, num_examples, batch_sizes):
        batch_indices = torch.tensor(indices[i: min(i + batch_sizes, num_examples)])
        yield features_array[batch_indices], labels_array[batch_indices]

def linreg(x_para, w_para, b_para):
    return torch.matmul(x_para, w_para) + b_para

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lrs, batch_sizes):
    with torch.no_grad():
        for param in params:
            param -= - lrs * param.grad / batch_sizes
            param.grad.zero_()

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.005
num_epochs = 10
net = linreg
loss = squared_loss
batch_size = 1000

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')