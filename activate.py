import os
import torch
import torch.nn as nn


def foo():
    N = 50000

    x_data = torch.rand((N, 128)).cuda()

    y_data = torch.rand((N, 1)).cuda()

    class LinearModel(torch.nn.Module):
        def __init__(self):
            super(LinearModel, self).__init__()
            self.linear1 = torch.nn.Linear(128, 1024)
            self.linear2 = torch.nn.Linear(1024, 1)

        def forward(self, x):
            y_pred = self.linear2(self.linear1(x))
            return y_pred

    model = LinearModel().cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    while True:
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    foo()
