import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), # input layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128), # hidden layer 1
            nn.ReLU(),
            nn.Linear(128, 64), # hidden layer 2
            nn.ReLU(),
            nn.Linear(64, 1), # output layer
            nn.Sigmoid()  # For binary classification
        )

    def forward(self, x):
        return self.net(x)

