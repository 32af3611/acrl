import numpy as np
from torch import nn
from torch.nn.modules import activation


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: list, hidden_activation: activation, bias=True, output_activation=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.bias = bias
        self.hidden_activation = hidden_activation

        in_sizes = [input_size] + hidden_sizes
        out_sizes = hidden_sizes + [output_size]

        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size, bias=bias) for in_size, out_size in zip(in_sizes[:-1], out_sizes[:-1])])

        self.output_layer = nn.Linear(in_sizes[-1], out_sizes[-1])
        self.output_activation = output_activation

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        for hidden in self.hidden_layers:
            x = hidden(x)
            if self.hidden_activation:
                x = self.hidden_activation(x)

        x = self.output_layer(x)

        if self.output_activation:
            x = self.output_activation(x)

        return x


class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []

        self.n = 0

    def __len__(self):
        return len(self.buffer)

    def put(self, item):
        if self.n < self.size:
            self.buffer.append(item)
            self.n += 1
            return
        self.n += 1
        self.buffer[self.n % self.size] = item

    def get(self, n):
        indices = np.random.choice(np.arange(0, len(self.buffer)), size=n)
        return [self.buffer[i] for i in indices]
