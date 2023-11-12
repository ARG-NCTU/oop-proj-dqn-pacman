import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import add_path
from deep_Q_network import DQN 

# Create a DQN object
dqn = DQN(4)

# get the conv1 layer
conv1_layer = dqn.conv1

plt.figure(figsize=(12, 4))
# Visualize filter (first filter in this case)
filter_grid = make_grid(conv1_layer.weight, nrow=4, normalize=True)
plt.imshow(transforms.ToPILImage()(filter_grid))
plt.title('Filter')

plt.show()
