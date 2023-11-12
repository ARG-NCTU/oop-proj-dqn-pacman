import torch

import pytest

import add_path
from deep_Q_network import DQN

# create two instances of DQN: target and policy
def test_dqn_model():
    target = DQN(4)
    policy = DQN(4)
    assert target is not None
    assert policy is not None

# check for model modules
def test_dqn_model_modules():
    policy = DQN(4)
    for module in policy.modules():
        print(module)
        assert module is not None

# check for model parameters
def test_dqn_model_params():
    policy = DQN(4)
    for param in policy.parameters():
        assert param.requires_grad

# check for named parameters
def test_dqn_model_named_params():
    policy = DQN(4)
    for name, param in policy.named_parameters():
        # print name and size of param
        print(name, param.size())
        assert name is not None
        assert param.requires_grad

# check for model conv1 layer inputs/outputs
def test_dqn_model_conv1():

    policy = DQN(4)
    conv1_layer = policy.conv1
    
    # create an input tensor of size torch.Size([1, 4, 44, 40])
    input_tensor = torch.randn(1, 4, 44, 40)
    # feed input tensor to conv1 layer
    output_tensor = conv1_layer(input_tensor)
    # check output tensor size
    assert output_tensor.size() == torch.Size([1, 32, 23, 21])

