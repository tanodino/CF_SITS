import torch
import torch.nn as nn
import torch.nn.functional as F


def discriminator_loss(real_output, fake_output, loss_bce, device):
    # real_labels = torch.full((real_output.shape[0],), 1., dtype=torch.float, device=device)
    # fake_labels = torch.full((fake_output.shape[0],), 0., dtype=torch.float, device=device)
    real_labels = torch.Tensor([1.] * real_output.shape[0]).to(device)
    fake_labels = torch.Tensor([0.] * fake_output.shape[0]).to(device)
    # print("real_output ",real_output)
    # print("fake_output ",fake_output)

    real_loss = loss_bce(real_output, real_labels)
    fake_loss = loss_bce(fake_output, fake_labels)
    total_loss = (real_loss + fake_loss) / 2
    return total_loss


def generator_loss(fake_output, loss_bce, device):
    # real_labels = torch.full((fake_output.shape[0],), 1., dtype=torch.float, device=device)
    real_labels = torch.Tensor([1.] * fake_output.shape[0]).to(device)
    return loss_bce(fake_output, real_labels)
