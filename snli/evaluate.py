"""
Evaluating the model on the test set.
"""

import torch


def evaluate(test_loader, net, criterion):
    """
    Args:
        test_loader: an iterator over the test set.
        net: the neural network model employed.
        criterion: the loss function.

    Outputs:
        Average loss and accuracy achieved by the model in the test set.
    """
    net.eval()

    accurate = 0
    loss = 0.0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            (s1_embed, s2_embed), (s1_lens, s2_lens), targets = data
            s1_embed, s2_embed = s1_embed.cuda(), s2_embed.cuda()
            targets = targets.cuda()
            scores = net((s1_embed, s1_lens), (s2_embed, s2_lens))
            loss += criterion(scores, targets) * targets.size(0)
            _, predicted = torch.max(scores, 1)
            accurate += (predicted == targets).sum().item()
            total += targets.size(0)

        accuracy = 1.0 * accurate / total
        loss = loss.item() / total

    return (loss, accuracy)
