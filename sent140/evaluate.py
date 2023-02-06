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
        test_loader.reset()
        while test_loader.has_next():
            data = test_loader.next_batch()
            sent_embeds, sent_lens, labels = data
            sent_embeds = sent_embeds.cuda()
            labels = labels.cuda()
            scores = net((sent_embeds, sent_lens))
            loss += criterion(scores, labels) * labels.size(0)
            _, predicted = torch.max(scores, 1)
            accurate += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 1.0 * accurate / total
        loss = loss.item() / total

    return (loss, accuracy)
