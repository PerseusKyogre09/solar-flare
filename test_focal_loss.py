import torch

from train_cnn_lstm import FocalLoss


def test_focal_loss_emphasizes_hard_minorities():
    logits = torch.tensor([
        [4.0, 0.0, 0.0],
        [0.2, 2.0, 0.2],
    ])
    targets = torch.tensor([0, 2])

    loss = FocalLoss(gamma=2.0, weight=torch.tensor([1.0, 2.0, 4.0]))
    value = loss(logits, targets)

    assert torch.isfinite(value)
    assert value > 0.0
    assert value.item() > 0.1
