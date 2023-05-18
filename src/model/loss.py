from functools import reduce

from torch import Tensor


def _single_averaged_l2_loss(data: Tensor, reference: Tensor):
    assert data.dim() == 1
    assert reference.dim() == 2
    assert data.size(0) == reference.size(1)
    ret = (reference - data) ** 2
    ret = ret.mean(dim=1)
    return ret.mean(dim=0)


def batched_average_l2_loss(data: Tensor, reference: Tensor | list[Tensor]):
    if isinstance(reference, list):
        assert data.dim() == 2
        assert data.size(0) == len(reference)
        losses = [
            _single_averaged_l2_loss(data[i, :], ref)
            for i, ref in enumerate(reference)
        ]
        return reduce(lambda a, b: a + b, losses) / len(losses)
    else:
        return _single_averaged_l2_loss(data, reference)
