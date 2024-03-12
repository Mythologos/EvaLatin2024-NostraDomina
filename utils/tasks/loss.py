from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Optional, Type

from torch import Tensor, mean, sum, vmap
from torch.nn import NLLLoss
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.nn.functional import nll_loss


class WrappedNLLLoss(NLLLoss):
    def forward(self, predictions: Tensor, ground_truth_values: Tensor, **kwargs) -> Tensor:
        return super().forward(predictions, ground_truth_values)


class DistanceWeightedNLL(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, size_average: Optional[bool] = None,
                 ignore_index: Optional[int] = -100, reduce: Optional[bool] = None,
                 reduction: Optional[str] = 'mean'):
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.mapped_nll = vmap(nll_loss, 0)
        if reduction == 'mean':
            reduction_function: Callable[[Tensor, bool], Tensor] = mean
        elif reduction == 'sum':
            reduction_function: Callable[[Tensor, bool], Tensor] = sum
        else:
            raise ValueError("Reduction currently required for use.")
        self.reduction_function = reduction_function

    @abstractmethod
    def forward(self, predictions: Tensor, ground_truth_values: Tensor, distance_weights: Tensor) -> Tensor:
        raise NotImplementedError

    def get_nll_kwargs(self) -> dict[str, Any]:
        nll_kwargs: dict[str, Any] = {
            "ignore_index": self.ignore_index,
            "reduction": self.reduction
        }
        return nll_kwargs


class GoldDistanceWeightedNLL(DistanceWeightedNLL):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index=-100, reduce=None,
                 reduction='mean'):
        super().__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, predictions: Tensor, ground_truth_values: Tensor, distance_weights: Tensor) -> Tensor:
        # We compute the NLLLoss for each item in the batch.
        # (B, C), (C) -> (C)
        nll_kwargs: dict[str, Any] = self.get_nll_kwargs()
        nll_kwargs["weight"] = self.weight
        nll_losses: Tensor = self.mapped_nll(predictions, ground_truth_values, **nll_kwargs)
        # (C), (C) -> (1)
        gdw_nll_losses: Tensor = self.reduction_function(nll_losses * distance_weights, dim=0, keepdim=False)
        return gdw_nll_losses


class NamedLossFunction(str, Enum):
    GDW_NLL: str = "gdw-nll"
    NLL: str = "nll"


CLASSIFICATION_LOSS_FUNCTIONS: dict[NamedLossFunction, Type[_Loss]] = {
    NamedLossFunction.GDW_NLL: GoldDistanceWeightedNLL,
    NamedLossFunction.NLL: WrappedNLLLoss,
}
CLASSIFICATION_LOSS_FUNCTION_DEFAULTS: dict[str, dict[str, Any]] = {
    NamedLossFunction.GDW_NLL: {"reduction": "mean"},
    NamedLossFunction.NLL: {"reduction": "mean"}
}


def get_classification_loss_function(loss_function_name: NamedLossFunction) -> tuple[str, _Loss]:
    try:
        loss_function_class: Type[_Loss] = CLASSIFICATION_LOSS_FUNCTIONS[loss_function_name]
        loss_function_defaults: dict[str, Any] = CLASSIFICATION_LOSS_FUNCTION_DEFAULTS[loss_function_name]
    except KeyError:
        raise ValueError(f"The loss function <{loss_function_name}> is not currently recognized.")
    loss_function: _Loss = loss_function_class(**loss_function_defaults)
    return loss_function_name, loss_function
