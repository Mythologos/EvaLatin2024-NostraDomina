from inspect import getfullargspec
from math import inf
from typing import Any, Union

from torch import optim
from torch.optim import Optimizer


def get_optimizer(optimizer_name: str):
    if hasattr(optim, optimizer_name):
        optimizer: Optimizer = getattr(optim, optimizer_name)
        if not isinstance(Optimizer, optimizer.__class__):
            raise ValueError(f"The item selected is in torch.optim, but it is not a valid optimizer.")
    else:
        raise ValueError(f"The optimizer with name <{optimizer_name}> is currently not supported by PyTorch. "
                         f"Please try again.")
    return optimizer


def define_optimizer_args(optimizer: Optimizer, general_kwargs: dict[str, Any]) -> dict[str, Any]:
    current_optimizer_kwargs: dict[str, Any] = {}

    # The below uses function inspection so that only relevant arguments are added to optimizer_args.
    # It requires using consistent nomenclature,
    # and it assumes that PyTorch is standardized enough among its optimizers for this to work.
    # It avoids examining the "self" and "params" arguments, hence the '[2:]'.
    optional_optimizer_parameters: list[str] = getfullargspec(optimizer.__init__).args[2:]
    for parameter in optional_optimizer_parameters:
        if general_kwargs.get(parameter, None) is not None:
            current_optimizer_kwargs[parameter] = general_kwargs[parameter]

    return current_optimizer_kwargs


def parse_training_interval(training_interval: str) -> Union[int, float]:
    if training_interval.isdigit() is True:
        interval: int = int(training_interval)
        if interval == 0:
            raise ValueError("The interval must be a positive integer.")
    elif training_interval == "inf":
        interval: float = inf
    else:
        raise ValueError(f"Could not parse the interval <{training_interval}>. Please use a positive integer or inf")

    return interval


def define_training_loop_kwargs(general_kwargs: dict[str, Any]) -> dict[str, Any]:
    optimizer: Optimizer = general_kwargs["optimizer"]
    training_loop_kwargs: dict[str, Any] = {
        "optimizer": optimizer,
        "optimizer_kwargs": define_optimizer_args(optimizer, general_kwargs)
    }

    # We can define a maximum number of epochs, a patience, or both.
    # If epochs is defined (but patience is not), then there's a maximum number of epochs for which training occurs.
    # If patience is defined (but epochs is not), then there's an eventual convergence based on a model's performance.
    # If both are defined, then this is effectively an early stopping strategy.
    if general_kwargs.get("epochs", None) is None and general_kwargs.get("patience", None) is None:
        raise ValueError("At least one of --epochs or --patience must be defined. Please try again.")
    else:
        if general_kwargs.get("epochs", None) is not None:
            training_loop_kwargs["epochs"] = general_kwargs["epochs"]
        else:
            training_loop_kwargs["epochs"] = inf

        if general_kwargs.get("patience", None) is not None:
            training_loop_kwargs["patience"] = general_kwargs["patience"]
        else:
            training_loop_kwargs["patience"] = inf

    _, training_loop_kwargs["loss_function"] = general_kwargs["loss_function"]
    training_loop_kwargs["tqdm"] = general_kwargs["tqdm"]
    training_loop_kwargs["training_interval"] = parse_training_interval(general_kwargs["training_interval"])
    return training_loop_kwargs


def define_evaluation_loop_kwargs(general_kwargs: dict[str, Any]) -> dict[str, Any]:
    evaluation_kwargs: dict[str, Any] = {"tqdm": general_kwargs["tqdm"]}
    return evaluation_kwargs
