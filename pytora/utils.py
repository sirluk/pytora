import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization, remove_parametrizations
import importlib
from functools import partial
from collections import OrderedDict

from .lora_layer import LoraLayer

from typing import Optional


_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
if _TRANSFORMERS_AVAILABLE:
    from transformers import Conv1D


def module_name_check(
    name: str,
    include_names: Optional[list[str]] = None,
    exclude_names: Optional[list[str]] = None,
):
    if include_names is not None:
        inclusion = [n == name[-len(n):] for n in include_names]
        return any(inclusion) 

    if exclude_names is not None:
        exclusion = [n == name[-len(n):] for n in exclude_names]
        return not any(exclusion)

    return True


@torch.no_grad()
def apply_lora(
    model: nn.Module,
    lora_r: int = 4,
    lora_alpha: int = 1,
    lora_dropout: float = 0.0,
    include_names: Optional[list[str]] = None,
    exclude_names: Optional[list[str]] = None
):
    check = partial(
        module_name_check,
        include_names = include_names,
        exclude_names = exclude_names
    )

    for name, module in model.named_modules():
        
        if check(name):
            if type(module) == torch.nn.Linear:
                l = LoraLayer(
                    weight = module.weight,
                    r = lora_r,
                    alpha = lora_alpha,
                    dropout_prob = lora_dropout
                )
                register_parametrization(module, "weight", l)
                module.weight.requires_grad = False
            elif _TRANSFORMERS_AVAILABLE and type(module) == Conv1D:
                # same as linear layer, was implemented to keep gpt2 style
                l = LoraLayer(
                    weight = module.weight,
                    r = lora_r,
                    alpha = lora_alpha,
                    dropout_prob = lora_dropout,
                    fan_in_fan_out = True
                )
                register_parametrization(module, "weight", l)
                module.weight.requires_grad = False


@torch.no_grad()
def remove_lora(
    model: nn.Module,
    merge: bool = True,
    return_lora_state_dict: bool = True,
    pidx: int = 0
):
    lora_state_dict = None
    if return_lora_state_dict:
        lora_state_dict = []
    
    for n, m in model.named_modules():
        if not hasattr(m, "parametrizations"):
            continue
        elif return_lora_state_dict:
            lora_state_dict.extend([
                (f'{n}.parametrizations.weight.{pidx}.lora_A', m.parametrizations.weight[pidx].lora_A),
                (f'{n}.parametrizations.weight.{pidx}.lora_B', m.parametrizations.weight[pidx].lora_B)
            ])
        remove_parametrizations(m, "weight", leave_parametrized=merge)
    
    return OrderedDict(lora_state_dict)