import torch
from torch import nn
import math


class LoraLayer(nn.Module):

    def __init__(self, weight, r, alpha = 1, dropout_prob = 0, fan_in_fan_out = False):
        super().__init__()

        if fan_in_fan_out:
            self.in_features = weight.shape[0]
            self.out_features = weight.shape[1]
        else:
            self.in_features = weight.shape[1]
            self.out_features = weight.shape[0]
        self.alpha = alpha
        self.fan_in_fan_out = fan_in_fan_out

        if dropout_prob > 0.:
            self.lora_dropout = nn.Dropout(p=dropout_prob)
        else:
            self.lora_dropout = nn.Identity()

        self._init_lora(r, weight_dtype=weight.dtype)

    def _init_lora(self, r, weight_dtype = None):
        # Actual trainable parameters
        if r > 0:
            if weight_dtype == None:
                weight_dtype = self.lora_A.dtype
            self.register_parameter(
                'lora_A',
                nn.Parameter(torch.empty((self.in_features, r), dtype=weight_dtype))
            )
            self.register_parameter(
                'lora_B',
                nn.Parameter(torch.zeros((r, self.out_features), dtype=weight_dtype))
            )
            self.scaling = self.alpha / r
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            try:
                # ensure parameters do not exist if they are zero
                delattr(self, "lora_A")
                delattr(self, "lora_B")
                delattr(self, "scaling")
            except AttributeError:
                pass
        self.r = r

    def change_lora_rank(self, new_rank):
        if new_rank != self.r:
            self._init_lora(new_rank)

    def forward(self, X):
        if self.r == 0:
            return X
        else:
            lora = self.lora_dropout(self.lora_A @ self.lora_B * self.scaling)
            if not self.fan_in_fan_out:
                lora = lora.T
            return X + lora