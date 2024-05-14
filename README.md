# Pytora

A minimal pytorch implementation for lora via [pytorch parametrizations](https://pytorch.org/tutorials/intermediate/parametrizations.html)

## Installation

```pip install pytora```

## Usage

```
from pytora import apply_lora

apply_lora(
    model,
    lora_r = 4,
    lora_alpha = 1,
    lora_dropout = 0.0
)

```

A simple example use case with a pretrained huggingface model can be found in demo.ipynb