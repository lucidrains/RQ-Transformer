<img src="./rq-transformer.png" width="500px"></img>

## RQ-Transformer

Implementation of <a href="https://arxiv.org/abs/2203.01941">RQ Transformer</a>, which proposes a more efficient way of training multi-dimensional sequences autoregressively. This repository will only contain the transformer for now, and attempt to extend it to greater than 2 dimensions. You can use <a href="https://github.com/lucidrains/vector-quantize-pytorch#residual-vq">this vector quantization library</a> for the residual VQ.

This type of axial autoregressive transformer should be compatible with <a href="https://github.com/lucidrains/nwt-pytorch">memcodes</a>, proposed in <a href="https://arxiv.org/abs/2106.04283">NWT</a>. It would likely also work well with <a href="https://github.com/lucidrains/vector-quantize-pytorch#multi-headed-vq">multi-headed VQ</a>

## Install

```bash
$ pip install RQ-transformer
```

## Usage

```python
import torch
from rq_transformer import RQTransformer

model = RQTransformer(
    num_tokens = 16000,             # number of tokens, in the paper they had a codebook size of 16k
    dim = 512,                      # transformer model dimension
    max_spatial_seq_len = 1024,     # maximum positions along space
    depth_seq_len = 4,              # number of positions along depth (residual quantizations in paper)
    spatial_layers = 8,             # number of layers for space
    depth_layers = 4,               # number of layers for depth
    dim_head = 64,                  # dimension per head
    heads = 8,                      # number of attention heads
)

x = torch.randint(0, 16000, (1, 1024, 4))

loss = model(x, return_loss = True)
loss.backward()

# then after much training

logits = model(x)

# and sample from the logits accordingly
# or you can use the generate function

sampled = model.generate(temperature = 0.9, filter_thres = 0.9) # (1, 1024, 4)
```

I also think there is something deeper going on, and have generalized this to any number of dimensions. You can use it by importing the `HierarchicalCausalTransformer`

```python
import torch
from rq_transformer import HierarchicalCausalTransformer

model = HierarchicalCausalTransformer(
    num_tokens = 16000,                   # number of tokens
    dim = 512,                            # feature dimension
    dim_head = 64,                        # dimension of attention heads
    heads = 8,                            # number of attention heads
    depth = (4, 4, 2),                    # 3 stages (but can be any number) - transformer of depths 4, 4, 2
    max_seq_len = (16, 4, 5)              # the maximum sequence length of first, stage, then the fixed sequence length of all subsequent stages
).cuda()

x = torch.randint(0, 16000, (1, 10, 4, 5)).cuda()

loss = model(x, return_loss = True)
loss.backward()
```

## Todo

- [x] take care of sampling with generate method
- [x] generalize to any number of preceding dimension (full hierarchical or axial autoregressive transformer)

## Citations

```bibtex
@unknown{unknown,
    author  = {Lee, Doyup and Kim, Chiheon and Kim, Saehoon and Cho, Minsu and Han, Wook-Shin},
    year    = {2022},
    month   = {03},
    title   = {Autoregressive Image Generation using Residual Quantization}
}
```
