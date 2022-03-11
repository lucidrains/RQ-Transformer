## RQ-Transformer (wip)

Implementation of <a href="https://arxiv.org/abs/2203.01941">RQ Transformer</a>, which proposes a more efficient way of training multi-dimensional sequences autoregressively. This repository will only contain the transformer for now, and attempt to extend it to greater than 2 dimensions. You can use <a href="https://github.com/lucidrains/vector-quantize-pytorch#residual-vq">this vector quantization library</a> for the residual VQ.

## Citations

```bibtex
@unknown{unknown,
    author  = {Lee, Doyup and Kim, Chiheon and Kim, Saehoon and Cho, Minsu and Han, Wook-Shin},
    year    = {2022},
    month   = {03},
    title   = {Autoregressive Image Generation using Residual Quantization}
}
```
