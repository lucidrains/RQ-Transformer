import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, reduce, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def check_shape(t, pattern, **kwargs):
    return rearrange(t, f'{pattern} -> {pattern}', **kwargs)

# helper classes

def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        mask_value = -torch.finfo(sim.dtype).max
        mask = torch.ones((i, j), dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(mask, mask_value)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# main class

class RQTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_spatial_seq_len,
        max_depth_seq_len,
        spatial_layers,
        depth_layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        pad_id = 0
    ):
        super().__init__()
        self.dim = dim
        self.max_spatial_seq_len = max_spatial_seq_len
        self.max_depth_seq_len = max_depth_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.spatial_start_token = nn.Parameter(torch.randn(dim))

        self.spatial_pos_emb = nn.Embedding(max_spatial_seq_len, dim)
        self.depth_pos_emb = nn.Embedding(max_depth_seq_len, dim)

        self.spatial_layers = nn.ModuleList([])
        for _ in range(spatial_layers):
            self.spatial_layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.depth_layers = nn.ModuleList([])
        for _ in range(depth_layers):
            self.depth_layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.to_logits = nn.Linear(dim, num_tokens)
        self.pad_id = pad_id

    def forward(self, ids, return_loss = False):
        check_shape(ids, 'b s d') # b - batch, s - spatial dimension, d - depth dimension

        b, space, depth, device = *ids.shape, ids.device
        assert space <= self.max_spatial_seq_len, 'spatial dimension is greater than the max_spatial_seq_len set'
        assert depth <= self.max_depth_seq_len, 'depth dimension is greater than the max_depth_seq_len set'

        tokens = self.token_emb(ids)

        spatial_pos = self.spatial_pos_emb(torch.arange(space, device = device))
        depth_pos = self.depth_pos_emb(torch.arange(depth, device = device))

        tokens_with_depth_pos = tokens + depth_pos

        # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions
        spatial_tokens = reduce(tokens_with_depth_pos, 'b s d f -> b s f', 'sum') + spatial_pos

        spatial_tokens = torch.cat((
            repeat(self.spatial_start_token, 'f -> b 1 f', b = b),
            spatial_tokens
        ), dim = -2)

        spatial_tokens = spatial_tokens[:, :-1]

        for attn, ff in self.spatial_layers:
            spatial_tokens = attn(spatial_tokens) + spatial_tokens
            spatial_tokens = ff(spatial_tokens) + spatial_tokens

        spatial_tokens = rearrange(spatial_tokens, 'b s f -> b s 1 f')

        # spatial tokens become the start tokens of the depth dimension

        depth_tokens = torch.cat((spatial_tokens, tokens_with_depth_pos), dim = -2)
        depth_tokens = depth_tokens[:, :, :-1]

        depth_tokens = rearrange(depth_tokens, '... n d -> (...) n d')

        for attn, ff in self.depth_layers:
            depth_tokens = attn(depth_tokens) + depth_tokens
            depth_tokens = ff(depth_tokens) + depth_tokens

        depth_tokens = rearrange(depth_tokens, '(b s) d f -> b s d f', b = b)
        logits = self.to_logits(depth_tokens)

        if not return_loss:
            return logits

        assert self.training

        preds = rearrange(logits, 'b s d c -> b c (s d)')
        labels = rearrange(ids, 'b s d -> b (s d)')

        loss = F.cross_entropy(preds, labels, ignore_index = self.pad_id)
        return loss