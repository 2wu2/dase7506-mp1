"""Student implementation for MP1.

Diagnosed limitations of the classroom baseline (model.py):
  1. Capacity: 4 blocks x width 128 (1.09M params) leaves most of the 64 MiB
     inference-asset budget unused.
  2. Regularization: no dropout at all, while WikiText-2 train text is only
     ~2M tokens; the baseline recipe (~5 epochs) stops early to avoid
     overfitting instead of regularizing.
  3. Noise: a single end-of-training checkpoint is a noisy sample of the
     loss surface; there is no weight averaging.

Mechanisms implemented here (all config-driven and ablatable):
  * Scaled-up GPT (configurable width/depth/heads), tied embedding-head kept.
  * Dropout regularization (embedding, attention, residual/MLP) for the
    many-epoch regime that the small dataset requires.
  * Exponential moving average (EMA) of the weights, updated by the trainer
    after every optimizer step; `predict_log_probs` evaluates with the
    bias-corrected EMA weights (swapped in and restored under no_grad, so
    training is unaffected and evaluation stays strictly causal and
    stateless across windows).

The two required interfaces are preserved exactly:
  forward(ids)            -> unnormalized logits [batch, time, 2048]
  predict_log_probs(ids)  -> normalized log probabilities [batch, time, 2048]
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    """Pre-norm transformer block with causal self-attention and dropout."""

    def __init__(self, width=128, heads=4, dropout=0.):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.resid_drop = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        # Each position attends only to itself and earlier input tokens.
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.)
        x = x + self.resid_drop(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        return x + self.resid_drop(self.mlp(self.norm2(x)))


class StudentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        dropout = float(config.get('dropout', 0.))
        self.ema_decay = float(config.get('ema_decay', 0.))  # 0 disables EMA.
        # Prediction temperature, selected on validation (1.0 = no calibration).
        self.temperature = float(config.get('temperature', 1.))
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(width, config['heads'], dropout) for _ in range(config['depth'])])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        # Output head may be tied to the token embedding (default) or a free
        # matrix (config 'tied': false, +vocab*width parameters).
        if config.get('tied', True):
            self.head.weight = self.token.weight  # tied embedding and output head
        # EMA shadow weights, stored as buffers so they persist in checkpoints.
        # Registered only when EMA is enabled; `ema_step` counts EMA updates
        # (0 means "not trained, use raw weights"). Slim submission checkpoints
        # use ema_decay=0, carry no buffers, and hold the EMA weights directly.
        if self.ema_decay > 0.:
            self.register_buffer('ema_step', torch.zeros((), dtype=torch.long))
            for name, parameter in self.named_parameters():  # unique params only
                self.register_buffer('ema_' + name.replace('.', '_'),
                                     torch.empty_like(parameter))

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.emb_drop(self.token(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device)))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    @torch.no_grad()
    def update_ema(self):
        """Called by the trainer after each optimizer step."""
        if self.ema_decay <= 0.:
            return
        self.ema_step += 1
        for name, parameter in self.named_parameters():
            shadow = getattr(self, 'ema_' + name.replace('.', '_'))
            if int(self.ema_step) == 1:
                shadow.copy_(parameter)
            else:
                shadow.mul_(self.ema_decay).add_(parameter.detach(), alpha=1. - self.ema_decay)

    def _ema_weights_ready(self):
        return self.ema_decay > 0. and int(self.ema_step) > 0

    @torch.no_grad()
    def _load_ema_weights(self):
        """Swap bias-corrected EMA weights in; returns the backed-up raw weights."""
        correction = 1. - self.ema_decay ** int(self.ema_step)
        backup = []
        for name, parameter in self.named_parameters():
            backup.append(parameter.detach().clone())
            parameter.copy_(getattr(self, 'ema_' + name.replace('.', '_')) / correction)
        return backup

    @torch.no_grad()
    def _restore_weights(self, backup):
        for (_, parameter), saved in zip(self.named_parameters(), backup):
            parameter.copy_(saved)

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities, with no access to targets.

        A prediction at position t uses ids[:, :t+1] and nothing later.
        Temporary state (the EMA weight swap) is created and destroyed within
        this call; every evaluation window starts fresh. Logits are scaled by
        the validation-selected temperature before normalization.
        """
        if self._ema_weights_ready():
            backup = self._load_ema_weights()
            try:
                return F.log_softmax(self(ids).float() / self.temperature, dim=-1)
            finally:
                self._restore_weights(backup)
        return F.log_softmax(self(ids).float() / self.temperature, dim=-1)


def build_model(config):
    """Student model factory used by train.py/evaluate.py via --implementation student."""
    if 'members' in config:  # ensemble of independent causal sub-models
        return EnsembleGPT(config)
    return StudentGPT(config)


class EnsembleGPT(nn.Module):
    """Geometric-mean ensemble of N independently causal sub-models.

    Each member is a StudentGPT configured independently (different seeds,
    dropout, tied/untied, training recipe). The state_dict stores each
    member's parameters under ``members.<i>.<param>`` keys, so total
    inference asset is the sum of member assets and total evaluation time
    is the sum of member times.

    predict_log_probs returns the arithmetic mean (in natural units) of the
    members' log-probabilities -- i.e. the log of the geometric mean of
    their per-token distributions. This is the standard Bayesian-style
    model average and tends to dominate logit-averaging on BPB.

    Each member remains strictly causal (SDPA with is_causal=True) and
    stateless across windows (each predict_log_probs call only sees the
    input ids). The ensemble wrapper adds no new state.
    """

    def __init__(self, config):
        super().__init__()
        self.context = 256
        self.config = config
        self.temperature = float(config.get('temperature', 1.))
        self.members = nn.ModuleList([
            StudentGPT(sub_cfg) for sub_cfg in config['members']
        ])

    def forward(self, ids):
        return sum(member.forward(ids) for member in self.members) / len(self.members)

    @torch.no_grad()
    def predict_log_probs(self, ids):
        log_probs = torch.stack(
            [member.predict_log_probs(ids) for member in self.members], dim=0)
        return log_probs.logsumexp(dim=0) - math.log(len(self.members))
