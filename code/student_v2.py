"""Architecture variant: RMSNorm + SwiGLU + zero-initialized residual projections.

Same interfaces, regularization and EMA machinery as student.py, so it can be
dropped into train.py/evaluate.py/distill.py via --implementation student_v2.

Changes vs the classroom baseline (diagnosis item 5, "naive architecture"):
  * RMSNorm instead of LayerNorm (cheaper, no mean-centering, standard in
    modern small LMs).
  * SwiGLU MLP. To keep the parameter count matched to the 4x-width GELU MLP
    (which uses 8W^2 weights), the hidden size is h = 8W/3 rounded up to a
    multiple of 8, since SwiGLU has three matrices (gate, up, down).
  * Residual-branch output projections (attention proj and MLP down-projection)
    are zero-initialized (GPT-2 trick): each block starts as the identity, which
    stabilizes early optimization of deeper residual stacks.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def swiglu_hidden(width, multiple=8):
    """Parameter-matched SwiGLU hidden size: 3 * W * h == 8 * W^2."""
    return multiple * round((8 * width / 3) / multiple)


class SwiGLU(nn.Module):
    def __init__(self, width, dropout=0.):
        super().__init__()
        hidden = swiglu_hidden(width)
        self.gate_up = nn.Linear(width, 2 * hidden)
        self.down = nn.Linear(hidden, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.drop(self.down(F.silu(gate) * up))


class BlockV2(nn.Module):
    def __init__(self, width=128, heads=4, dropout=0.):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.mlp = SwiGLU(width, dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.)
        x = x + self.resid_drop(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        return x + self.mlp(self.norm2(x))


class StudentGPTV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        dropout = float(config.get('dropout', 0.))
        self.ema_decay = float(config.get('ema_decay', 0.))
        self.temperature = float(config.get('temperature', 1.))
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([BlockV2(width, config['heads'], dropout) for _ in range(config['depth'])])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        # Zero-init the residual output projections of every block (identity start).
        for block in self.blocks:
            nn.init.zeros_(block.proj.weight)
            nn.init.zeros_(block.mlp.down.weight)
        self.head.weight = self.token.weight  # tied embedding and output head
        if self.ema_decay > 0.:
            self.register_buffer('ema_step', torch.zeros((), dtype=torch.long))
            for name, parameter in self.named_parameters():
                self.register_buffer('ema_' + name.replace('.', '_'), torch.empty_like(parameter))

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

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
        """Evaluation interface: normalized log probabilities (EMA weights, temperature)."""
        if self._ema_weights_ready():
            backup = self._load_ema_weights()
            try:
                return F.log_softmax(self(ids).float() / self.temperature, dim=-1)
            finally:
                self._restore_weights(backup)
        return F.log_softmax(self(ids).float() / self.temperature, dim=-1)


def build_model(config):
    return StudentGPTV2(config)
