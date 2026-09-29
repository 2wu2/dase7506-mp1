"""Two-model log-probability ensemble.

Both members are StudentGPT models trained from scratch on the supplied
training text with different seeds; the ensemble averages their per-token
log-probabilities (a geometric mixture, weighted) and renormalizes. Each
member is strictly causal and stateless across windows, so the ensemble is
too: a prediction at position t uses only ids[:, :t+1].

Used with a merged checkpoint built by merge_ensemble.py:
  config = {'members': [member_config, ...], 'weights': [...], 'temperature': T}
  state  = {'m0.<key>': ..., 'm1.<key>': ...}
"""
import torch
from torch import nn
from torch.nn import functional as F

from student import StudentGPT


class Ensemble(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = 256
        self.members = nn.ModuleList([StudentGPT(dict(c, ema_decay=0.))
                                      for c in config['members']])
        weights = config.get('weights', [1.] * len(self.members))
        total = float(sum(weights))
        self.register_buffer('mixture_weights',
                             torch.tensor([w / total for w in weights], dtype=torch.float32))
        self.temperature = float(config.get('temperature', 1.))

    def forward(self, ids):
        """Training interface (unused at evaluation): summed member logits."""
        return sum(w * m(ids).float() for w, m in zip(self.mixture_weights, self.members))

    def predict_log_probs(self, ids):
        """Evaluation interface: weighted geometric mixture of member log-probs."""
        with torch.no_grad():
            mixed = None
            for w, member in zip(self.mixture_weights, self.members):
                logp = F.log_softmax(member(ids).float() / self.temperature, dim=-1)
                mixed = w * logp if mixed is None else mixed + w * logp
            return F.log_softmax(mixed, dim=-1)  # renormalize the mixture


def build_model(config):
    return Ensemble(config)
