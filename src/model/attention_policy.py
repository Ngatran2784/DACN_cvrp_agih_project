import torch
import torch.nn as nn


class AttentionPolicy(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128):
        super().__init__()

        self.node_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        """
        x: [batch, n_customers, input_dim]
        mask: [batch, n_customers]
              True = node không được chọn
        """
        h = self.node_embed(x)

        context = h.mean(dim=1, keepdim=True)
        h = h + torch.tanh(self.context_proj(context))

        logits = self.score(h).squeeze(-1)

        if mask is not None:
            logits = logits.masked_fill(mask, -1e9)

        return logits