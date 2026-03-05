"""
PAtt (lightweight) - Probabilistic Attention for Sequential Recommendation

This is a lightweight, RecBole-compatible approximation that injects
DPP-inspired diversity into self-attention by penalizing similarity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender


def _cfg(config, key, default):
    try:
        return config[key] if key in config else default
    except Exception:
        return default

class DiversityAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, diversity_gamma):
        super(DiversityAttention, self).__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.diversity_gamma = diversity_gamma

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        batch_size, seq_len, _ = x.shape

        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if self.diversity_gamma > 0:
            normed = F.normalize(x, dim=-1)
            sim = torch.matmul(normed, normed.transpose(-1, -2))
            sim = sim.unsqueeze(1)
            scores = scores - self.diversity_gamma * sim

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        elif attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)
        elif attn_mask.dim() == 4 and attn_mask.size(1) not in (1, self.num_heads):
            attn_mask = attn_mask[:, :1, :, :]
        scores = scores.masked_fill(attn_mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(context)


class PAttLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, inner_size, dropout, diversity_gamma):
        super(PAttLayer, self).__init__()
        self.attn = DiversityAttention(hidden_size, num_heads, dropout, diversity_gamma)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, attn_mask):
        attn_out = self.attn(x, attn_mask)
        x = self.attn_norm(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)
        return x


class PAtt(SequentialRecommender):
    """
    Lightweight PAtt with DPP-inspired diversity penalty in attention.
    """

    input_type = "point"

    def __init__(self, config, dataset):
        super(PAtt, self).__init__(config, dataset)

        self.n_items = dataset.item_num
        self.embedding_size = _cfg(config, "embedding_size", 64)
        self.hidden_size = _cfg(config, "hidden_size", 64)
        self.num_layers = _cfg(config, "num_layers", 2)
        self.num_heads = _cfg(config, "num_heads", 2)
        self.inner_size = _cfg(config, "inner_size", self.hidden_size * 4)
        self.dropout_prob = _cfg(config, "hidden_dropout_prob", 0.1)
        self.max_seq_length = _cfg(config, "MAX_ITEM_LIST_LENGTH", 50)
        self.loss_type = _cfg(config, "loss_type", "CE")
        self.diversity_gamma = _cfg(config, "diversity_gamma", 0.1)

        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        if self.embedding_size != self.hidden_size:
            self.embedding_proj = nn.Linear(self.embedding_size, self.hidden_size)
        else:
            self.embedding_proj = None

        self.layers = nn.ModuleList(
            [
                PAttLayer(
                    self.hidden_size,
                    self.num_heads,
                    self.inner_size,
                    self.dropout_prob,
                    self.diversity_gamma,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.output_layer = nn.Linear(self.hidden_size, self.embedding_size)

        self.to(self.device)

    def _get_attention_mask(self, item_seq, item_seq_len):
        batch_size, seq_len = item_seq.shape
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=item_seq.device), diagonal=1).bool()
        padding_mask = torch.arange(seq_len, device=item_seq.device).unsqueeze(0) >= item_seq_len.unsqueeze(1)
        padding_mask = padding_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        return causal_mask.unsqueeze(0) | padding_mask

    def forward(self, item_seq, item_seq_len):
        batch_size, seq_len = item_seq.shape
        pos_ids = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(pos_ids)

        item_emb = self.item_embedding(item_seq)
        if self.embedding_proj is not None:
            item_emb = self.embedding_proj(item_emb)

        hidden = self.layer_norm(item_emb + pos_emb)
        hidden = self.dropout(hidden)
        attn_mask = self._get_attention_mask(item_seq, item_seq_len)

        for layer in self.layers:
            hidden = layer(hidden, attn_mask)

        last_hidden = hidden[torch.arange(batch_size, device=item_seq.device), item_seq_len - 1]
        return self.output_layer(last_hidden)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return F.cross_entropy(logits, pos_items)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        return torch.mul(seq_output, test_item_emb).sum(dim=1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding.weight
        return torch.matmul(seq_output, test_item_emb.transpose(0, 1))
