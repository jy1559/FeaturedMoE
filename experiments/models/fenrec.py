"""
FENRec (lightweight) - Feature-Enhanced Sequential Recommendation

This lightweight version keeps a SASRec-style encoder and adds
contrastive learning with simple in-batch negatives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender


def _cfg(config, key, default):
    try:
        return config[key] if key in config else default
    except Exception:
        return default

class SimpleTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, inner_size, dropout):
        super(SimpleTransformerLayer, self).__init__()
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, attn_mask, key_padding_mask):
        attn_out, _ = self.attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.attn_norm(x + self.attn_dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)
        return x


class SimpleTransformerEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, inner_size, dropout):
        super(SimpleTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                SimpleTransformerLayer(hidden_size, num_heads, inner_size, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, attn_mask, key_padding_mask):
        for layer in self.layers:
            x = layer(x, attn_mask, key_padding_mask)
        return x


class FENRec(SequentialRecommender):
    """
    Lightweight FENRec:
    - SASRec-style encoder
    - Contrastive loss between original and dropout-augmented views
    """

    input_type = "point"

    def __init__(self, config, dataset):
        super(FENRec, self).__init__(config, dataset)

        self.n_items = dataset.item_num
        self.embedding_size = _cfg(config, "embedding_size", 64)
        self.hidden_size = _cfg(config, "hidden_size", 64)
        self.num_layers = _cfg(config, "num_layers", 2)
        self.num_heads = _cfg(config, "num_heads", 2)
        self.inner_size = _cfg(config, "inner_size", self.hidden_size * 4)
        self.dropout_prob = _cfg(config, "hidden_dropout_prob", 0.1)
        self.max_seq_length = _cfg(config, "MAX_ITEM_LIST_LENGTH", 50)
        self.loss_type = _cfg(config, "loss_type", "CE")

        self.cl_weight = _cfg(config, "cl_weight", 0.1)
        self.cl_temperature = _cfg(config, "cl_temperature", 0.2)
        self.aug_dropout_prob = _cfg(config, "aug_dropout_prob", 0.1)

        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        if self.embedding_size != self.hidden_size:
            self.embedding_proj = nn.Linear(self.embedding_size, self.hidden_size)
        else:
            self.embedding_proj = None

        self.encoder = SimpleTransformerEncoder(
            self.num_layers,
            self.hidden_size,
            self.num_heads,
            self.inner_size,
            self.dropout_prob,
        )

        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.aug_dropout = nn.Dropout(self.aug_dropout_prob)
        self.output_layer = nn.Linear(self.hidden_size, self.embedding_size)

        self.to(self.device)

    def _get_causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def _encode(self, item_seq, item_seq_len, augment=False):
        batch_size, seq_len = item_seq.shape
        pos_ids = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(pos_ids)

        item_emb = self.item_embedding(item_seq)
        if self.embedding_proj is not None:
            item_emb = self.embedding_proj(item_emb)

        if augment:
            item_emb = self.aug_dropout(item_emb)

        hidden = self.layer_norm(item_emb + pos_emb)
        hidden = self.dropout(hidden)

        attn_mask = self._get_causal_mask(seq_len, item_seq.device)
        key_padding_mask = item_seq == 0
        hidden = self.encoder(hidden, attn_mask, key_padding_mask)

        last_hidden = hidden[torch.arange(batch_size, device=item_seq.device), item_seq_len - 1]
        return last_hidden

    def _contrastive_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = torch.matmul(z1, z2.transpose(0, 1)) / self.cl_temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        loss_1 = F.cross_entropy(logits, labels)
        loss_2 = F.cross_entropy(logits.transpose(0, 1), labels)
        return 0.5 * (loss_1 + loss_2)

    def forward(self, item_seq, item_seq_len, augment=False):
        hidden = self._encode(item_seq, item_seq_len, augment=augment)
        return self.output_layer(hidden)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = F.cross_entropy(logits, pos_items)

        if self.cl_weight > 0:
            aug_output = self._encode(item_seq, item_seq_len, augment=True)
            cl_loss = self._contrastive_loss(seq_output, aug_output)
            loss = loss + self.cl_weight * cl_loss

        return loss

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
