"""
MSSR (lightweight) - Multi-Side Sequential Recommendation

This is a lightweight, RecBole-compatible implementation that uses
item sequences plus a single item-side feature (e.g., category) and
fuses them with a simple gate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


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


class MSSR(SequentialRecommender):
    """
    Lightweight MSSR:
    - Item transformer encoder
    - Single side feature (category) embedding with masked mean pooling
    - Gate to fuse item and side representations
    """

    input_type = "point"

    def __init__(self, config, dataset):
        super(MSSR, self).__init__(config, dataset)

        self.n_items = dataset.item_num
        self.embedding_size = _cfg(config, "embedding_size", 64)
        self.hidden_size = _cfg(config, "hidden_size", 64)
        self.num_layers = _cfg(config, "num_layers", 2)
        self.num_heads = _cfg(config, "num_heads", 2)
        self.inner_size = _cfg(config, "inner_size", self.hidden_size * 4)
        self.dropout_prob = _cfg(config, "hidden_dropout_prob", 0.1)
        self.max_seq_length = _cfg(config, "MAX_ITEM_LIST_LENGTH", 50)
        self.loss_type = _cfg(config, "loss_type", "CE")

        selected_features = _cfg(config, "selected_features", ["category"])
        if isinstance(selected_features, str):
            selected_features = [selected_features]
        self.feature_name = selected_features[0] if selected_features else None

        attr_hidden_size = _cfg(config, "attribute_hidden_size", self.hidden_size)
        if isinstance(attr_hidden_size, (list, tuple)):
            attr_hidden_size = attr_hidden_size[0]
        self.attr_hidden_size = attr_hidden_size

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

        self.item_layer_norm = nn.LayerNorm(self.hidden_size)
        self.attr_layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.attr_embedding = None
        self.attr_proj = None
        self.item_to_attr = None

        if self.feature_name and self.feature_name in dataset.field2token_id:
            n_attr = len(dataset.field2token_id[self.feature_name])
            self.attr_embedding = nn.Embedding(n_attr, self.attr_hidden_size, padding_idx=0)
            if self.attr_hidden_size != self.hidden_size:
                self.attr_proj = nn.Linear(self.attr_hidden_size, self.hidden_size)

            item_to_attr = torch.zeros(self.n_items, dtype=torch.long)
            if dataset.item_feat is not None and self.feature_name in dataset.item_feat.columns:
                item_ids = torch.tensor(dataset.item_feat["item_id"].values)
                feat_ids = torch.tensor(dataset.item_feat[self.feature_name].values)
                item_to_attr[item_ids] = feat_ids
            self.register_buffer("item_to_attr", item_to_attr)

        self.gate = nn.Linear(self.hidden_size * 2, 1)
        self.output_layer = nn.Linear(self.hidden_size, self.embedding_size)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        self.to(self.device)

    def _get_causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def _get_attr_output(self, item_seq, pos_emb, key_padding_mask):
        if self.attr_embedding is None or self.item_to_attr is None:
            return None
        attr_ids = self.item_to_attr[item_seq]
        attr_emb = self.attr_embedding(attr_ids)
        if self.attr_proj is not None:
            attr_emb = self.attr_proj(attr_emb)
        attr_emb = self.attr_layer_norm(attr_emb + pos_emb)
        attr_emb = self.dropout(attr_emb)

        valid_mask = (~key_padding_mask).float().unsqueeze(-1)
        attr_sum = torch.sum(attr_emb * valid_mask, dim=1)
        attr_len = valid_mask.sum(dim=1).clamp(min=1.0)
        return attr_sum / attr_len

    def forward(self, item_seq, item_seq_len):
        batch_size, seq_len = item_seq.shape
        pos_ids = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(pos_ids)

        item_emb = self.item_embedding(item_seq)
        if self.embedding_proj is not None:
            item_emb = self.embedding_proj(item_emb)
        item_emb = self.item_layer_norm(item_emb + pos_emb)
        item_emb = self.dropout(item_emb)

        attn_mask = self._get_causal_mask(seq_len, item_seq.device)
        key_padding_mask = item_seq == 0

        hidden = self.encoder(item_emb, attn_mask, key_padding_mask)
        item_output = hidden[torch.arange(batch_size, device=item_seq.device), item_seq_len - 1]

        attr_output = self._get_attr_output(item_seq, pos_emb, key_padding_mask)
        if attr_output is None:
            fused = item_output
        else:
            gate = torch.sigmoid(self.gate(torch.cat([item_output, attr_output], dim=-1)))
            fused = gate * item_output + (1.0 - gate) * attr_output

        return self.output_layer(fused)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            return self.loss_fct(pos_score, neg_score)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return self.loss_fct(logits, pos_items)

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
