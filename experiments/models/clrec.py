"""
CLRec - Contrastive Learning for Sequential Recommendation

Implements contrastive learning framework for session-based recommendation.
Uses data augmentation and InfoNCE loss for better representation learning.

Key components:
- Sequential embedding via LSTM/GRU
- Multi-view data augmentation (crop, mask)
- Contrastive loss (InfoNCE) for representation alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
import math


class CLRec(SequentialRecommender):
    """
    CLRec: Contrastive Learning for Sequential Recommendation
    
    Learns robust representations by contrasting augmented views of sequences.
    Uses LSTM encoder with augmentation and InfoNCE loss.
    """

    input_type = "point"  # RecBole requirement

    def __init__(self, config, dataset):
        super(CLRec, self).__init__(config, dataset)
        
        self.n_items = dataset.item_num
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers'] if 'num_layers' in config else 1
        self.dropout_prob = config['hidden_dropout_prob'] if 'hidden_dropout_prob' in config else 0.0
        self.tau = config['tau'] if 'tau' in config else 0.07  # temperature parameter
        self.lambda_cl = config['lambda_cl'] if 'lambda_cl' in config else 0.1  # weight for CL loss
        self.aug_type = config['aug_type'] if 'aug_type' in config else 'crop'  # augmentation type
        
        # Item embedding
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_prob if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.embedding_size)
        )
        
        # Output layer for prediction
        self.output_layer = nn.Linear(self.hidden_size, self.embedding_size)
        
        self.loss_type = config['loss_type'] if 'loss_type' in config else 'CE'
        self.to(self.device)

    def forward(self, item_seq, item_seq_len):
        """
        Forward pass for prediction.
        
        Args:
            item_seq: (batch_size, seq_len) long tensor
            item_seq_len: (batch_size,) long tensor
        
        Returns:
            seq_output: (batch_size, embedding_size) - sequence representation
        """
        item_emb = self.item_embedding(item_seq)
        
        # Pack and LSTM
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            item_emb,
            item_seq_len.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_out, (hidden, _) = self.lstm(packed_emb)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        # Get last hidden state
        batch_size = item_seq.size(0)
        last_hidden = lstm_out[range(batch_size), item_seq_len - 1]
        
        # Output projection
        seq_output = self.output_layer(last_hidden)
        
        return seq_output

    def _augment_sequence(self, item_seq, item_seq_len, aug_type='crop'):
        """
        Augment sequence for contrastive learning.
        
        Args:
            item_seq: (batch_size, seq_len)
            item_seq_len: (batch_size,)
            aug_type: 'crop', 'mask', or 'reorder'
        
        Returns:
            augmented_seq: (batch_size, seq_len)
        """
        batch_size, seq_len = item_seq.shape
        
        if aug_type == 'crop':
            # Random crop: keep last k items where k ~ U[0.5*seq_len, seq_len]
            aug_seq = item_seq.clone()
            for i in range(batch_size):
                length = item_seq_len[i].item()
                crop_len = max(1, int(length * (0.5 + 0.5 * torch.rand(1).item())))
                start_idx = length - crop_len
                # Pad with zeros at the beginning
                aug_seq[i, :start_idx] = 0
        
        elif aug_type == 'mask':
            # Random masking: randomly set items to 0 with probability p
            aug_seq = item_seq.clone()
            mask_prob = 0.15
            for i in range(batch_size):
                length = item_seq_len[i].item()
                mask = torch.rand(length) < mask_prob
                aug_seq[i, :length][mask] = 0
        
        elif aug_type == 'reorder':
            # Random reorder: shuffle last k items
            aug_seq = item_seq.clone()
            for i in range(batch_size):
                length = item_seq_len[i].item()
                reorder_len = max(1, int(length * 0.2))
                start_idx = length - reorder_len
                if reorder_len > 1:
                    perm = torch.randperm(reorder_len)
                    aug_seq[i, start_idx:length] = item_seq[i, start_idx:length][perm]
        
        else:
            aug_seq = item_seq.clone()
        
        return aug_seq

    def _get_contrastive_rep(self, item_seq, item_seq_len):
        """
        Get representation for contrastive loss.
        
        Args:
            item_seq: (batch_size, seq_len)
            item_seq_len: (batch_size,)
        
        Returns:
            rep: (batch_size, embedding_size) representation
        """
        item_emb = self.item_embedding(item_seq)
        
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            item_emb,
            item_seq_len.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_out, (hidden, _) = self.lstm(packed_emb)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        batch_size = item_seq.size(0)
        last_hidden = lstm_out[range(batch_size), item_seq_len - 1]
        
        # Project to contrastive space
        rep = self.projection_head(last_hidden)
        rep = F.normalize(rep, dim=1)
        
        return rep

    def _contrastive_loss(self, z1, z2):
        """
        Compute InfoNCE contrastive loss.
        
        Args:
            z1: (batch_size, embedding_size) representation 1
            z2: (batch_size, embedding_size) representation 2
        
        Returns:
            loss: scalar
        """
        batch_size = z1.size(0)
        
        # Similarity matrix
        sim = torch.matmul(z1, z2.t())  # (batch_size, batch_size)
        sim = sim / self.tau
        
        # Labels: diagonal elements are positives
        labels = torch.arange(batch_size, device=z1.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)
        return loss / 2

    def calculate_loss(self, interaction):
        """
        Calculate loss combining CE loss and contrastive loss.
        
        Args:
            interaction: RecBole interaction dict
        
        Returns:
            loss: scalar tensor
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        
        # Main task: next item prediction
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        
        if self.loss_type.upper() == 'CE':
            ce_loss = F.cross_entropy(logits, pos_items)
        else:
            ce_loss = F.cross_entropy(logits, pos_items)
        
        # Contrastive learning task
        # Generate two augmented views
        aug1 = self._augment_sequence(item_seq, item_seq_len, self.aug_type)
        aug2 = self._augment_sequence(item_seq, item_seq_len, self.aug_type)
        
        # Get representations
        z1 = self._get_contrastive_rep(aug1, item_seq_len)
        z2 = self._get_contrastive_rep(aug2, item_seq_len)
        
        # Contrastive loss
        cl_loss = self._contrastive_loss(z1, z2)
        
        # Combined loss
        total_loss = ce_loss + self.lambda_cl * cl_loss
        
        return total_loss

    def predict(self, interaction):
        """
        Predict score for a specific item (used when full_sort_predict raises NotImplementedError).
        
        Args:
            interaction: RecBole interaction dict with ITEM_ID
        
        Returns:
            scores: (batch_size,) - score for the specific item
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        
        return scores
    
    def full_sort_predict(self, interaction):
        """
        Predict scores for all items (used for full-sort evaluation).
        
        Args:
            interaction: RecBole interaction dict
        
        Returns:
            scores: (batch_size, n_items)
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        
        return scores
