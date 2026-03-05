"""
BiLSTM - Bidirectional LSTM for Sequential Recommendation

Simple bidirectional LSTM model for session-based recommendation.
Processes item sequences in both forward and backward directions.
"""

import torch
import torch.nn as nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import MLPLayers


class BiLSTM(SequentialRecommender):
    """
    BiLSTM: Bidirectional LSTM for session-based recommendation.
    
    Processes sequential item embeddings through bidirectional LSTM,
    then uses the final hidden state to predict next item.
    """
    
    input_type = "point"  # RecBole requirement

    def __init__(self, config, dataset):
        super(BiLSTM, self).__init__(config, dataset)
        
        self.n_items = dataset.item_num
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers'] if 'num_layers' in config else 1
        self.dropout_prob = config['hidden_dropout_prob'] if 'hidden_dropout_prob' in config else 0.0
        self.bidirectional = config['bidirectional'] if 'bidirectional' in config else True
        self.use_attention = config['use_attention'] if 'use_attention' in config else False
        
        # Item embedding
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_prob if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # Output projection to item embeddings for softmax
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        
        if self.use_attention:
            # Simple attention layer
            self.attention = nn.Linear(lstm_output_size, 1)
            self.output_layer = nn.Linear(lstm_output_size, self.embedding_size)
        else:
            self.output_layer = nn.Linear(lstm_output_size, self.embedding_size)
        
        self.loss_type = config['loss_type'] if 'loss_type' in config else 'CE'
        
        self.to(self.device)

    def forward(self, item_seq, item_seq_len):
        """
        Args:
            item_seq: (batch_size, seq_len) long tensor of item IDs
            item_seq_len: (batch_size,) long tensor of actual sequence lengths
        
        Returns:
            seq_output: (batch_size, embedding_size) - sequence representation
        """
        # Embed items
        item_emb = self.item_embedding(item_seq)  # (batch_size, seq_len, embedding_size)
        
        # Pack padded sequences for efficient LSTM processing
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            item_emb, 
            item_seq_len.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # LSTM forward pass
        packed_out, (hidden, cell) = self.lstm(packed_emb)
        
        # Unpack sequences
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # lstm_out: (batch_size, seq_len, hidden_size * num_directions)
        
        # Get last valid hidden state for each sequence
        batch_size = item_seq.size(0)
        last_hidden = lstm_out[range(batch_size), item_seq_len - 1]  # (batch_size, lstm_output_size)
        
        # Project to embedding space
        seq_output = self.output_layer(last_hidden)  # (batch_size, embedding_size)
        
        return seq_output

    def calculate_loss(self, interaction):
        """
        Calculate loss for the model.
        
        Args:
            interaction: RecBole interaction dict
        
        Returns:
            loss: scalar tensor
        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        
        if self.loss_type.upper() == 'CE':
            loss = nn.functional.cross_entropy(logits, pos_items)
        elif self.loss_type.upper() == 'BPR':
            raise NotImplementedError("BPR loss not supported for BiLSTM (use CE)")
        else:
            loss = nn.functional.cross_entropy(logits, pos_items)
        
        return loss

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
