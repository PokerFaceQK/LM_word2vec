import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

freeze = False

class Encoder(nn.Module):
    def __init__(self, src_dictionary_size, embed_size, hidden_size, nlayers=2, rnn_type='LSTM', dropout=0.2, embeddings=None):
        """
        Args:
          src_dictionary_size: The number of words in the source dictionary.
          embed_size: The number of dimensions in the word embeddings.
          hidden_size: The number of features in the hidden state of GRU.
        """
        super(Encoder, self).__init__()
        # self.embedding = nn.Embedding(src_dictionary_size, embed_size)
        if not embeddings == None:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=freeze)
            self.from_word2vec = True
        else:
            self.embedding = nn.Embedding(src_dictionary_size, embed_size)
            if freeze:
                for p in self.embedding.parameters():
                    p.requires_grad = False
            self.from_word2vec = False
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=nlayers, dropout=dropout)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=nlayers, dropout=dropout)
        self.vocab_size = src_dictionary_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, seq_lengths, hidden):
        """
        Args:
          pad_seqs of shape (max_seq_length, batch_size): Padded source sequences.
          seq_lengths: List of sequence lengths.
          hidden of shape (1, batch_size, hidden_size): Initial states of the GRU/LSTM.

        Returns:
          outputs of shape (max_seq_length, batch_size, hidden_size): Padded outputs of GRU at every step.
          hidden of shape (1, batch_size, hidden_size): Updated states of the GRU/LSTM.
        """
        # YOUR CODE HERE
        embedded = self.embedding(x)
        # embedded = x
        embedded = self.dropout(embedded)

        outputs = pack_padded_sequence(embedded, seq_lengths, batch_first=False)
        outputs, hidden = self.rnn(outputs, hidden)
        outputs, _ = pad_packed_sequence(outputs)
        return outputs, hidden

    def init_hidden(self, batch_size=1):
        if self.rnn_type == 'GRU':
            return torch.zeros(self.nlayers, batch_size, self.hidden_size)
        elif self.rnn_type == 'LSTM':
            return (torch.zeros(self.nlayers, batch_size, self.hidden_size),
                    torch.zeros(self.nlayers, batch_size, self.hidden_size))


class RnnLanguageModel(nn.Module):
    def __init__(self, src_dictionary_size, embed_size, hidden_size, dropout=0.2, embeddings=None):
        super(RnnLanguageModel, self).__init__()
        self.vocab_size = src_dictionary_size
        self.encoder = Encoder(src_dictionary_size, embed_size, hidden_size, dropout=dropout, embeddings=embeddings)
        self.decoder = nn.Linear(hidden_size, src_dictionary_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=2)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if not self.encoder.from_word2vec:
            self.encoder.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, seq_lengths, hidden=None):
        batch_size = x.size(1)
        if hidden == None:
            encoder_hidden = self.encoder.init_hidden(batch_size=batch_size)
        else:
            encoder_hidden = hidden
        if self.encoder.rnn_type == 'GRU':
            encoder_hidden = encoder_hidden.to(x.device)
        elif self.encoder.rnn_type == 'LSTM':
            encoder_hidden = (encoder_hidden[0].to(x.device), encoder_hidden[1].to(x.device))
        out, hidden = self.encoder(x, seq_lengths, encoder_hidden)
        out = self.dropout(out)
        out = self.decoder(out)
        prob = self.softmax(out)
        return prob, hidden

