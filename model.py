from hparams import *
from usegpu import *
import torch.nn as nn

'''
pytorch model
'''
class GRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, vocab_size, embedding_dim, n_classes, dropout_p = 0.2):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embedding_dim, self.hidden_dim, num_layers = self.n_layers, batch_first = True)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size = x.size(0)) # 첫 번째 히든 스테이트를 제로벡터로 초기화
        x, _ = self.gru(x, h_0) # GRU의 리턴 값은 (batch size, sequence length, hidden state size)
        h_t = x[:,-1,:] # (batch size, hidden state size) -> (batch size, output layer size)
        self.dropout(h_t)
        logit = self.out(h_t) # (batch size, hidden state size) -> (batch size, output layer size)

        return logit
    
    def _init_state(self, batch_size = 1):
        weight = next(self.parameters()).data
        
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()