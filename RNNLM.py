# -- coding:utf-8
import torch.nn as nn
from torch.autograd import Variable
import json

class RNNConfig(object):
    def __init__(self, \
            type = 'lstm', \
            embedding_dim = 32, \
            num_layers = 2, \
            bidirectional = False, \
            dp_keep_prob = 0.1, \
            vocab_size = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob
        self.vocab_size = vocab_size
        pass

    def write2json(self, jsonPath):
        with open(jsonPath, 'w') as fOut:
            paraDict = dict()
            paraDict['type'] = self.type
            paraDict['embedding_dim'] = self.embedding_dim
            paraDict['num_layers'] = self.num_layers
            paraDict['bidirectional'] = self.bidirectional
            paraDict['dp_keep_prob'] = self.dp_keep_prob
            paraDict['vocab_size'] = self.vocab_size
            json.dump(paraDict, fOut)
        pass

    def readfrmjson(self, jsonPath):
        with open(jsonPath, 'r') as fIn:
            paraDict = json.load(fIn)
            self.type = paraDict['type']
            self.embedding_dim = paraDict['embedding_dim']
            self.num_layers = paraDict['num_layers']
            self.bidirectional = paraDict['bidirectional']
            self.dp_keep_prob = paraDict['dp_keep_prob']
            self.vocab_size = paraDict['vocab_size']
        pass

class RNNLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(1 - self.config.dp_keep_prob)
        self.word_embeddings = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        if self.config.type == 'lstm':
            self.rnn = nn.LSTM(
                    input_size = self.config.embedding_dim,
                    hidden_size = self.config.embedding_dim,
                    num_layers = self.config.num_layers,
                    bidirectional = self.config.bidirectional,
                    dropout = 1 - self.config.dp_keep_prob)
        elif self.config.type == 'gru':
            self.rnn = nn.GRU(
                    input_size = self.config.embedding_dim,
                    hidden_size = self.config.embedding_dim,
                    num_layers = self.config.num_layers,
                    bidirectional = self.config.bidirectional,
                    dropout = 1 - self.config.dp_keep_prob)
        else:
            print('not support model type, please input lstm or gru')
        if self.config.bidirectional:
            self.sm_fc = nn.Linear(
                    in_features = self.config.embedding_dim * 2,
                    out_features = self.config.vocab_size)
        else:
            self.sm_fc = nn.Linear(
                    in_features = self.config.embedding_dim,
                    out_features = self.config.vocab_size)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.word_embeddings.weight.data.uniform_(-init_range, init_range)
        self.sm_fc.bias.data.fill_(0.0)
        self.sm_fc.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.config.type == 'lstm':
            if self.config.bidirectional:
                return (Variable(weight.new(self.config.num_layers*2, batch_size, self.config.embedding_dim).zero_()),
                        Variable(weight.new(self.config.num_layers*2, batch_size, self.config.embedding_dim).zero_()))
            else:
                return (Variable(weight.new(self.config.num_layers, batch_size, self.config.embedding_dim).zero_()),
                        Variable(weight.new(self.config.num_layers, batch_size, self.config.embedding_dim).zero_()))
        elif self.config.type == 'gru':
            if self.config.bidirectional:
                return Variable(weight.new(self.config.num_layers*2, batch_size, self.config.embedding_dim).zero_())
            else:
                return Variable(weight.new(self.config.num_layers, batch_size, self.config.embedding_dim).zero_())
        else:
            print('not support model type, please input lstm or gru')

    def forward(self, inputs, hidden):
        embeds = self.dropout(self.word_embeddings(inputs))
        rnn_out, hidden = self.rnn(embeds, hidden)
        rnn_out = self.dropout(rnn_out)
        if self.config.bidirectional:
            logits = self.sm_fc(rnn_out.view(-1, self.config.embedding_dim*2))
        else:
            logits = self.sm_fc(rnn_out.view(-1, self.config.embedding_dim))
        return logits.view(inputs.shape[0], inputs.shape[1], self.config.vocab_size), hidden
        #return logits.view(self.num_steps, self.batch_size, self.vocab_size), hidden

def main():
    print('hello')
    myRNNConfig = RNNConfig()
    myRNNConfig.readfrmjson('.//configs//config_lstm_128dim_2L.json')
    print('olleh')

if __name__ == '__main__':
    main()