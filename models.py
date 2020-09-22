
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch as t


class BiLSTMCRF(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, n_classes, embedding_vectors=None, train_embedding=True):
        super(BiLSTMCRF, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        if embedding_vectors is not None:
            self.embedding.weight.data.copy_(t.from_numpy(embedding_vectors))
        self.embedding.weight.requires_grad = train_embedding

        self.lstm = nn.LSTM(embed_size, hidden_size//2,  # num_layers=2, dropout=.2,
                            batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size, n_classes)

        self.crf = mycrf.CRF(n_classes-2, 'cuda')

    def forward(self, x, mask):
        x = self.embedding(x)
        x, (h, c) = self.lstm(x)
        x = self.fc1(x)
        scores, best_tag_sequence = self.crf.forward(x, mask)
        return x, scores, best_tag_sequence

    def loss_fn(self, x, mask, tags):
        return self.crf.neg_log_likelihood_loss(x, mask, tags)


class BiLSTM(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, n_classes, embedding_vectors=None, train_embedding=True):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        if embedding_vectors is not None:
            self.embedding.weight.data.copy_(t.from_numpy(embedding_vectors))
            self.embedding.weight.requires_grad = train_embedding

        self.lstm = nn.LSTM(embed_size, hidden_size//2,  # num_layers=2, dropout=.2,
                            batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size, n_classes)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, mask,  y=None):
        x = self.embedding(x)
        x, (h, c) = self.lstm(x)
        x = self.fc1(x)
        return x

    def loss_fn(self, z, y):
        return self.loss_func(z, y)


class BiLSTMChar(nn.Module):

    def __init__(self, vocab_size,
                 embed_size,
                 hidden_size,
                 n_classes,
                 n_char_class,
                 char_embed_size,
                 max_char_length,
                 embedding_vectors=None,
                 train_embedding=True):
        super(BiLSTMChar, self).__init__()
        self.hidden_size = hidden_size
        self.n_char_class = n_char_class
        self.max_char_length = max_char_length
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.char_embedding = nn.Embedding(n_char_class, char_embed_size)
        nn.init.kaiming_uniform_(
            self.char_embedding.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(
            self.embedding.weight, mode='fan_in', nonlinearity='relu')
        if embedding_vectors is not None:
            self.embedding.weight.data.copy_(t.from_numpy(embedding_vectors))
            self.embedding.weight.requires_grad = train_embedding

        self.lstm = nn.LSTM(hidden_size, hidden_size//2,
                            batch_first=True)
        # nn.init.xavier_normal_(self.lstm)
        # self.cnn = nn.Conv2d(1, 32, 5, 2)
        self.fc1 = nn.Linear(hidden_size//2, n_classes)
        nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(32*14+200, hidden_size)
        nn.init.xavier_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(embed_size*2, hidden_size)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, cs):
        x = self.embedding(x)
        char_word_embs = []
        # print(x.shape)
        for i in range(cs.shape[1]):
            c = cs[:, i]
            c = self.char_embedding(c).view(x.shape[0], -1)
            c = t.cat((c, x[:, i]), dim=1)
            c = F.relu(c)
            c = self.fc2(c)

            # xx = F.relu(x[:, i])

            # c = t.cat((c, xx), dim=1)

            # c = self.fc3(c)
            char_word_embs.append(c)

        x = t.stack(char_word_embs, dim=1)
        x, (h, _) = self.lstm(x)

        x = self.fc1(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)


class BiLSTMChar(nn.Module):

    def __init__(self, vocab_size,
                 embed_size,
                 hidden_size,
                 n_classes,
                 n_char_class,
                 char_embed_size,
                 max_char_length,
                 embedding_vectors=None,
                 train_embedding=True):
        super(BiLSTMChar, self).__init__()
        self.hidden_size = hidden_size
        self.n_char_class = n_char_class
        self.max_char_length = max_char_length
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.char_embedding = nn.Embedding(n_char_class, char_embed_size)
        nn.init.kaiming_uniform_(self.char_embedding.weight)
        nn.init.kaiming_uniform_(self.embedding.weight)
        if embedding_vectors is not None:
            self.embedding.weight.data.copy_(t.from_numpy(embedding_vectors))
            self.embedding.weight.requires_grad = train_embedding

        self.lstm = nn.LSTM(485, hidden_size//2, 
                           bidirectional=True,
                           batch_first=True)

        self.fc1 = nn.Linear(896, hidden_size//2)
        nn.init.kaiming_normal_(self.fc1.weight)

        self.cnn = nn.Conv2d(1, 64, 7, 2)
        nn.init.xavier_normal_(self.cnn.weight)
        self.pool = nn.MaxPool2d(3, stride=2)

        # self.cnn2 = nn.Conv2d(64, 32, 5, 2)
        # nn.init.xavier_normal_(self.cnn.weight)
        # self.pool2 = nn.MaxPool2d(2, stride=2)

        self.fc2 = nn.Linear(512, hidden_size//2)
        nn.init.kaiming_normal_(self.fc2.weight)

        self.fc3 = nn.Linear(512, n_classes)
        nn.init.kaiming_normal_(self.fc3.weight)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, cs, f):
        x = self.embedding(x)
        # x = t.cat((x, f), dim=2)
        char_word_embs = []
        # print(x.shape)
        for i in range(cs.shape[1]):
            c = cs[:, i]

            c = self.char_embedding(c).unsqueeze(1)
            c = self.cnn(c)
            c = self.pool(c)
            c = c.view(x.shape[0], -1)

            c = self.fc1(c)
            # c = F.relu6(c)
            
            # xx = self.fc2(x[:, i])
            # xx = F.relu6(xx)
            c = t.cat((c, x[:, i], f[:, i]), dim=1)
            # c = self.fc2(c)

            char_word_embs.append(c)

        x = t.stack(char_word_embs, dim=1)
        # x = t.cat((x, f), dim=2)
        # print(x.shape)
        x, (h, c) = self.lstm(x)
        # print(x.shape)
        # x = t.cat((x, f), dim=2)
        # print(x.shape)
        # print(x.shape)
        # x = self.fc2(x)
        # x = F.relu6(x)
        x = self.fc3(x)
        
        # x = F.relu(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

# model = BiLSTMChar(20, 200, 256, 6, 13, 32, 7)

# sen_length = 9
# sen = np.random.randint(0, 20, sen_length).reshape((1, sen_length))
# sen_char = np.vstack([np.random.randint(0, 13, 7) for x in range(sen_length)]).reshape((1, sen_length, 7))
# sen = t.tensor(sen).long()
# sen_char = t.tensor(sen_char).long()
# print(sen_char.shape)
# model(sen, sen_char)
