from torchtext.data import Field, Iterator, TabularDataset
from torchtext import data

from hparams import *
from usegpu import *
from model import GRU
from setting import *

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
필드객체 생성
'''
TEXT = Field(sequential=True,
             use_vocab=True,
             tokenize=str.split,
             lower=True, 
             batch_first=True)  
LABEL = Field(sequential=False,  
              use_vocab=False,   
              #preprocessing = lambda x: int(x),
              batch_first=True)

'''
데이터셋 생성
'''
document = data.Field()
label = data.Field()
#fields = {'document': ('text', document), 'label': ('label', label)}
train_dataset, validation_dataset, test_dataset = data.TabularDataset.splits(
    path = './data/',
    train = 'train_t.txt',
    validation = 'validation_t.txt',
    test = 'test_t.txt',
    format = 'csv',
    fields = [('text', TEXT), ('label', LABEL)]
)

#print(test_dataset.fields)
#print(vars(train_dataset[0]))
#print(vars(train_dataset[0]))
'''
단어집합 생성
'''
TEXT.build_vocab(train_dataset, min_freq = 3)
LABEL.build_vocab(train_dataset)

#print(TEXT.vocab.stoi)
#print(LABEL.vocab.stoi)
vocab_size = len(TEXT.vocab)

'''
데이터로더 생성
'''
device = UseGPU()
train_loader, validation_loader = data.BucketIterator.splits(
    (train_dataset, validation_dataset),
    batch_size = batch_size,
    device = device,
    repeat = False,
    sort = False)

#print(len(train_loader))
#print(len(validation_loader))

'''
모델, 비용함수, 옵티마이저 정의
'''
model = GRU(n_layers, hidden_dim, vocab_size, embedding_dim, n_classes, 0.5).to(device)
#print(model.parameters)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

'''
학습
'''
best_val_loss = None
for epoch in range(nb_epochs):
    train(model, criterion, optimizer, train_loader)
    validation_loss, validation_accuracy = evaluate(model, criterion, validation_loader)

    print("Epoch: {} | Val Loss: {} | Val Accuracy: {}".format(epoch + 1, validation_loss, validation_accuracy))

    if not best_val_loss or validation_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = validation_loss