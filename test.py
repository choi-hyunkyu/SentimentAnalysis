from setting import *
from hparams import *
from usegpu import *
from model import GRU
from torchtext.data import Field, Iterator, TabularDataset
from torchtext import data
import torch.nn as nn

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
test_dataset = data.TabularDataset(
    path = './data/test_t.txt',
    format = 'csv',
    fields = [('text', TEXT), ('label', LABEL)]
)

#print(test_dataset.fields)
#print(vars(train_dataset[0]))
#print(vars(train_dataset[0]))
'''
단어집합 생성
'''
TEXT.build_vocab(test_dataset, min_freq = 3)
LABEL.build_vocab(test_dataset)

#print(TEXT.vocab.stoi)
#print(LABEL.vocab.stoi)
vocab_size = len(TEXT.vocab)

'''
데이터로더 생성
'''
device = UseGPU()
test_loader = data.BucketIterator(
    test_dataset,
    batch_size = batch_size,
    device = device,
    repeat = False,
    sort = False
    )

model = GRU(n_layers, hidden_dim, vocab_size, embedding_dim, n_classes, 0.5).to(device)
criterion = nn.CrossEntropyLoss().to(device)
test_loss, test_accuracy = evaluate(model, criterion, test_loader)
print('Test Loss: {} | Test Accuracy: {}'.format(test_loss, test_accuracy))