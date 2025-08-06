import torch
import torch.nn as nn
import pandas as pd 
import numpy as np
from collections import Counter


def data_initialize():
    data = pd.read_csv('IMDB Dataset.csv')

    word_counts = Counter()
    word_to_num={}
    tokenized_reviews=[]

    for review in data['review']:
        review = review.lower()
        review = ''.join(char for char in review if char.isalpha() or char.isspace() )
        words = review.split()
        word_counts.update(words)



    most_common= word_counts.most_common(5000)


    for i, (word,_) in  enumerate(most_common):
            word_to_num[word]=i +1

    word_to_num['<UNK>'] = 0


    for review in data['review']:
        temp=[]
        review = review.lower()
        review = ''.join(char for char in review if char.isalpha() or char.isspace() )
        review = review.split()
        for word in review:
            temp.append(word_to_num[word])

        tokenized_reviews.append(temp)


    vocab_size = len(word_to_num)

    return vocab_size, tokenized_reviews






class RNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,100)
        self.lstm = nn.LSTM(100,128)
        self.linear = nn.Linear(128,1)
        pass
    def forward(self):
        self.lstm
        self.linear

    def calculate_error(self):
        pass


def training():
    pass
def testing():
    pass