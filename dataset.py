import csv

import numpy as np
import spacy
import torch
from torch.utils.data import Dataset


class SentenceDataset(Dataset):
    def __init__(self, sentences, labels, seq_len, vocab):
        self.sentences = sentences
        self.labels = labels
        self.seq_len = seq_len
        self.vocab = vocab
        self.nlp = spacy.load('en_core_web_sm')

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def load_vocab(path):
        vocab, freq = [], []
        with open(path, 'r') as file:
            csvreader = csv.reader(file)
            for word, f in csvreader:
                vocab.append(word)
                freq.append(f)
        return vocab, freq

    def spacy_tokenize(self, sentence):
        return [token.text for token in self.nlp(sentence)]

    def digitalize_sentence(self, sntc):
        result = [1]
        # 0 for unk, 1 for start, 2 for end
        token_lst = self.spacy_tokenize(str.lower(sntc))
        for token in token_lst:
            if token in self.vocab:
                result.append(self.vocab.index(token) + 3)
            else:
                result.append(0)
        result.append(2)
        return result

    def one_hot_emb(self, word_indxes, size):
        embedings = []
        for idx in word_indxes:
            word_embd = [0] * size
            word_embd[idx] = 1
            embedings.append(word_embd)
        return np.array(embedings)

    def parse_sentnence(self, digital_sntc):
        result = []
        for index in digital_sntc:
            if index == 0:
                result.append("<UNK>")
            elif index == 1:
                result.append("<s>")
            elif index == 2:
                result.append("<\s>")
            else:
                result.append(self.vocab[index - 3])
        return " ".join(result)

    def __getitem__(self, idx):
        word_indxes = self.digitalize_sentence(self.sentences[idx])
        padding_size = 0
        if len(word_indxes) > self.seq_len:
            word_indxes = word_indxes[:self.seq_len]
        if len(word_indxes) < self.seq_len:
            padding_size = self.seq_len - len(word_indxes)
            word_indxes = word_indxes + [0] * padding_size

        return torch.tensor(word_indxes), padding_size, self.labels[idx]
