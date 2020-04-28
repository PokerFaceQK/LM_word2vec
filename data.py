import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import gensim
from gensim.models import Word2Vec
from gensim.utils import tokenize
from train_word2vec import EpochLogger
import random
import numpy as np
MASKED_TOKEN = -1
PADDING_VALUE = 0


def WordvecId2DataId(idx):
    return idx


def DataId2WordvecId(idx):
    return idx


class Corpus(Dataset):
    def __init__(self, path, **kwargs):
        super(Dataset, self).__init__()
        self.path = path
        self._init_corpus(kwargs)

    def _init_corpus(self, **kwargs):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class WikipediaEn2008(Corpus):
    def _init_corpus(self, kwargs):
        self.embedding = kwargs['model']
        self.set = []
        print_int = 100000
        self.mask_id = []
        with open(self.path) as f:
            for i, line in enumerate(f):
                tokens = line.split()
                if len(tokens) <= 1:
                    continue
                try:  # let's ignore sentences if any of its words is not in the embedding's vocabulary
                    self.set.append(self._preprocess(tokens))
                    self.mask_id.append(random.randint(0, len(tokens)-1))
                except KeyError:
                    pass
                if (i+1) % print_int == 0:
                    print('%d lines have been read' % (i, ))
                if i > 10000000:
                    break

    def _preprocess(self, sentence):
        return [WordvecId2DataId(self.embedding.wv.vocab[word].index) for word in sentence]

    def _should_trim(self, sentence):
        pass

    def __getitem__(self, idx):
        return torch.tensor(self.set[idx]),  self.mask_id[idx]

    def __len__(self):
        return len(self.mask_id)


class NewsCrawlCorpus(Corpus):
    def _init_corpus(self, kwargs):
        self.embedding = kwargs['model']
        self.set = []
        print_int = 100000
        self.mask_id = []
        with open(self.path) as f:
            for i, line in enumerate(f):
                tokens = list(tokenize(line))
                if len(tokens) <= 1:
                    continue
                try:  # let's ignore sentences if any of its words is not in the embedding's vocabulary
                    self.set.append(self._preprocess(tokens))
                    self.mask_id.append(random.randint(0, len(tokens)-1))
                except KeyError:
                    pass
                if (i+1) % print_int == 0:
                    print('%d lines have been read' % (i, ))
                if i > 100000:
                    break

        self.seq_len = 35
        self.set = [i for sentence in self.set for i in sentence]
        self.set = self.set[:(len(self.set) // self.seq_len) * self.seq_len]
        self.set = torch.tensor(self.set)

    def _preprocess(self, sentence):
        return [WordvecId2DataId(self.embedding.wv.vocab[word.lower()].index) for word in sentence]

    def _should_trim(self, sentence):
        pass

    def __getitem__(self, idx):
        return self.set[idx*self.seq_len:self.seq_len*(idx+1)], 0

    def __len__(self):
        return self.set.size(0) // self.seq_len


def collate(list_of_samples):
    """Merges a list of samples to form a mini-batch.

    Args:
      list_of_samples is a list of tuples (src_seq, tgt_word):
          src_seq is of shape (src_seq_length,)
          tgt_seq is an integer
    """
    max_len_src = 0
    for src_seq, tgt_word in list_of_samples:
        if max_len_src < len(src_seq):
            max_len_src = len(src_seq)

    order = np.argsort([len(pair[0]) for pair in list_of_samples])[::-1]
    src_seqs = torch.zeros(max_len_src, len(list_of_samples), dtype=torch.long).fill_(PADDING_VALUE)
    tgt_seqs = torch.zeros(1, len(list_of_samples), dtype=torch.long)
    src_seq_len = torch.LongTensor(len(list_of_samples))

    for i, idx in enumerate(order):
        source, mask_pos = list_of_samples[idx]
        target = source[mask_pos]
        # source[mask_pos] = MASKED_TOKEN
        len_s = len(source)
        src_seqs[:len_s, i] = source
        # src_seqs[mask_pos, i] = MASKED_TOKEN
        tgt_seqs[0, i] = target
        src_seq_len[i] = len_s

    return src_seqs, src_seq_len, tgt_seqs


if __name__ == '__main__':
    w2v = Word2Vec.load('models/word2vec_100_False_False_5.model')
    corp = WikipediaEn2008('data/wikipedia2008_en.txt', model=w2v)
    trainloader = DataLoader(dataset=corp, batch_size=64, shuffle=True, num_workers=8, collate_fn=collate)
    for i, data in enumerate(trainloader):
        source, seq_len, target = data
        size = source.size()
        source_np = source.data.numpy()
        pass