from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.utils import tokenize
from multiprocessing import cpu_count
import argparse
import sys
import os


class SentencesIter(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        f = open(self.filename)
        for line in f:
            yield self._preprocess(line)

    def _preprocess(self, sentence):
        tokens = list(tokenize(sentence))
        return [word.lower() for word in tokens]


class NewsCrawl(SentencesIter):
    def _preprocess(self, sentence):
        tokens = list(tokenize(sentence))
        return [word.lower() for word in tokens]


class WikiText(SentencesIter):
    def __iter__(self):
        f = open(self.filename)
        for line in f:
            res = self._preprocess(line)
            if len(res) == 0 or res[1] == '=':
                continue
            yield res


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, default='models',
                        help="save the model in this folder")
    parser.add_argument("--size", '-s', type=int, default=200,
                        help="the dimensionality of hidden representations")
    parser.add_argument("--winsize", '-ws', type=int, default=10,
                        help="window size")
    parser.add_argument("--skipgram", '-sg', type=bool, default=False,
                        help="use skipgram if true, otherwise use cbow")
    parser.add_argument("--corpus", '-cp', type=str, default='wikitext.txt',
                        help="path to the training corpus")
    parser.add_argument("--epochs", "-e", type=int, default=5,
                        help="epochs to train")
    parser.add_argument("--min_count", type=int, default=10,
                        help="the threshold to ignore words. Any word of count(words) < min_count should be ignored")
    parser.add_argument("--hs", type=bool, default=False,
                        help='if use hierarchical softmax')
    parser.add_argument("--negative", "-n", type=int, default=5)
    parser.add_argument("--lr_s", type=float, default=0.025,
                        help='learning rate to start with')
    parser.add_argument("--lr_e", type=float, default=0.001,
                        help='the final learning rate')
    parser.add_argument("--resume", '-r', type=bool, default=False,
                        help="resume training if true, otherwise train a new model")
    parser.add_argument("--loadname", type=str, default='word2vec.model')
    parser.add_argument("--loadepoch", type=str, default=0)

    return parser.parse_args(args)
    

def main(args):
    epoch_logger = EpochLogger()
    path2corpus = os.path.join('data', args.corpus)
    sentences = NewsCrawl(path2corpus)

    if args.resume:
        loadname = os.path.join(args.dir, args.loadname)
        model = Word2Vec.load(loadname)
    else:
        model = Word2Vec(workers=cpu_count(),
                         size=args.size,
                         window=args.winsize,
                         sg=args.skipgram,
                         hs=args.hs,
                         negative=args.negative,
                         alpha=args.lr_s,
                         min_alpha=args.lr_e,
                         min_count=args.min_count,
                         compute_loss=True,
                         )
        model.build_vocab(sentences, progress_per=10000)
        # f_no_training = open(os.path.join(args.dir, 'word2vec_notrain.model'), 'wb')
        # model.save(f_no_training)

    for i in range(args.epochs):
        model.train(sentences, total_examples=model.corpus_count, epochs=1, compute_loss=True)
        loss = model.get_latest_training_loss()
        print('[epoch %d] loss: %.5f' % (i+args.loadepoch, loss))
    savename = "wikitext_{size}_{winsize}_{sg}_{hs}_{epoch}.model".format(size=args.size,
                                                                          winsize=args.winsize,
                                                                          sg=args.skipgram,
                                                                          hs=args.hs,
                                                                          epoch=args.epochs+args.loadepoch)
    f_model = open(os.path.join(args.dir, savename), 'wb')
    model.save(f_model)
    f_model.close()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
