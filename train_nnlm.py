import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from gensim.models import Word2Vec
from data import WikipediaEn2008, NewsCrawlCorpus, PADDING_VALUE, MASKED_TOKEN, collate
import time
from model import RnnLanguageModel
from numpy import floor
import numpy as np
from math import exp
# this import is necessary. It's implicitly used for loading the pre-trained word2vec model
from train_word2vec import EpochLogger
from torch.utils.data.sampler import SubsetRandomSampler

LR = [20.] * 3 + [2.]*15 + [0.2] * 5 + [0.02] * 5
bs = 20

def loss_fun(pred, target, seq_len):
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    loss_total = 0.0
    for id_batch in range(pred.size(1)):
        length = seq_len[id_batch]
        loss = criterion(pred[:length-1, id_batch, :], target[:length-1, id_batch])
        loss_total = loss_total + loss.sum()
    with torch.no_grad():
        valid_num = (seq_len.sum() - seq_len.size(-1))
    return loss_total / valid_num


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def train(model: torch.nn.modules, trainloader: DataLoader, validloader: DataLoader, n_epochs: int, w2v: Word2Vec, device: torch.device):
    optimizer = optim.SGD(model.parameters(), lr=20, momentum=0., weight_decay=0.)
    model.train()
    for epoch in range(1, 1+n_epochs):
        loss_tmp = 0.0
        print_int = 100
        start_time = time.time()
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR[epoch-1]
        model.train()
        # bs = 20
        hidden = model.encoder.init_hidden(batch_size=bs)
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()
            source, source_len, target = data

            if source.size(1) != bs:
                continue
            source = source.to(device)
            source_len = source_len.to(device)

            hidden = repackage_hidden(hidden)
            output, hidden = model(source, source_len, hidden)
            # use the shifted source sentences as target (the model should predict s[t] based on s[:t-1])
            with torch.no_grad():
                target = source[1:, :].clone().detach()
            loss_fun = torch.nn.NLLLoss()
            # loss = loss_fun(output, target, seq_len=source_len)
            loss = loss_fun(output[:-1].view(-1, output.size(-1)), target.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()

            loss_tmp += loss.item()
            if (i + 1) % print_int == 0:
                print('[epoch %d] [iteration %d] loss: %.5f  time: %.2f' %
                      (epoch, i+1, loss_tmp / print_int, time.time() - start_time))
                start_time = time.time()
                loss_tmp = 0



        # adjust learning rate

        # compute loss on validation set
        valid_count = 0
        loss_tmp = 0
        model.eval()
        for i, data in enumerate(validloader):
            with torch.no_grad():
                optimizer.zero_grad()
                source, source_len, target = data

                source = source.to(device)
                source_len = source_len.to(device)
                output, _ = model(source, source_len)
                # use the shifted source sentences as target (the model should predict s[t] based on s[:t-1])
                target = source[1:, :].clone().detach()
                loss = loss_fun(output[:-1].view(-1, output.size(-1)), target.view(-1))
                loss_tmp += loss.item()
                valid_count += 1
        print('loss on validation set: %.5f, ppl: %.5f' % (loss_tmp / valid_count, exp(loss_tmp / valid_count)))

    savename = 'lstm_{epoch}.pth'.format(epoch=n_epochs)
    torch.save(model.state_dict(), savename)
    print('model %s saved' % (savename, ))
    pass



if __name__ == '__main__':
    use_gpu = True
    if torch.cuda.is_available() and use_gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    w2v = Word2Vec.load('models/wikitext_200_10_False_False_10.model')
    embeddings = torch.from_numpy(w2v.wv.vectors)
    model = RnnLanguageModel(len(w2v.wv.index2word),
                             embed_size=w2v.wv.vector_size,
                             hidden_size=200,
                             embeddings=embeddings
                             ).to(device)
    corpus = NewsCrawlCorpus('data/wikitext.txt', model=w2v)
    corpus_valid = NewsCrawlCorpus('data/wikitext_valid.txt', model=w2v)
    corpus_test = NewsCrawlCorpus('data/wikitext_test.txt', model=w2v)

    indices = list(range(len(corpus)))
    shuffle_dataset = False
    ratio = 1
    split = int(floor(ratio * len(corpus)))
    random_seed = 42

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    trainloader = DataLoader(dataset=corpus, batch_size=20, collate_fn=collate, shuffle=True)
    validloader = DataLoader(dataset=corpus_test, batch_size=bs, collate_fn=collate, shuffle=False)
    train(model, trainloader, validloader, 10, w2v, device)
