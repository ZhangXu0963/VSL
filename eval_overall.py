"""Evaluation"""

from __future__ import print_function
import os
from re import T
import sys
import time

import torch
import torch.nn as nn
import numpy as np

from data import get_test_loader
from vocab import Vocabulary, deserialize_vocab
from model import SGRAF
from collections import OrderedDict
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    max_n_word = 0
    for i, (images, captions, lengths, ids, sem_for_caption, sems_for_image) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    for i, (images, captions, lengths, ids, sem_for_caption, sems_for_image) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        with torch.no_grad():
            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths)
            # img_cider = cider_for_image(sems_for_image)
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, :max(lengths), :] = cap_emb.data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]

        del images, captions
    return img_embs, cap_embs, cap_lens


def evalrank(sgr_model_path, saf_model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    sgr_checkpoint = torch.load(sgr_model_path)
    sgr_opt = sgr_checkpoint['opt']
    sgr_save_epoch = sgr_checkpoint['epoch']
    print(sgr_opt)

    saf_checkpoint = torch.load(saf_model_path)
    saf_opt = saf_checkpoint['opt']
    saf_save_epoch = saf_checkpoint['epoch']
    print(saf_opt)

    if data_path is not None:
        sgr_opt.data_path = data_path
        saf_opt.data_path = data_path

    # load vocabulary used by the model
    vocab = deserialize_vocab('./vocab/%s_vocab.json' % sgr_opt.data_name)
    sgr_opt.vocab_size = len(vocab)
    vocab = deserialize_vocab('./vocab/%s_vocab.json' % saf_opt.data_name)
    saf_opt.vocab_size = len(vocab)

    # construct model
    sgr_model = SGRAF(sgr_opt)
    saf_model = SGRAF(saf_opt)

    # load model state
    sgr_model.load_state_dict(sgr_checkpoint['model'])
    saf_model.load_state_dict(sgr_checkpoint['model'])

    print('Loading dataset')
    sgr_data_loader = get_test_loader(split, sgr_opt.data_name, vocab,
                                  sgr_opt.batch_size, sgr_opt.workers, sgr_opt)
    saf_data_loader = get_test_loader(split, saf_opt.data_name, vocab,
                                  saf_opt.batch_size, saf_opt.workers, saf_opt)
    print("=> loaded SGR checkpoint_epoch {}".format(sgr_save_epoch))
    print("=> loaded SAF checkpoint_epoch {}".format(saf_save_epoch))

    print('Computing results...')
    sgr_img_embs, sgr_cap_embs, sgr_cap_lens = encode_data(sgr_model, sgr_data_loader)
    saf_img_embs, saf_cap_embs, saf_cap_lens = encode_data(saf_model, saf_data_loader)
    img_embs = (sgr_img_embs + saf_img_embs)/2
    cap_embs = (sgr_cap_embs + saf_cap_embs)/2

    print('SGR: Images: %d, Captions: %d' %(sgr_img_embs.shape[0] / 5, sgr_cap_embs.shape[0]))
    print('SAF: Images: %d, Captions: %d' %(saf_img_embs.shape[0] / 5, saf_cap_embs.shape[0]))

    if not fold5:#False
        # no cross-validation, full evaluation
        sgr_img_embs = np.array([sgr_img_embs[i] for i in range(0, len(sgr_img_embs), 5)])
        saf_img_embs = np.array([saf_img_embs[i] for i in range(0, len(saf_img_embs), 5)])

        # record computation time of validation
        start = time.time()
        sgr_sims = shard_attn_scores(sgr_model, sgr_img_embs, sgr_cap_embs, sgr_cap_lens, sgr_opt, shard_size=100)
        saf_sims = shard_attn_scores(saf_model, saf_img_embs, saf_cap_embs, saf_cap_lens, saf_opt, shard_size=100)
        
        sims = (saf_sims + sgr_sims)/2

        end = time.time()
        print("calculate similarity time:", end-start)

        # bi-directional retrieval
        r, rt = i2t(img_embs, cap_embs, sgr_cap_lens, sims, return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs, sgr_cap_lens, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []

        for i in range(5):
            sgr_img_embs_shard = sgr_img_embs[i * 5000:(i + 1) * 5000:5]
            sgr_cap_embs_shard = sgr_cap_embs[i * 5000:(i + 1) * 5000]
            sgr_cap_lens_shard = sgr_cap_lens[i * 5000:(i + 1) * 5000]

            saf_img_embs_shard = saf_img_embs[i * 5000:(i + 1) * 5000:5]
            saf_cap_embs_shard = saf_cap_embs[i * 5000:(i + 1) * 5000]
            saf_cap_lens_shard = saf_cap_lens[i * 5000:(i + 1) * 5000]

            img_embs_shard = (sgr_img_embs_shard + saf_img_embs_shard)/2
            cap_embs_shard = (sgr_cap_embs_shard + saf_cap_embs_shard)/2

            start = time.time()
            
            sgr_sims = shard_attn_scores(sgr_model, sgr_img_embs_shard, sgr_cap_embs_shard, sgr_cap_lens_shard, sgr_opt, shard_size=100)
            saf_sims = shard_attn_scores(saf_model, saf_img_embs_shard, saf_cap_embs_shard, saf_cap_lens_shard, saf_opt, shard_size=100)

            sims = (sgr_sims + saf_sims)/2
            
            end = time.time()
            print("calculate similarity time:", end-start)

            r, rt0 = i2t(img_embs_shard, cap_embs_shard, sgr_cap_lens_shard, sims, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs_shard, cap_embs_shard, sgr_cap_lens_shard, sims, return_ranks=True)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])


def shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=100):
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

    sims = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                l = cap_lens[ca_start:ca_end]
                sim = model.forward_sim(im, ca, l)

            sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return sims


def i2t(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
        

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            # index_all.append(inds)
            tmp = np.where(inds == index)[0][0]
            ranks[5 * index + i] = tmp
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


if __name__ == '__main__':
    evalrank(sgr_model_path="./checkpoint/SGR+VSL_COCO.pth.tar",
             saf_model_path="./checkpoint/SAF+VSL_COCO.pth.tar",
             data_path='./data', split="testall", fold5=True) #mscoco1k: fold5=True, else: fold5=False
