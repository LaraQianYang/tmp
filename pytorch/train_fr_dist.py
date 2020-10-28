# coding: utf-8
import argparse
import time
import math
import os, sys
import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import get_lm_corpus
from mem_transformer_fr import MemTransformerLM_fr
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel

from torch.autograd import Variable

from sharedtensor import SharedTensor
import multiprocessing as mp

GPUS = [0, 1, 2, 3]
torch.backends.cudnn.deterministic = True

def main(args, logging):

    device = torch.device('cuda' if args.cuda else 'cpu')

    ###############################################################################
    # Load data
    ###############################################################################
    corpus = get_lm_corpus(args.data, args.dataset)
    ntokens = len(corpus.vocab)
    args.n_token = ntokens

    eval_batch_size =  10
    tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
        device=device, ext_len=args.ext_len)
    va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len,
        device=device, ext_len=args.ext_len)
    te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len,
        device=device, ext_len=args.ext_len)

    # adaptive softmax / embedding
    cutoffs, tie_projs = [], [False]
    if args.adaptive:
        assert args.dataset in ['wt103', 'lm1b']
        if args.dataset == 'wt103':
            cutoffs = [20000, 40000, 200000]
            tie_projs += [True] * len(cutoffs)
        elif args.dataset == 'lm1b':
            cutoffs = [60000, 100000, 640000]
            tie_projs += [False] * len(cutoffs)

    ###############################################################################
    # Build the model
    ###############################################################################
    def init_weight(weight):
        if args.init == 'uniform':
            nn.init.uniform_(weight, -args.init_range, args.init_range)
        elif args.init == 'normal':
            nn.init.normal_(weight, 0.0, args.init_std)

    def init_bias(bias):
        nn.init.constant_(bias, 0.0)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('AdaptiveEmbedding') != -1:
            if hasattr(m, 'emb_projs'):
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                init_weight(m.weight)
        elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
            if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
                init_weight(m.cluster_weight)
            if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
                init_bias(m.cluster_bias)
            if hasattr(m, 'out_projs'):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, args.init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('TransformerLM') != -1 or classname.find('TransformerLM_fr_begin_end') != -1 or classname.find('TransformerLM_fr_mid') != -1:
            if hasattr(m, 'r_emb'):
                init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                init_bias(m.r_bias)

    def update_dropout(m):
        classname = m.__class__.__name__
        if classname.find('Dropout') != -1:
            if hasattr(m, 'p'):
                m.p = args.dropout

    def update_dropatt(m):
        if hasattr(m, 'dropatt'):
            m.dropatt.p = args.dropatt

    models = MemTransformerLM_fr(args.num_splits, ntokens, args.n_layer, args.n_head, args.d_model,
            args.d_head, args.d_inner, args.dropout, args.dropatt,
            tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
            tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
            ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
            same_length=args.same_length, attn_type=args.attn_type,
            clamp_len=args.clamp_len, sample_softmax=args.sample_softmax)

    for i, model in enumerate(models):
        model.apply(weights_init)
        if i == 0:
            model.word_emb.apply(weights_init) # ensure embedding init is not overridden by out_layer in case of weight sharing

    args.n_all_param = 0
    args.n_nonemb_param = 0

    for i in range(len(models)):
        args.n_all_param += sum([p.nelement() for p in models[i].parameters()])
        if i == 0:
            args.n_nonemb_param += sum([p.nelement() for p in models[i].layers_begin.parameters()])
            args.n_nonemb_param += sum([p.nelement() for p in models[i].layers_end.parameters()])
        else:
            args.n_nonemb_param += sum([p.nelement() for p in models[i].layers.parameters()])


    #### optimizer
    if args.optim.lower() == 'sgd':
        if args.sample_softmax > 0:
            assert args.sample_softmax>0, 'not implemented yet!'
        else:
            optimizers = []
            for model in models:
                optimizer = optim.SGD(model.parameters(), lr=args.lr,
                    momentum=args.mom)
                optimizers.append(optimizer)
    if args.optim.lower() == 'adam':
        if args.sample_softmax > 0:
            assert args.sample_softmax>0, 'not implemented yet!'
        else:
            optimizers = []
            for model in models:
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                optimizers.append(optimizer)

    #### scheduler
    if args.scheduler == 'cosine':
        # here we do not set eta_min to lr_min to be backward compatible
        # because in previous versions eta_min is default to 0
        # rather than the default value of lr_min 1e-6
        schedulers = []
        for optimizer in optimizers:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                args.max_step, eta_min=args.eta_min) # should use eta_min arg
            schedulers.append(scheduler)
        if args.sample_softmax > 0:
            scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(optimizer_sparse,
                args.max_step, eta_min=args.eta_min) # should use eta_min arg
    elif args.scheduler == 'inv_sqrt':
        # originally used for Transformer (in Attention is all you need)
        def lr_lambda(step):
            # return a multiplier instead of a learning rate
            if step == 0 and args.warmup_step == 0:
                return 1.
            else:
                return 1. / (step ** 0.5) if step > args.warmup_step \
                       else step / (args.warmup_step ** 1.5)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif args.scheduler == 'dev_perf':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
        if args.sample_softmax > 0:
            scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sparse,
                factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
    elif args.scheduler == 'constant':
        pass


    logging('=' * 100)
    for k, v in args.__dict__.items():
        logging('    - {} : {}'.format(k, v))
    logging('=' * 100)
    logging('#params = {}'.format(args.n_all_param))
    logging('#non emb params = {}'.format(args.n_nonemb_param))

    ###############################################################################
    # Training code
    ###############################################################################

    shm_lists = []
    shape = [args.tgt_len, args.batch_size, args.d_model]
    for i in range(args.num_splits):
        shm_data = SharedTensor(shape)
        shm_grad = SharedTensor(shape)
        shm_lists.append(shm_data)
        shm_lists.append(shm_grad)

    # eval shm_lists
    shm_lists_eval = []
    shape = [args.eval_tgt_len, eval_batch_size, args.d_model]
    for i in range(args.num_splits):
        shm_data = SharedTensor(shape)
        shm_grad = SharedTensor(shape)
        shm_lists_eval.append(shm_data)
        shm_lists_eval.append(shm_grad)


    # Loop over epochs.
    processes = []
    for i in range(args.num_splits):
        if i == 0:
            p = mp.Process(target=train_begin_end, args=(models[i], optimizers[i], tr_iter, va_iter, shm_lists, shm_lists_eval, i, schedulers[i], logging, args))
        else:
            p = mp.Process(target=train_mid, args=(models[i], optimizers[i], tr_iter.n_batch, va_iter.n_batch, shm_lists, shm_lists_eval,  i, schedulers[i], args))

        p.start()
        processes.append(p)

    for p in processes:
        p.join()



def train_mid(model, optimizer, n_batch, n_batch_val, shm_lists, shm_lists_eval, split_id, scheduler, args):
    model.cuda(GPUS[split_id])
    train_step = 0
    # Turn on training mode which enables dropout.
    for epoch in itertools.count(start=1):

        model.train()
    
        mems = tuple()

        #for batch, (data, target, seq_len) in enumerate(train_iter):
        for batch in range(n_batch):
            model.zero_grad()

            # forward
            hidden = shm_lists[2*(split_id-1)].recv()
            #print(split_id, ' get hidden from ', 2*(split_id-1) , hidden.norm())
            if args.cuda:
                hidden = hidden.cuda(GPUS[split_id])

            hidden = Variable(hidden.data)
            with torch.no_grad():
                ret = model.forward(hidden, *mems)
            hidden, mems = ret[0], ret[1:]

            shm_lists[2*split_id].send(hidden.data.cpu())
            #print(split_id, ' sent hidden to ', 2*(split_id) , hidden.norm())

            # backward  
            model.fr_backward()

            if model.delay < 0:
                grad = model.get_grad()
                shm_lists[2*(split_id-1)+1].send(grad.cpu())
                #print(split_id, ' sent grad to ', 2*(split_id-1)+1 , grad.norm())

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()

            # receive and save grad 
            if model.delay <= 0:
                grad = shm_lists[2*split_id+1].recv()
                if args.cuda:
                    grad = grad.cuda(GPUS[split_id])
                model.save_grad(grad) 
                #print(split_id, ' receive grad from ', 2*(split_id)+1 , grad.norm())

            # step-wise learning rate annealing
            train_step += 1
            if args.scheduler in ['cosine', 'constant', 'dev_perf']:
                # linear warmup stage
                if train_step < args.warmup_step:
                    curr_lr = args.lr * train_step / args.warmup_step
                    optimizer.param_groups[0]['lr'] = curr_lr
                else:
                    if args.scheduler == 'cosine':
                        scheduler.step(train_step)
            elif args.scheduler == 'inv_sqrt':
                scheduler.step(train_step)


            if train_step % args.eval_interval == 0:
                model.eval()
                # If the model does not use memory at all, make the ext_len longer.
                # Otherwise, make the mem_len longer and keep the ext_len the same.
                if args.mem_len == 0:
                    model.reset_length(args.eval_tgt_len,
                        args.ext_len+args.tgt_len-args.eval_tgt_len, args.mem_len)
                else:
                    model.reset_length(args.eval_tgt_len,
                        args.ext_len, args.mem_len+args.tgt_len-args.eval_tgt_len)

                mems_eval = tuple()
                with torch.no_grad():
                    for batch in range(n_batch_val):
                        # forward
                        hidden = shm_lists_eval[2*(split_id-1)].recv()
                        if args.cuda:
                            hidden = hidden.cuda(GPUS[split_id])

                        hidden = Variable(hidden.data)
                        ret = model.forward(hidden, *mems_eval)
                        hidden, mems_eval = ret[0], ret[1:]

                        shm_lists_eval[2*split_id].send(hidden.data.cpu())

                # Switch back to the training mode
                model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
                model.train()


            if train_step == args.max_step:
                break
        if train_step == args.max_step:
            break


def train_begin_end(model, optimizer, tr_iter, va_iter, shm_lists, shm_lists_eval, split_id, scheduler, logging, args):
    model.cuda(GPUS[0])
    # Turn on training mode which enables dropout.
    log_start_time = time.time()
    eval_start_time = time.time()
    train_step = 0
    train_loss = 0
    best_val_loss = None
    val_loss = 0

    for epoch in itertools.count(start=1):

        model.train()
    
        mems_begin = tuple()
        mems_end = tuple()

        train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter
        for batch, (data, target, seq_len) in enumerate(train_iter):
            if seq_len > args.tgt_len:
                data = data[0:args.tgt_len]
                target = target[0:args.tgt_len]
            else:
                while len(data) != args.tgt_len:
                    data_copy_len = (args.tgt_len - len(data)) if (args.tgt_len - len(data) <= len(data)) else len(data)
                    data = torch.cat([data, data[0:data_copy_len]], 0)
                    target = torch.cat([target, target[0:data_copy_len]], 0)

            model.zero_grad()

            # forward begin
            with torch.no_grad():
                ret = model.forward_begin(data, *mems_begin)
            hidden, mems_begin = ret[0], ret[1:]
            shm_lists[split_id*2].send(hidden.data.cpu())
            #print(0, ' sent hidden to ', 2*(split_id) , hidden.norm())


            # forward end
            hidden_end = shm_lists[2*(args.num_splits-1)].recv()
            if args.cuda:
                hidden = hidden_end.cuda(GPUS[0])
            #print(args.num_splits, ' receive hidden from ', 2*(args.num_splits-1) , hidden.norm())

            # backward  end 
            model.fr_backward_begin()

            hidden = Variable(hidden.data, requires_grad=True)
            ret = model.forward_end(hidden, target, *mems_end)
            loss, mems_end = ret[0], ret[1:]

            loss = loss.float().mean().type_as(loss)

            # backward  end 
            loss.backward()
            grad = model.get_grad_end()
            #shm_lists[2*(args.num_splits-1)+1].send(grad.cpu())
            shm_lists[2*(args.num_splits-1)+1].send(hidden.grad.data.cpu())
            #print(args.num_splits, ' send grad to ', 2*(args.num_splits-1)+1 , grad.norm())

            train_loss += loss.float().item()


            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()

            # receive and save grad 
            if model.delay <= 0:
                grad = shm_lists[1].recv()
                if args.cuda:
                    grad = grad.cuda(GPUS[0])
                model.save_grad_begin(grad) 
                #print(0, ' receive grad from ', 1 , grad.norm())

            # step-wise learning rate annealing
            train_step += 1
            if args.scheduler in ['cosine', 'constant', 'dev_perf']:
                # linear warmup stage
                if train_step < args.warmup_step:
                    curr_lr = args.lr * train_step / args.warmup_step
                    optimizer.param_groups[0]['lr'] = curr_lr
                else:
                    if args.scheduler == 'cosine':
                        scheduler.step(train_step)
            elif args.scheduler == 'inv_sqrt':
                scheduler.step(train_step)


            if train_step % args.log_interval == 0:
                cur_loss = train_loss / args.log_interval
                elapsed = time.time() - log_start_time
                log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                          '| ms/batch {:5.2f} | loss {:5.2f}'.format(
                    epoch, train_step, batch+1, optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval, cur_loss)
                if args.dataset in ['enwik8', 'text8']:
                    log_str += ' | bpc {:9.5f}'.format(cur_loss / math.log(2))
                else:
                    log_str += ' | ppl {:9.3f}'.format(math.exp(cur_loss))
                logging(log_str)
                train_loss = 0
                log_start_time = time.time()

            if train_step % args.eval_interval == 0:
                model.eval()

                # If the model does not use memory at all, make the ext_len longer.
                # Otherwise, make the mem_len longer and keep the ext_len the same.
                if args.mem_len == 0:
                    model.reset_length(args.eval_tgt_len,
                        args.ext_len+args.tgt_len-args.eval_tgt_len, args.mem_len)
                else:
                    model.reset_length(args.eval_tgt_len,
                        args.ext_len, args.mem_len+args.tgt_len-args.eval_tgt_len)

                mems_begin_eval = tuple()
                mems_end_eval = tuple()

                # Evaluation
                total_len, total_loss = 0, 0.
                with torch.no_grad():
                    for i, (data, target, seq_len) in enumerate(va_iter):
                        if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                            break
                        if seq_len > args.eval_tgt_len:
                            data = data[0:args.eval_tgt_len]
                            target = target[0:args.eval_tgt_len]
                        else:
                            while len(data) != args.eval_tgt_len:
                                data_copy_len = (args.eval_tgt_len - len(data)) if (args.eval_tgt_len - len(data) <= len(data)) else len(data)
                                data = torch.cat([data, data[0:data_copy_len]], 0)
                                target = torch.cat([target, target[0:data_copy_len]], 0)

                        ret = model.forward_begin(data, *mems_begin_eval)
                        hidden, mems_begin_eval = ret[0], ret[1:]
                        shm_lists_eval[split_id*2].send(hidden.data.cpu())

                        # forward end
                        hidden_end = shm_lists_eval[2*(args.num_splits-1)].recv()
                        if args.cuda:
                            hidden = hidden_end.cuda(GPUS[0])

                        hidden = Variable(hidden.data)
                        ret = model.forward_end(hidden, target, *mems_end_eval)
                        loss, mems_end_eval = ret[0], ret[1:]

                        loss = loss.mean()
                        total_loss += seq_len * loss.float().item()
                        total_len += seq_len

                # Switch back to the training mode
                model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
                model.train()

                val_loss = total_loss / total_len


                logging('-' * 100)
                log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| valid loss {:5.2f}'.format(
                    train_step // args.eval_interval, train_step,
                    (time.time() - eval_start_time), val_loss)
                if args.dataset in ['enwik8', 'text8']:
                    log_str += ' | bpc {:9.5f}'.format(val_loss / math.log(2))
                else:
                    log_str += ' | valid ppl {:9.3f}'.format(math.exp(val_loss))
                logging(log_str)
                logging('-' * 100)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    if not args.debug:
                        with open(os.path.join(args.work_dir, 'model_0.pt'), 'wb') as f:
                            torch.save(model, f)
                        with open(os.path.join(args.work_dir, 'optimizer_0.pt'), 'wb') as f:
                            torch.save(optimizer.state_dict(), f)
                    best_val_loss = val_loss

                # dev-performance based learning rate annealing
                if args.scheduler == 'dev_perf':
                    scheduler.step(val_loss)

                eval_start_time = time.time()

            if train_step == args.max_step:
                break
        if train_step == args.max_step:
            break



if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
    parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                    help='dataset name')
    parser.add_argument('--num_splits', type=int, default=2,
                    help='number of splits')
    parser.add_argument('--n_layer', type=int, default=12,
                    help='number of total layers')
    parser.add_argument('--n_head', type=int, default=10,
                    help='number of heads')
    parser.add_argument('--d_head', type=int, default=50,
                    help='head dimension')
    parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension')
    parser.add_argument('--d_model', type=int, default=500,
                    help='model dimension')
    parser.add_argument('--d_inner', type=int, default=1000,
                    help='inner dimension in FF')
    parser.add_argument('--dropout', type=float, default=0.0,
                    help='global dropout rate')
    parser.add_argument('--dropatt', type=float, default=0.0,
                        help='attention probability dropout rate')
    parser.add_argument('--init', default='normal', type=str,
                        help='parameter initializer to use.')
    parser.add_argument('--emb_init', default='normal', type=str,
                        help='parameter initializer to use.')
    parser.add_argument('--init_range', type=float, default=0.1,
                        help='parameters initialized by U(-init_range, init_range)')
    parser.add_argument('--emb_init_range', type=float, default=0.01,
                        help='parameters initialized by U(-init_range, init_range)')
    parser.add_argument('--init_std', type=float, default=0.02,
                        help='parameters initialized by N(0, init_std)')
    parser.add_argument('--proj_init_std', type=float, default=0.01,
                        help='parameters initialized by N(0, init_std)')
    parser.add_argument('--optim', default='adam', type=str,
                        choices=['adam', 'sgd', 'adagrad'],
                        help='optimizer to use.')
    parser.add_argument('--lr', type=float, default=0.00025,
                        help='initial learning rate (0.00025|5 for adam|sgd)')
    parser.add_argument('--mom', type=float, default=0.0,
                        help='momentum for sgd')
    parser.add_argument('--scheduler', default='cosine', type=str,
                        choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                        help='lr scheduler to use.')
    parser.add_argument('--warmup_step', type=int, default=0,
                        help='upper epoch limit')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='decay factor when ReduceLROnPlateau is used')
    parser.add_argument('--lr_min', type=float, default=0.0,
                        help='minimum learning rate during annealing')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--clip_nonemb', action='store_true',
                        help='only clip the gradient of non-embedding params')
    parser.add_argument('--max_step', type=int, default=100000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=60,
                        help='batch size')
    parser.add_argument('--batch_chunk', type=int, default=1,
                        help='split batch into chunks to save memory')
    parser.add_argument('--tgt_len', type=int, default=70,
                        help='number of tokens to predict')
    parser.add_argument('--eval_tgt_len', type=int, default=50,
                        help='number of tokens to predict for evaluation')
    parser.add_argument('--ext_len', type=int, default=0,
                        help='length of the extended context')
    parser.add_argument('--mem_len', type=int, default=0,
                        help='length of the retained previous heads')
    parser.add_argument('--not_tied', action='store_true',
                        help='do not tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--adaptive', action='store_true',
                        help='use adaptive softmax')
    parser.add_argument('--div_val', type=int, default=1,
                        help='divident value for adapative input and softmax')
    parser.add_argument('--pre_lnorm', action='store_true',
                        help='apply LayerNorm to the input instead of the output')
    parser.add_argument('--varlen', action='store_true',
                        help='use variable length')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='use multiple GPU')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='report interval')
    parser.add_argument('--eval-interval', type=int, default=4000,
                        help='evaluation interval')
    parser.add_argument('--work_dir', default='LM-TFM', type=str,
                        help='experiment directory.')
    parser.add_argument('--restart', action='store_true',
                        help='restart training from the saved checkpoint')
    parser.add_argument('--restart_dir', type=str, default='',
                        help='restart dir')
    parser.add_argument('--debug', action='store_true',
                        help='run in debug mode (do not create exp dir)')
    parser.add_argument('--same_length', action='store_true',
                        help='use the same attn length for all tokens')
    parser.add_argument('--attn_type', type=int, default=0,
                        help='attention type. 0 for ours, 1 for Shaw et al,'
                        '2 for Vaswani et al, 3 for Al Rfou et al.')
    parser.add_argument('--clamp_len', type=int, default=-1,
                        help='use the same pos embeddings after clamp_len')
    parser.add_argument('--eta_min', type=float, default=0.0,
                        help='min learning rate for cosine scheduler')
    parser.add_argument('--gpu0_bsz', type=int, default=-1,
                        help='batch size on gpu 0')
    parser.add_argument('--max_eval_steps', type=int, default=-1,
                        help='max eval steps')
    parser.add_argument('--sample_softmax', type=int, default=-1,
                        help='number of samples in sampled softmax')
    parser.add_argument('--patience', type=int, default=0,
                        help='patience')
    parser.add_argument('--finetune_v2', action='store_true',
                        help='finetune v2')
    parser.add_argument('--finetune_v3', action='store_true',
                        help='finetune v3')
    parser.add_argument('--fp16', action='store_true',
                        help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--static-loss-scale', type=float, default=1,
                        help='Static loss scale, positive power of 2 values can '
                        'improve fp16 convergence.')
    parser.add_argument('--dynamic-loss-scale', action='store_true',
                        help='Use dynamic loss scaling.  If supplied, this argument'
                        ' supersedes --static-loss-scale.')
    args = parser.parse_args()
    args.tied = not args.not_tied

    if args.d_embed < 0:
        args.d_embed = args.d_model

    assert args.ext_len >= 0, 'extended context length must be non-negative'
    assert args.batch_size % args.batch_chunk == 0

    args.work_dir = '{}-{}'.format(args.work_dir, args.dataset)
    args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
    logging = create_exp_dir(args.work_dir,
        scripts_to_save=['train.py', 'mem_transformer.py'], debug=args.debug)



    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed_all(args.seed)

    # Validate `--fp16` option
    if args.fp16:
        if not args.cuda:
            print('WARNING: --fp16 requires --cuda, ignoring --fp16 option')
            args.fp16 = False
        else:
            try:
                from apex.fp16_utils import FP16_Optimizer
            except:
                print('WARNING: apex not installed, ignoring --fp16 option')
                args.fp16 = False


    main(args, logging)
