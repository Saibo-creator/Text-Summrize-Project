import copy
import os
import pdb
import shutil
import time
from collections import OrderedDict, defaultdict

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

from data_loaders.summ_dataset import SummDataset
from data_loaders.summ_dataset_factory import SummDatasetFactory
#from data_loaders.yelp_dataset import YelpDataset
from data_loaders.hotel_dataset import HotelDataset
from evaluation.eval_utils import EvalMetrics
import sys

from models.nn_utils import classify_summ_batch, calc_lm_nll

# sys.path.append('external/text_summarizer')
# from external.text_summarizer.centroid_w2v import CentroidW2VSummarizer
from models.custom_parallel import DataParallelModel
from models.mlstm import StackedLSTMDecoder, StackedLSTMEncoder, StackedLSTM, mLSTM
from models.nn_utils import setup_gpus, OptWrapper, calc_grad_norm, \
    save_models, freeze, move_to_cuda, StepAnnealer
from models.summarization import SummarizationModel
from models.text_cnn import BasicTextCNN
from pretrain_classifier import TextClassifier
from project_settings import HParams, SAVED_MODELS_DIR, \
    EDOC_ID, RESERVED_TOKENS, WORD2VEC_PATH, EDOC_TOK, DatasetConfig, OUTPUTS_EVAL_DIR, OUTPUTS_COMPAR_DIR
from utils import create_argparse_and_update_hp, save_run_data, update_moving_avg, sync_run_data_to_bigstore, save_file
import warnings; warnings.simplefilter('ignore')


class Summarizer(object):
    def __init__(self, hp, opt, save_dir):
        self.hp = hp
        self.opt = opt
        self.save_dir = save_dir

    def unpack_sum_model_output(self, output):
        """
        SummmarizationModel is wrapped in a DataParallelModel (nn.DataParallel without final gather step).
        Depending on the number of GPUs being used, we may have to zip and combine the outputs.
        When there are multiple GPUs, the outputs are only cleanly combined along the batch dimension
        if the outputs are tensors.

        Returns:
            stats: dict (str to Tensor)
            summ_texts: list of strs
        """
        if self.ngpus == 1:
            stats, summ_texts = output
        else:
            stats_list, summ_texts_nested = zip(*output)  # list of dicts; list of lists
            stats = defaultdict(int)
            for stats_gpu in stats_list:
                for k, v in stats_gpu.items():
                    stats[k] += v
            stats = {k: v / self.ngpus for k, v in stats.items()}  # mean over gpus (e.g. mean of means)
            summ_texts = [text for gpu_texts in summ_texts_nested for text in gpu_texts]  # flatten
        return stats, summ_texts

    def update_dict(self, orig, updates):
        """
        Helper function to update / overwrite the orig dict
        """
        for k, v in updates.items():
            orig[k] = v
        return orig

    def prepare_individual_revs(self, texts, append_edoc=False):
        """
        Split concatenated reviews into individual reviews, tokenize, and create tensor

        Args:
            texts: list of strs, each str is n_docs concatenated together with EDOC_TOK delimiter

        Returns: [batch, n_docs, max_len (across all reviews)]
        """
        batch_size = len(texts)
        docs_list_reviews = [SummDataset.split_docs(text) for text in texts]  # list of lists of strs batch_size* ndoc*review
        docs_reviews = [rev for batch_item in docs_list_reviews for rev in batch_item]  # flatten
        dummy_ratings = [torch.LongTensor([0]) for _ in range(len(docs_reviews))]
        # We do this so that max_len is across all reviews
        if append_edoc:
            # Can use global_append_id because docs_ids is a flat [batch * n_docs]
            docs_ids, _, _ = self.dataset.prepare_batch(docs_reviews, dummy_ratings, global_append_id=EDOC_ID)
        else:
            docs_ids, _, _ = self.dataset.prepare_batch(docs_ids, dummy_ratings)  # [batch * n_docs, max_len]
        docs_ids = docs_ids.view(batch_size, -1, docs_ids.size(1))  # [batch, n_docs, max_len]
        return docs_ids

    def run_epoch(self, data_iter, nbatches, epoch, split,
                  sum_optimizer=None, discrim_optimizer=None, clf_optimizer=None,
                  cpkt_every=float('inf'), save_intermediate=False, run_val_subset=False,
                  store_all_rouges=False, store_all_summaries=False,
                  tb_writer=None, tb_start_step=0,true_summary=False):
        """
        Iterate through data in data_iter

        Args:
            data_iter: iterable providing minibatches
            nbatches: int (number of batches in data_iter)
                - could be less than then number of batches in the iter (e.g. when hp.train_subset is True)
            epoch: int
            split: str ('train', 'val')
            *_optimizer: Wrapped optim (e.g. OptWrapper)
                Passed during training split, not passed for validation

            cpkt_every: int (save a checkpoint and run on subset of validation set depending on subsequent two flags)
            save_intermediate: bool (save checkpoints every cpokt_every minibatches)
            run_val_subset: bool (run model on subset of validation set every cpokt_Every minibatches)

            store_all_rouges: boolean (store all rouges in addition to taking the average
                so we can plot the distribution)
            store_all_summaries: boolean (return all summaries)

            tb_writer: Tensorboard SummaryWriter
            tb_start_step: int
                - Starting step. Used when running on subset of validation set. This way the results
                can appear on the same x-axis timesteps as the training.
            true_summary: True when calculates rouge w.r.t true summary

        Returns:
            dict of str, floats containing losses and stats
            dict of rouge scores
            list of summaries
        """
        stats_avgs = defaultdict(int)
        evaluator = EvalMetrics(remove_stopwords=self.hp.remove_stopwords,
                                use_stemmer=self.hp.use_stemmer,
                                store_all=store_all_rouges,true_summary=self.hp.true_summary)
        summaries = []  # this is only added to if store_all_summaries is True
        ids = []

        #here print many reviews
        for s, (hotel_url ,texts, ratings, metadata) in enumerate(data_iter):#data_iter: iterable providing minibatches
            #以batch 为单位itererate
            print("ratings:",ratings)
            # texts: list of strs, each str is n_docs concatenated together with EDOC_TOK delimiter
            if s > nbatches:
                break
            # print(ratings) tensor([5, 4])
            stats = {}
            start = time.time()

            if sum_optimizer:
                sum_optimizer.optimizer.zero_grad()
            if discrim_optimizer:
                discrim_optimizer.optimizer.zero_grad()
            if clf_optimizer:
                clf_optimizer.optimizer.zero_grad()

            # Get data
            cycle_tgt_ids = None
            if self.hp.concat_docs:
                docs_ids, _, labels = self.dataset.prepare_batch(texts, ratings, doc_append_id=EDOC_ID)
                # docs_ids: [batch_size, max_len]
                if self.sum_cycle and (self.cycle_loss == 'rec'):
                    cycle_tgt_ids = self.prepare_individual_revs(texts)
            else:
                docs_ids = self.prepare_individual_revs(texts, append_edoc=True)
                cycle_tgt_ids = docs_ids
                labels = move_to_cuda(ratings - 1)

            extract_summ_ids = None
            if self.hp.extract_loss:
                extract_summs = []
                for text in texts:
                    summary = self.extract_sum.summarize(text.replace(EDOC_TOK, ''),
                                                         limit=self.hp.extractive_max_len)# limit=self.hp.yelp_extractive_max_len)
                    extract_summs.append(summary)
                dummy_ratings = [torch.LongTensor([0]) for _ in range(len(extract_summs))]
                extract_summ_ids, _, _ = self.dataset.prepare_batch(extract_summs, dummy_ratings)

            cur_tau = self.tau if isinstance(self.tau, float) else self.tau.val

            # Step for tensorboard: global steps in terms of number of reviews
            # This accounts for runs with different batch sizes and n_docs
            step = tb_start_step
            # We do the following so that if run_epoch is iterating over the validation subset,
            # the step is right around when run_epoch(self.val_subset_iter) was called. If we did step +=
            # (epoch * nbatches ...) for the validation subset and the cpkt_every was small, then the next
            # time run_epoch(self.val_subset_iter) was called might have a tb_start_step that was smaller
            # than the last step used for self.tb_val_sub_writer. This would make the Tensorboard line chart
            # loop back on itself.
            if tb_writer == self.tb_val_sub_writer:
                step += s
            else:
                step += (epoch * nbatches * self.hp.batch_size * self.hp.n_docs) + \
                        s * self.hp.batch_size * self.hp.n_docs

            # Adversarial
            discrim_gn = -1.0
            




            #do summerization on the batch of texts
            output = self.sum_model(docs_ids, labels,
                                    cycle_tgt_ids=cycle_tgt_ids,
                                    extract_summ_ids=extract_summ_ids,
                                    tau=cur_tau,
                                    adv_step=None,
                                    minibatch_idx=s, print_every_nbatches=self.opt.print_every_nbatches,
                                    tb_writer=tb_writer, tb_step=step,
                                    wass_loss=stats_avgs['wass_loss'],
                                    grad_pen_loss=stats_avgs['grad_pen_loss'],
                                    adv_gen_loss=stats_avgs['adv_gen_loss'],
                                    clf_loss=stats_avgs['clf_loss'],
                                    clf_acc=stats_avgs['clf_acc'],
                                    clf_avg_diff=stats_avgs['clf_avg_diff'])
            fwd_stats, summ_texts = self.unpack_sum_model_output(output)# sum_texts is of lenth = batch size 
            stats = self.update_dict(stats, fwd_stats)

            if self.hp.decay_tau:
                self.tau.step()



            # Classifier loss
            clf_gn = -1.0
            if clf_optimizer: # set be fault in MeanSum and our model
                retain_graph = sum_optimizer is not None
                stats['clf_loss'].backward(retain_graph=retain_graph)
                clf_gn = calc_grad_norm(self.clf_model)
                clf_optimizer.step()

            # Cycle loss
            sum_gn = -1.0
            if sum_optimizer:
                if self.hp.autoenc_docs and \
                        (not self.hp.load_ae_freeze):  # don't backward() if loaded pretrained autoenc (it's frozen)
                    retain_graph = self.hp.early_cycle or self.hp.sum_cycle or self.hp.extract_loss
                    stats['autoenc_loss'].backward(retain_graph=retain_graph)
                if self.hp.early_cycle and (not self.hp.autoenc_only):# False
                    stats['early_cycle_loss'].backward()
                if self.hp.length_loss:#True
                    stats['length_loss'].backward(retain_graph=True)
                if self.hp.sum_cycle and (not self.hp.autoenc_only):#True
                    retain_graph = self.hp.extract_loss
                    stats['cycle_loss'].backward(retain_graph=retain_graph)
                if self.hp.extract_loss and (not self.hp.autoenc_only):  #False
                    retain_graph = clf_optimizer is not None
                    stats['extract_loss'].backward(retain_graph=retain_graph)

                sum_gn = calc_grad_norm(self.docs_enc)
                sum_optimizer.step()

            # Gather summaries so we can calculate rouge
            clean_summs = []
            for idx in range(len(summ_texts)):
                summ = summ_texts[idx]
                for tok in RESERVED_TOKENS:  # should just be <pad> I think
                    summ = summ.replace(tok, '')
                clean_summs.append(summ)
                if store_all_summaries:
                    summaries.append(summ)
                    ids.append(hotel_url)


            # Calculate log likelihood of summaries using fixed language model (the one that was used to
            # initialize the models)
            ppl_time = time.time()
            summs_x, _, _ = self.dataset.prepare_batch(clean_summs, ratings)
            nll = calc_lm_nll(self.fixed_lm, summs_x)
            ppl_time = time.time() - ppl_time
            stats['nll'] = nll


            #
            # Stats, print, etc.
            #
            stats['total_loss'] = torch.tensor([v for k, v in stats.items() if 'loss' in k]).sum()
            for k, v in stats.items():
                stats_avgs[k] = update_moving_avg(stats_avgs[k], v.item(), s + 1)

            if s % self.opt.print_every_nbatches == 0:# self.opt.print_every_nbatches (default=1)
                # Calculate rouge
                if self.hp.true_summary:
                    try:
                        #(metadata['short_summary'])   ['Excellent highway hotel. Close to Universal Studios. Easily accessible by car and good for sightseeing. Great rooms and fantastic service. Great pool. Awesome vibe.', 'Excellent golf hotel. Fantastic service. Great pool. Great location. Awesome vibe.']
                        src_docs = [[summary] for summary in metadata['short_summary']] # it's src_docs when batch size >1  and src_doc when else

                        batch_avg_rouges = evaluator.batch_update_avg_rouge(clean_summs,src_docs,true_summary=self.hp.true_summary)  # def score(self, target, prediction):
                        #             FOUCS
                        print(type(batch_avg_rouges))
                        print(batch_avg_rouges)
                    except Exception as e:  # IndexError in computing (see commit for stack trace)
                        # This started occurring when I switched to Google's Rouge script
                        # It's happened after many minibatches (e.g. half way through the first epoch)
                        # I'm not sure if this is because the summary has degenerated into something that
                        # throws an error, or just that it's a rare edge case with the data.
                        # For now, print and log to tensorboard and see when and how often this occurs.
                        # batch_avg_rouges = evaluator.avg_rouges.
                        # Note: after some experiments, this only occurred twice in 4 epochs.
                        avg_rouges, min_rouges, max_rouges, std_rouges = \
                            evaluator.avg_avg_rouges, evaluator.avg_min_rouges, \
                            evaluator.avg_max_rouges, evaluator.avg_std_rouges
                        print('Error in calculating rouge')
                        if tb_writer:
                            tb_writer.add_scalar('other/rouge_error', 1, step)


                else:
                    try:

                        src_docs = [SummDataset.split_docs(concatenated) for concatenated in texts] #len(texts)= batch size



                        avg_rouges, min_rouges, max_rouges, std_rouges = \
                            evaluator.batch_update_avg_rouge(clean_summs, src_docs) #def score(self, target, prediction):
                        #             FOUCS src_docs=>

                    except Exception as e:  # IndexError in computing (see commit for stack trace)
                        # This started occurring when I switched to Google's Rouge script
                        # It's happened after many minibatches (e.g. half way through the first epoch)
                        # I'm not sure if this is because the summary has degenerated into something that
                        # throws an error, or just that it's a rare edge case with the data.
                        # For now, print and log to tensorboard and see when and how often this occurs.
                        # batch_avg_rouges = evaluator.avg_rouges.
                        # Note: after some experiments, this only occurred twice in 4 epochs.
                        avg_rouges, min_rouges, max_rouges, std_rouges = \
                            evaluator.avg_avg_rouges, evaluator.avg_min_rouges, \
                            evaluator.avg_max_rouges, evaluator.avg_std_rouges
                        print('Error in calculating rouge')
                        if tb_writer:
                            tb_writer.add_scalar('other/rouge_error', 1, step)

                # Construct print statements
                #{'rouge1': defaultdict(<class 'float'>, {}), 'rouge2': defaultdict(<class 'float'>, {}), 'rougeL': defaultdict(<class 'float'>, {})} {'rouge1': defaultdict(<class 'float'>, {}), 'rouge2': defaultdict(<class 'float'>, {}), 'rougeL': defaultdict(<class 'float'>, {})} {'rouge1': defaultdict(<class 'float'>, {}), 'rouge2': defaultdict(<class 'float'>, {}), 'rougeL': defaultdict(<class 'float'>, {})} {'rouge1': defaultdict(<class 'float'>, {}), 'rouge2': defaultdict(<class 'float'>, {}), 'rougeL': defaultdict(<class 'float'>, {})}
                mb_time = time.time() - start
                main_str = 'Epoch={}, batch={}/{}, split={}, time={:.4f}, tau={:.4f}'.format(
                    epoch, s, nbatches, split, mb_time, cur_tau)
                stats_str = ', '.join(['{}={:.4f}'.format(k, v) for k, v in stats.items()])
                stats_avgs_str = ', '.join(['{}_curavg={:.4f}'.format(k, v) for k, v in stats_avgs.items()])
                gn_str = 'sum_gn={:.2f}, discrim_gn={:.2f}, clf_gn={:.2f}'.format(sum_gn, discrim_gn, clf_gn)

                ################################# print rouge #######################################


                batch_rouge_strs = []
                if self.hp.true_summary:
                    for stat, rouges in {'avg': batch_avg_rouges}.items():
                        batch_rouge_strs.append('batch {} rouges: '.format(stat) + evaluator.to_str(rouges))
                    epoch_rouge_strs = []
                    for stat, rouges in evaluator.get_avg_stats_dicts().items():
                        epoch_rouge_strs.append('epoch  {} rouges: '.format(stat) + evaluator.to_str(rouges))

                    print_str = ' --- '.join([main_str, stats_str, stats_avgs_str, gn_str] +
                                             batch_rouge_strs + epoch_rouge_strs)

                    print(print_str)
                else:
                    for stat, rouges in {'avg': avg_rouges, 'min': min_rouges,
                                         'max': max_rouges, 'std': std_rouges}.items():
                        # FOUCS
                        batch_rouge_strs.append('batch avg {} rouges: '.format(stat) + evaluator.to_str(rouges))
                    epoch_rouge_strs = []
                    for stat, rouges in evaluator.get_avg_stats_dicts().items():
                        epoch_rouge_strs.append('epoch avg {} rouges: '.format(stat) + evaluator.to_str(rouges))

                    print_str = ' --- '.join([main_str, stats_str, stats_avgs_str, gn_str] +
                                             batch_rouge_strs + epoch_rouge_strs)

                    print(print_str)

                ################################# print example summary #######################################


                # Example summary to get qualitative sense
                print('\n', '-' * 100)
                print('ORIGINAL REVIEWS: ', texts[0].encode('utf8'))
                print('-' * 100)
                print('SUMMARY: ', summ_texts[0].encode('utf8'))
                print('-' * 100, '\n')

                if self.hp.true_summary:
                    print('true_summary: ', metadata['short_summary'][0].encode('utf8'))
                    print('-' * 100, '\n')


                print('\n', '#' * 100,'one batch finished','#' * 100, '\n')



            dataset_dir = self.opt.dataset if self.opt.az_cat is None else 'amazon_{}'.format(self.opt.az_cat)
            if self.opt.test_on_another_dataset:
                out_dir = os.path.join(OUTPUTS_COMPAR_DIR, '{} test_on_another_dataset:{}'.format(dataset_dir,self.opt.test_on_another_dataset))
            else:
                out_dir = os.path.join(OUTPUTS_EVAL_DIR, dataset_dir, 'n_docs_{}'.format(self.hp.n_docs),
                               'unsup_{}'.format(self.opt.notes))



            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            self.opt.write_every_nbatches = 5
            if s % self.opt.write_every_nbatches == 0:   

                output_per_batch_fp = os.path.join(out_dir, 'output_per_batch.txt')
                with open(output_per_batch_fp,'w') as file:
                    file.write(print_str)

                results=[]

                dic = {'docs': texts[0],
                        'summary': summ_texts[0],
                        'id': ids }  
                results.append(dic)

                per_batch_summs_out_fp = os.path.join(out_dir, 'per_batch_summaries.json')
                save_file(results, per_batch_summs_out_fp)

                # Write to tensorboard
                if tb_writer:
                    for k, v in stats.items():
                        tb_writer.add_scalar('stats/{}'.format(k), v, step)
                    for k, v in {'sum_gn': sum_gn, 'discrim_gn': discrim_gn, 'clf_gn': clf_gn}.items():
                        tb_writer.add_scalar('grad_norm/{}'.format(k), v, step)


                    if self.hp.true_summary:
                        for stat, rouges in {'avg': batch_avg_rouges}.items():
                            for rouge_name, d in rouges.items():
                                for metric_name, v in d.items():
                                    tb_writer.add_scalar('rouges_{}/{}/{}'.format(stat, rouge_name, metric_name), v, step)
                    else:
                        for stat, rouges in {'avg': avg_rouges, 'min': min_rouges,
                                             'max': max_rouges, 'std': std_rouges}.items():
                            for rouge_name, d in rouges.items():
                                for metric_name, v in d.items():
                                    tb_writer.add_scalar('rouges_{}/{}/{}'.format(stat, rouge_name, metric_name), v, step)

                    tb_writer.add_scalar('stats/sec_per_nll_calc', time.time() - ppl_time, step)

                    tb_writer.add_text('summary/orig_reviews', texts[0], step)
                    tb_writer.add_text('summary/summary', summ_texts[0], step)

                    tb_writer.add_scalar('stats/sec_per_batch', mb_time, step)

                    if self.hp.docs_attn:  # scalar may be learnable depending on flag
                        tb_writer.add_scalar('stats/context_alpha', self.summ_dec.context_alpha.item(), step)

                    mean_summ_len = np.mean([len(self.dataset.subwordenc.encode(summ)) for summ in clean_summs])
                    tb_writer.add_scalar('stats/mean_summ_len', mean_summ_len, step)

                    if (not self.hp.debug) and (not self.opt.no_bigstore):
                        sync_time = time.time()
                        sync_run_data_to_bigstore(self.save_dir, exp_sub_dir=self.opt.bs_dir,
                                                  method='rsync', tb_only=True)
                        tb_writer.add_scalar('stats/sec_per_bigstore_sync', time.time() - sync_time, step)

            # Periodic checkpointing
            if s % cpkt_every == 0:
                if save_intermediate:
                    print('Intermdediate checkpoint during training epoch')
                    save_model = self.sum_model.module if self.ngpus > 1 else self.sum_model
                    save_models(self.save_dir, {'sum_model': save_model, 'tau': self.tau},
                                self.optimizers, epoch, self.opt,
                                'sub{}'.format(int(s / cpkt_every)))

                if (s > 0) and run_val_subset:
                    start = time.time()
                    start_step = (epoch * nbatches * self.hp.batch_size * self.hp.n_docs) + \
                                 s * self.hp.batch_size * self.hp.n_docs

                    with torch.no_grad():
                        self.run_epoch(self.val_subset_iter, self.val_subset_iter.__len__(), epoch, 'val_subset',
                                       save_intermediate=False, run_val_subset=False,
                                       tb_writer=self.tb_val_sub_writer, tb_start_step=start_step)
                    tb_writer.add_scalar('stats/sec_per_val_subset', time.time() - start, start_step)
        print('*'*20)
        print('one epoch finished')
        print('*' * 20)
        return stats_avgs, evaluator, summaries, ids

    def train(self):
        """
        Main train loop
        """
        #
        # Get data, setup
        #
        self.dataset = SummDatasetFactory.get(self.opt.dataset)
        train_iter = self.dataset.get_data_loader(split='train', n_docs=self.hp.n_docs, sample_reviews=True,
                                                  category=self.opt.az_cat,
                                                  batch_size=self.hp.batch_size, shuffle=True)
        val_iter = self.dataset.get_data_loader(split='val', n_docs=self.hp.n_docs, sample_reviews=False,
                                                category=self.opt.az_cat,
                                                batch_size=self.hp.batch_size, shuffle=False)
        val_subset_iter = self.dataset.get_data_loader(split='val', n_docs=self.hp.n_docs, sample_reviews=False,
                                                       category=self.opt.az_cat,
                                                       subset=0.1,
                                                       batch_size=self.hp.batch_size, shuffle=False)
        self.val_subset_iter = val_subset_iter

        self.tau = self.hp.tau
        if self.hp.decay_tau:
            self.tau = StepAnnealer(self.hp.tau,
                                    interval_size=self.hp.decay_interval_size,
                                    # intervals=intervals, intervals_vals=intervals_vals,
                                    alpha=self.hp.decay_tau_alpha, method=self.hp.decay_tau_method,
                                    min_val=self.hp.min_tau)

        tb_path = os.path.join(self.save_dir, 'tensorboard/')
        print('Tensorboard events will be logged to: {}'.format(tb_path))
        os.mkdir(tb_path)
        os.mkdir(tb_path + 'train/')
        os.mkdir(tb_path + 'val/')
        self.tb_tr_writer = SummaryWriter(tb_path + 'train/')
        self.tb_val_writer = SummaryWriter(tb_path + 'val/')
        self.tb_val_sub_writer = SummaryWriter(tb_path + 'val_sub/')
        #
        # Get models, optimizers, and loss functions
        #
        self.ngpus = 1 if len(self.opt.gpus) == 1 else len(self.opt.gpus.split(','))
        self.models = {}  # used for saving
        self.optimizers = {}  # used for saving

        #
        # Summarization model
        #
        # Encoder-decoder for documents to summary
        self.fixed_lm = None
        if len(self.opt.load_lm) > 1:
            print('Loading pretrained language model from: {}'.format(self.opt.load_lm))


            if self.opt.cpu:
                self.docs_enc = torch.load(self.opt.load_lm, map_location='cpu')['model'] 

            elif torch.cuda.is_available():
                self.docs_enc = torch.load(self.opt.load_lm)['model']  # StackedLSTMEncoder


            
            self.docs_enc = self.docs_enc.module if isinstance(self.docs_enc, nn.DataParallel) \
                else self.docs_enc
        else:
            print('Training model from scratch')
            embed = nn.Embedding(self.dataset.subwordenc.vocab_size, self.hp.emb_size)
            lstm = StackedLSTM(mLSTM,
                               self.hp.lstm_layers, self.hp.emb_size, self.hp.hidden_size,
                               self.dataset.subwordenc.vocab_size,
                               self.hp.lstm_dropout,
                               layer_norm=self.hp.lstm_ln)
            self.docs_enc = StackedLSTMEncoder(embed, lstm)

        if self.hp.track_ppl:
            if len(self.opt.load_lm) > 1:
                self.fixed_lm = copy.deepcopy(self.docs_enc)
            else:
                # didn't pass in pretrained language model as we're training from scratch
                # load it from the default
                if self.opt.cpu:
                    self.fixed_lm = torch.load(self.dataset.conf.lm_path,map_location='cpu')['model']  # StackedLSTMEncoder

                elif torch.cuda.is_available():
                    self.fixed_lm = torch.load(self.dataset.conf.lm_path)['model']  # StackedLSTMEncoder




                self.fixed_lm = self.fixed_lm.module if isinstance(self.fixed_lm, nn.DataParallel) \
                    else self.fixed_lm

            freeze(self.fixed_lm)

        # --------------------------------Define document combination model-----------------------------------
        self.combine_encs_h_net = None
        self.combine_encs_c_net = None
        if self.hp.combine_encs == 'ff':
            self.combine_encs_h_net = nn.Sequential(OrderedDict([
                ('ln1', nn.LayerNorm(self.hp.n_docs * self.hp.hidden_size)),
                ('fc1', nn.Linear(self.hp.n_docs * self.hp.hidden_size, self.hp.hidden_size)),
                ('relu1', nn.ReLU()),
                ('ln2', nn.LayerNorm(self.hp.hidden_size)),
                ('fc2', nn.Linear(self.hp.hidden_size, self.hp.hidden_size))
            ]))
            if self.hp.combine_tie_hc:
                self.combine_encs_c_net = self.combine_encs_h_net
            else:
                self.combine_encs_c_net = copy.deepcopy(self.combine_encs_h_net)
        elif self.hp.combine_encs == 'gru':
            self.combine_encs_h_net = nn.GRU(self.hp.hidden_size, self.hp.hidden_size,
                                             num_layers=self.hp.combine_encs_gru_nlayers,
                                             batch_first=True,
                                             dropout=self.hp.combine_encs_gru_dropout,
                                             bidirectional=self.hp.combine_encs_gru_bi)
            if self.hp.combine_tie_hc:
                self.combine_encs_c_net = self.combine_encs_h_net
            else:
                self.combine_encs_c_net = copy.deepcopy(self.combine_encs_h_net)

        # Decoder for generating summaries
        self.summ_dec = StackedLSTMDecoder(copy.deepcopy(self.docs_enc.embed),
                                           copy.deepcopy(self.docs_enc.rnn),
                                           use_docs_attn=self.hp.docs_attn,
                                           attn_emb_size=self.hp.hidden_size,
                                           attn_hidden_size=self.hp.docs_attn_hidden_size,
                                           attn_learn_alpha=self.hp.docs_attn_learn_alpha)




        # -------------------------------- Autoencoder for documents-----------------------------------




        # Autoencoder for documents
        self.docs_autodec = None
        if self.hp.autoenc_docs:
            if self.hp.autoenc_docs_tie_dec:
                self.docs_autodec = StackedLSTMDecoder(self.summ_dec.embed, self.summ_dec.rnn)
            else:
                self.docs_autodec = StackedLSTMDecoder(copy.deepcopy(self.summ_dec.embed),
                                                       copy.deepcopy(self.summ_dec.rnn))

        # Encoder(-decoder) for summary to documents
        self.summ_enc = None
        self.docs_dec = None
        if self.hp.sum_cycle or self.hp.extract_loss:
            if self.hp.concat_docs:  # encoder is different: multi-reviews vs. "canonical" review -> representation
                self.summ_enc = StackedLSTMEncoder(copy.deepcopy(self.docs_enc.embed),
                                                   copy.deepcopy(self.docs_enc.rnn))
            else:  # encoder is same: one review  or "canonical" review (summary) -> representation
                if self.hp.tie_enc:
                    self.summ_enc = StackedLSTMEncoder(self.docs_enc.embed, self.docs_enc.rnn)
                else:
                    self.summ_enc = StackedLSTMEncoder(copy.deepcopy(self.docs_enc.embed),
                                                       copy.deepcopy(self.docs_enc.rnn))
        if self.hp.sum_cycle and self.hp.cycle_loss == 'rec':
            self.docs_dec = StackedLSTMDecoder(self.summ_dec.embed, self.summ_dec.rnn)


        # -------------------------------- Load a pretrained model and freeze----------------------------------

        # Load a pretrained model and freeze
        # 1. We may want this so that we have fixed, good representations for the documents.
        # This could be helpful, especially when we are using a FF or GRU to combine the n_docs representations
        # instead of taking the mean. Previous experiments without pretraining and freezing found these variants
        # of the model worse in terms of ROUGE and the loss decreasing. This may be simply because there's two
        # things to train at once (good document representations and how to combine them).
        # 2. Thus, we freeze everything except for the FF / GRU model
        if self.hp.load_ae_freeze:  # load autoencoder and freeze
            # SummarizationModel
            if self.opt.cpu:
                trained = torch.load(self.opt.load_autoenc, map_location='cpu')['sum_model']

            elif torch.cuda.is_available():
                trained = torch.load(self.opt.load_autoenc, map_location=lambda storage, loc: storage)['sum_model']

            
            trained = trained.module if isinstance(trained, nn.DataParallel) else trained
            # self.docs_enc = trained.docs_enc
            self.docs_enc = StackedLSTMEncoder(trained.docs_enc.embed, trained.docs_enc.rnn)
            self.summ_enc = StackedLSTMEncoder(self.docs_enc.embed, self.docs_enc.rnn)
            # self.sumn_enc = self.docs_enc # : not sure why this is different from the above

            self.docs_autodec = StackedLSTMDecoder(trained.docs_autodec.embed, trained.docs_autodec.rnn)
            # self.docs_autodec = trained.docs_autodec
            self.summ_dec = StackedLSTMDecoder(self.docs_autodec.embed,
                                               self.docs_autodec.rnn,
                                               use_docs_attn=self.hp.docs_attn,
                                               attn_emb_size=self.hp.hidden_size,
                                               attn_hidden_size=self.hp.docs_attn_hidden_size,
                                               attn_learn_alpha=self.hp.docs_attn_learn_alpha)
            if self.hp.sum_cycle and self.hp.cycle_loss == 'rec':
                self.docs_dec = StackedLSTMDecoder(self.summ_dec.embed, self.summ_dec.rnn)

            freeze(self.docs_enc)
            freeze(self.docs_autodec)
            freeze(self.summ_dec)
            freeze(self.summ_enc)

            # : I'm not sure if this is necessary or if it does anything
            # Note though that observing memory usage through nvidia-smi before and after doesn't
            # necessarily tell you, as the memory is "freed but not returned to the device"
            # https://discuss.pytorch.org/t/947
            del trained

        # Freeze embedding layers
        if self.hp.freeze_embed:
            for model in [self.docs_enc, self.docs_autodec, self.summ_dec, self.summ_enc, self.docs_dec]:
                if model:
                    freeze(model.embed)

        #
        # Discriminator
        #
        self.discrim_model = None
        self.discrim_optimizer = None
        if self.hp.sum_discrim:
            if len(self.opt.load_discrim):
                print('Loading pretrained discriminator from: {}'.format(self.opt.load_discrim))
                if self.hp.discrim_model == 'cnn':
                    text_model = torch.load(self.opt.load_discrim)['model']
                self.discrim_optimizer = OptWrapper(self.discrim_model, self.hp.sum_clip,
                                                    optim.Adam(text_model.parameters(), lr=self.hp.discrim_lr))
            else:
                print('Path to pretrained discriminator not given: training from scratch')
                if self.hp.discrim_model == 'cnn':
                    cnn_output_size = self.hp.cnn_n_feat_maps * len(self.hp.cnn_filter_sizes)
                    text_model = TextClassifier(self.dataset.subwordenc.vocab_size, self.hp.emb_size,
                                                self.hp.cnn_filter_sizes, self.hp.cnn_n_feat_maps, self.hp.cnn_dropout,
                                                cnn_output_size, self.dataset.n_ratings_labels,
                                                onehot_inputs=self.hp.discrim_onehot)
            self.discrim_model = Discriminator(text_model, self.hp.discrim_model)
            discrim_params = [p for p in self.discrim_model.parameters() if p.requires_grad]
            self.discrim_optimizer = OptWrapper(self.discrim_model, self.hp.sum_clip,
                                                optim.Adam(discrim_params, lr=self.hp.discrim_lr))
            self.optimizers['discrim_optimizer'] = self.discrim_optimizer

        # -------------------------------- Load a clf model  ----------------------------------

        #
        # Classifier
        #
        self.clf_model = None
        self.clf_optimizer = None


        if self.hp.sum_clf:
            if len(self.opt.load_clf) > 0:
                print('Loading pretrained classifier from: {}'.format(self.opt.load_clf))
                if self.opt.cpu:
                    self.clf_model = torch.load(self.opt.load_clf,map_location='cpu')['model']

                elif torch.cuda.is_available():
                    self.clf_model = torch.load(self.opt.load_clf)['model']


                
                #self.clf_model = None
            else:
                print('Path to pretrained classifer not given: training from scratch')
                cnn_output_size = self.hp.cnn_n_feat_maps * len(self.hp.cnn_filter_sizes)
                self.clf_model = nn.Sequential(OrderedDict([
                    ('embed', nn.Embedding(self.dataset.subwordenc.vocab_size, self.hp.emb_size)),
                    ('cnn', BasicTextCNN(self.hp.cnn_filter_sizes, self.hp.cnn_n_feat_maps, self.hp.emb_size,
                                         self.hp.cnn_dropout)),
                    ('fc_out', nn.Linear(cnn_output_size, self.dataset.n_ratings_labels))
                ]))
            clf_params = [p for p in self.clf_model.parameters() if p.requires_grad]

            if self.hp.sum_clf_lr > 0: #hp.sum_clf_lr=0.0 by default cf project_settings.py
                self.clf_optimizer = OptWrapper(self.clf_model, self.hp.sum_clip,
                                                optim.Adam(clf_params, lr=self.hp.sum_clf_lr))
                self.optimizers['clf_optimizer'] = self.clf_optimizer
            else:
                freeze(self.clf_model)   

  # -------------------------------- Init a summarization Model  ----------------------------------
        #
        # Overall model
        #
        self.sum_model = SummarizationModel(self.docs_enc, self.docs_autodec,
                                            self.combine_encs_h_net, self.combine_encs_c_net, self.summ_dec,
                                            self.summ_enc, self.docs_dec,
                                            self.discrim_model, self.clf_model,
                                            self.fixed_lm,
                                            self.hp, self.dataset)
        self.models['sum_model'] = self.sum_model

        # Exclude discriminator and classifier as they have their own optimizers
        sum_optim_params = [p for n, p in self.sum_model.named_parameters() if ('discrim' not in n) and \
                            ('clf' not in n) and p.requires_grad]
        self.sum_optimizer = OptWrapper(self.sum_model, self.hp.sum_clip,
                                        optim.Adam(sum_optim_params, lr=self.hp.sum_lr))
        self.optimizers['sum_optimizer'] = self.sum_optimizer

        # Count number of params
        all_params = self.sum_model.parameters()
        all_params = [p for params in all_params for p in params]  # flatten
        print('Number of parameters: {}'.format(sum([p.nelement() for p in all_params])))
        all_trainable_params = [p for p in all_params if p.requires_grad]
        print('Number of trainable parameters: {}'.format(sum([p.nelement() for p in all_trainable_params])))

        #
        # Get extractive summarizer if using that loss
        #
        if self.hp.extract_loss:
            self.extract_sum = CentroidW2VSummarizer(WORD2VEC_PATH, length_limit=2,
                                                     topic_threshold=0.3, sim_threshold=0.95,
                                                     reordering=True, subtract_centroid=False, keep_first=False,
                                                     bow_param=0, length_param=0, position_param=0,
                                                     debug=False)




  # -------------------------------- Move to cuda and parallelize ----------------------------------


        #
        # Move to cuda and parallelize
        #
        if torch.cuda.is_available():
            self.sum_model.cuda()
        if self.ngpus > 1:
            self.sum_model = DataParallelModel(self.sum_model) #Paralize Summarization model



        ########################################################
        ########################################################
        #
        # Train (Below)
        #
        ########################################################
        ########################################################



        #
        # Train
        #
        for epoch in range(self.hp.max_nepochs):
            try:
                self.sum_model.train()

                if (self.hp.n_docs_min > 0) and (self.hp.n_docs_max > 0):
                    # Creation of data loader shuffles and random seed will result in shuffling every epoch
                    train_iter = self.dataset.get_data_loader(split='train',
                                                              n_docs_min=self.hp.n_docs_min,
                                                              n_docs_max=self.hp.n_docs_max,
                                                              sample_reviews=True,
                                                              seed=epoch,
                                                              category=self.opt.az_cat)

                nbatches = train_iter.__len__()

  # -------------------------------- run epoch----------------------------------

                stats_avgs, evaluator, _,_ = self.run_epoch(
                    train_iter, nbatches, epoch, 'train',
                    sum_optimizer=self.sum_optimizer,
                    discrim_optimizer=self.discrim_optimizer,
                    clf_optimizer=self.clf_optimizer,
                    # cpkt_every=5, save_intermediate=True, run_val_subset=True,
                    cpkt_every=int(nbatches / 10), save_intermediate=True, run_val_subset=True,
                    tb_writer=self.tb_tr_writer)
  # -------------------------------- add results to tensorboard---------------------------------
                for k, v in stats_avgs.items():
                    self.tb_tr_writer.add_scalar('overall_stats/{}'.format(k), v, epoch)
                for stat, rouges in evaluator.get_avg_stats_dicts().items():
                    for rouge_name, d in rouges.items():
                        for metric_name, v in d.items():
                            self.tb_tr_writer.add_scalar('overall_rouges_{}/{}/{}'.format(
                                stat, rouge_name, metric_name), v, epoch)
            except KeyboardInterrupt:
                print('Exiting from training early')


  # -------------------------------- Run on validation ------------------------------------------
            # Run on validation
            self.sum_model.eval()
            if self.hp.train_subset == 1.0:
                stats_avgs, evaluator, _ ,_= self.run_epoch(val_iter, val_iter.__len__(), epoch, 'val',
                                                          save_intermediate=False, run_val_subset=False,
                                                          tb_writer=self.tb_val_writer)
                for k, v in stats_avgs.items():
                    self.tb_val_writer.add_scalar('overall_stats/{}'.format(k), v, epoch)
                for stat, rouges in evaluator.get_avg_stats_dicts().items():
                    for rouge_name, d in rouges.items():
                        for metric_name, v in d.items():
                            self.tb_val_writer.add_scalar('overall_rouges_{}/{}/{}'.format(
                                stat, rouge_name, metric_name), v, epoch)
            save_model = self.sum_model.module if self.ngpus > 1 else self.sum_model
            save_models(self.save_dir, {'sum_model': save_model, 'tau': self.tau}, self.optimizers, epoch, self.opt,
                        'tot{:.2f}_r1f{:.2f}'.format(stats_avgs['total_loss'],
                                                     evaluator.avg_avg_rouges['rouge1']['f']))











    def test(self):
        """
        Run trained model on test set
        true_summary: True when calculates rouge w.r.t true summary
        """
  # -------------------------------- Load data ------------------------------------------


        self.dataset = SummDatasetFactory.get(self.opt.dataset)
        if self.opt.test_on_another_dataset:
            self.second_dataset = SummDatasetFactory.get(self.opt.test_on_another_dataset)
        else:
            self.second_dataset = SummDatasetFactory.get(self.opt.dataset)

        if self.opt.test_group_ratings:
            def grouped_reviews_iter(n_docs):
                store_path = os.path.join(self.dataset.conf.processed_path, 'test',
                                          'plHKBwA18aWeP-TG8DC96Q_reviews.json')
                                          # 'SqxIx0KbTmCvUlOfkjamew_reviews.json')
                from utils import load_file
                revs = load_file(store_path)
                rating_to_revs = defaultdict(list)
                for rev in revs:
                    rating_to_revs[rev['stars']].append(rev['text'])
                for rating in [1, 3, 5]:
                    # Want to return same variables as dataloader iter
                    texts = [SummDataset.concat_docs(rating_to_revs[rating][:n_docs])]
                    ratings = torch.LongTensor([rating])
                    metadata = {'item': ['SqxIx0KbTmCvUlOfkjamew'],
                                'categories': ['Restaurants---Vegan---Thai'],
                                'city': ['Las Vegas']}
                    yield(texts, ratings, metadata)
            self.hp.batch_size = 1
            test_iter  = grouped_reviews_iter(self.hp.n_docs)
            test_iter_len = 3
        else:
            test_iter = self.second_dataset.get_data_loader(split='test', sample_reviews=False, n_docs=self.hp.n_docs,
                                                     category=self.opt.az_cat,
                                                     batch_size=self.hp.batch_size, shuffle=False)
            test_iter_len = test_iter.__len__()


        self.tb_val_sub_writer = None
  # -------------------------------- get model and loss ------------------------------------------
        #
        # Get model and loss
        #

        if self.opt.cpu:
            ckpt = torch.load(self.opt.load_test_sum, map_location='cpu')

        elif torch.cuda.is_available():
            ckpt = torch.load(self.opt.load_test_sum, map_location=lambda storage, loc: storage)
        
        self.sum_model = ckpt['sum_model']
        # We should always be loading from the checkpoint, but I wasn't saving it earlier
        # Tau may have been decayed over the course of training, so want to use the tau at the time of checkpointing
        self.tau = self.hp.tau
        if 'tau' in ckpt:
            self.tau = ckpt['tau']
        # We may want to test with a different n_docs than what was used during training
        # Update the checkpointed model
        self.sum_model.hp.n_docs = self.hp.n_docs

        # For tracking NLL of generated summaries


        if self.opt.cpu:
            self.fixed_lm = torch.load(self.dataset.conf.lm_path,map_location='cpu')['model']  # StackedLSTMEncoder

        elif torch.cuda.is_available():
            self.fixed_lm = torch.load(self.dataset.conf.lm_path)['model']  # StackedLSTMEncoder

        
        self.fixed_lm = self.fixed_lm.module if isinstance(self.fixed_lm, nn.DataParallel) \
            else self.fixed_lm


        # Adding this now for backwards compatability
        # Was testing with a model that didn't have early_cycle
        # Because the ckpt is the saved SummarizationModel, which contains a hp attribute, it will not have
        # self.hp.early_cycle. I do save a snapshot of the code used to train the model and could load that.
        # However, I should really just be saving the state_dict of the model.
        if not hasattr(self.sum_model.hp, 'early_cycle'):
            self.sum_model.hp.early_cycle = False
        if not hasattr(self.sum_model.hp, 'cos_honly'):
            self.sum_model.hp.cos_honly = False
        if not hasattr(self.sum_model.hp, 'cos_wgt'):
            self.sum_model.hp.cos_wgt = 1.0
        if not hasattr(self.sum_model.hp, 'tie_enc'):
            self.sum_model.hp.tie_enc = True

        if torch.cuda.is_available():
            self.sum_model.cuda()
        self.ngpus = 1
        if len(self.opt.gpus) > 1:
            self.ngpus = len(self.opt.gpus.split(','))
            self.sum_model = DataParallelModel(self.sum_model)

        n_params = sum([p.nelement() for p in self.sum_model.parameters()])

        print('Number of parameters: {}'.format(n_params))
        



        # Note: starting from here, this code is similar to lm_autoenc_baseline() and the
        # end of run_summarization_baseline()
  # -------------------------------- Run on test set(only one epoch precisely) ------------------------------------------
        #
        # Run on test set
        #
        self.sum_model.eval()  # like a switch, turn off some specific layers used for training to accelerate

        # Note: in order to run a model trained on the Yelp dataset on the Amazon dataset,
        # you have to uncomment the following line. This is because the two models
        # have slightly different vocab_size's, and vocab_size is used inside run_epoch.
        # (They have slightly different vocab_size because the subword encoder for
        # both is built using a *target* size of 32000, but the actual size is slightly
        # lower or higher than 32000).
        # self.dataset = SummDatasetFactory.get('yelp')
        # : handle this better
        with torch.no_grad():    
            evaluator: EvalMetrics
            stats_avgs, evaluator, summaries,ids  = self.run_epoch(test_iter, test_iter_len, 0, 'test',
                                                              save_intermediate=False, run_val_subset=False,
                                                              store_all_rouges=True, store_all_summaries=True,true_summary=self.hp.true_summary)
        # raise ValueError('A very specific bad thing happened after print')

        #
        # Pass summaries through classifier
        #
        # Note: I know that since the SummarizationModel already calculates the classification accuracy
        # if sum_clf=True. Hence, technically, I could refactor it to add everything that I'd like to compute
        # in the forward pass and add to stats(). However, I think it's cleaner /easier to just do everything
        # I want here, especially if I add more things like per rating counts and accuracy. Plus,
        # it's just one pass through the test set -- which I'll run infrequently to evaluate a trained model.
        # I think that it takes more time is fine.
        #
 # -------------------------------- collect output data ------------------------------------------

        results = []
        if self.hp.sum_clf:
            accuracy = 0.0
            true_rating_dist = defaultdict(int)  # used to track distribution of mean ratings
            per_rating_counts = defaultdict(int)  # these are predicted ratnigs
            per_rating_acc = defaultdict(int)

            clf_model = self.sum_model.module.clf_model if self.ngpus > 1 else self.sum_model.clf_model

            if self.opt.test_group_ratings:
                test_iter  = grouped_reviews_iter(self.hp.n_docs)

            for i, (hotel_url, texts, ratings_batch, metadata) in enumerate(test_iter):
                summaries_batch = summaries[i * self.hp.batch_size: i * self.hp.batch_size + len(texts)]
                ids_batch = ids[i * self.hp.batch_size: i * self.hp.batch_size + len(texts)]
                acc, per_rating_counts, per_rating_acc, pred_ratings_batch, pred_probs_batch = \
                    classify_summ_batch(clf_model, summaries_batch, ratings_batch, self.dataset,
                                        per_rating_counts, per_rating_acc)

                for rating in ratings_batch:
                    true_rating_dist[rating.item()] += 1

                if acc is None:
                    print('Summary was too short to classify')
                    pred_ratings_batch = [torch.LongTensor([-1]) for _ in range(len(summaries_batch))]
                    pred_probs_batch = [torch.LongTensor([-1]) for _ in range(len(summaries_batch))]
                else:
                    accuracy = update_moving_avg(accuracy, acc.item(), i + 1)

                for j in range(len(summaries_batch)):
                    try:
                        dic = {#'id':ids_batch[j],
                               'docs': texts[j],
                               'summary': summaries_batch[j],
                               'rating': ratings_batch[j].item(),
                               'pred_rating': pred_ratings_batch[j].item(),
                               'pred_prob': pred_probs_batch[j].item()}
                        for k, values in metadata.items():
                            dic[k] = values[j]
                        results.append(dic)
                    except Exception as e:
                        pass

                print(dic)
        else:
            for i, (hotel_url, texts, ratings_batch, metadata) in enumerate(test_iter):
                summaries_batch = summaries[i * self.hp.batch_size: i * self.hp.batch_size + len(texts)]

                for j in range(len(summaries_batch)):
                    dic = {#'id':ids_batch[j],
                           'docs': texts[j],
                           'summary': summaries_batch[j],
                           'rating': ratings_batch[j].item(),
                           'pred_rating': None,    #changed
                           'pred_prob': None}      #changed
                    for k, values in metadata.items():
                        dic[k] = values[j]
                    results.append(dic)
                    
 # -------------------------------- save output data ------------------------------------------

        # Save summaries, rouge scores, and rouge distributions figures
        dataset_dir = self.opt.dataset if self.opt.az_cat is None else 'amazon_{}'.format(self.opt.az_cat)
        
        dataset_dir = self.opt.dataset if self.opt.az_cat is None else 'amazon_{}'.format(self.opt.az_cat)
        if self.opt.test_on_another_dataset:
            out_dir = os.path.join(OUTPUTS_COMPAR_DIR, '{}_test_on_another_dataset:{}'.format(dataset_dir,self.opt.test_on_another_dataset))
        else:
            out_dir = os.path.join(OUTPUTS_EVAL_DIR, dataset_dir, 'n_docs_{}'.format(self.hp.n_docs),
                           'unsup_{}'.format(self.opt.notes))

        if not os.path.exists(OUTPUTS_COMPAR_DIR):
            os.mkdir(OUTPUTS_COMPAR_DIR)

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        summs_out_fp = os.path.join(out_dir, 'summaries.json')
        save_file(results, summs_out_fp)


        param_out_fp = os.path.join(out_dir, 'num_param.txt')
        with open(param_out_fp,'w') as file:
            file.write('Number of parameters: {}'.format(n_params))


        if self.hp.sum_clf:
            true_rating_dist = {k: v / float(sum(true_rating_dist.values())) for k, v in true_rating_dist.items()}
            out_fp = os.path.join(out_dir, 'classificaton_acc.json')
            save_file({'acc': accuracy, 'per_rating_acc': per_rating_acc, 'true_rating_dist': true_rating_dist}, out_fp)

            print('-' * 50)
            print('Stats:')
            print('Rating accuracy: ', accuracy)
            print('Per rating accuracy: ', dict(per_rating_acc))

        
        out_fp = os.path.join(out_dir, 'stats.json')
        save_file(stats_avgs, out_fp)

        print('-' * 50)
        print('Rouges:')
        for stat, rouge_dict in evaluator.get_avg_stats_dicts().items():
            #if true_summary: evaluator.get_avg_stats_dicts().items()={ "avg": self.avg_rouges }
            # else:  {'avg': self.avg_avg_rouges,
            #         'min': self.avg_min_rouges,
            #         'max': self.avg_max_rouges,
            #         'std': self.avg_std_rouges}

            print('-' * 50)
            print(stat.upper())
            print(evaluator.to_str(rouge_dict))

            out_fp = os.path.join(out_dir, 'avg_{}-rouges.json'.format(stat))
            save_file(rouge_dict, out_fp)
            out_fp = os.path.join(out_dir, 'avg_{}-rouges.csv'.format(stat))
            evaluator.to_csv(rouge_dict, out_fp)

        out_fp = os.path.join(out_dir, '{}-rouges.pdf')
        evaluator.plot_rouge_distributions(show=self.opt.show_figs, out_fp=out_fp)
