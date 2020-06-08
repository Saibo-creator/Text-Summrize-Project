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
    EDOC_ID, RESERVED_TOKENS, WORD2VEC_PATH, EDOC_TOK, DatasetConfig, OUTPUTS_EVAL_DIR
from utils import create_argparse_and_update_hp, save_run_data, update_moving_avg, sync_run_data_to_bigstore, save_file
from project_settings import HParams, SAVED_MODELS_DIR, \
    EDOC_ID, RESERVED_TOKENS, WORD2VEC_PATH, EDOC_TOK, DatasetConfig, OUTPUTS_DIR



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
                  tb_writer=None, tb_start_step=0):
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

        Returns:
            dict of str, floats containing losses and stats
            dict of rouge scores
            list of summaries
        """
        stats_avgs = defaultdict(int)
        evaluator = EvalMetrics(remove_stopwords=self.hp.remove_stopwords,
                                use_stemmer=self.hp.use_stemmer,
                                store_all=store_all_rouges)
        summaries = []  # this is only added to if store_all_summaries is True


        
        for s, (texts, ratings, metadata) in enumerate(data_iter):
            # texts: list of strs, each str is n_docs concatenated together with EDOC_TOK delimiter
            if s > nbatches:
                break

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


            # if self.hp.decay_tau:
            #     self.tau.step()

            # Classifier loss
            clf_gn = -1.0
            if clf_optimizer:
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
                if self.hp.early_cycle and (not self.hp.autoenc_only):
                    stats['early_cycle_loss'].backward()
                if self.hp.sum_cycle and (not self.hp.autoenc_only):
                    retain_graph = self.hp.extract_loss
                    stats['cycle_loss'].backward(retain_graph=retain_graph)
                if self.hp.extract_loss and (not self.hp.autoenc_only):
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

            if s % self.opt.print_every_nbatches == 0:
                # Calculate rouge
                try:
                    src_docs = [SummDataset.split_docs(concatenated) for concatenated in texts]
                    avg_rouges, min_rouges, max_rouges, std_rouges = \
                        evaluator.batch_update_avg_rouge(clean_summs, src_docs) #def score(self, target, prediction):
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
                mb_time = time.time() - start
                main_str = 'Epoch={}, batch={}/{}, split={}, time={:.4f}, tau={:.4f}'.format(
                    epoch, s, nbatches, split, mb_time, cur_tau)
                stats_str = ', '.join(['{}={:.4f}'.format(k, v) for k, v in stats.items()])
                stats_avgs_str = ', '.join(['{}_curavg={:.4f}'.format(k, v) for k, v in stats_avgs.items()])
                gn_str = 'sum_gn={:.2f}, discrim_gn={:.2f}, clf_gn={:.2f}'.format(sum_gn, discrim_gn, clf_gn)

                batch_rouge_strs = []
                for stat, rouges in {'avg': avg_rouges, 'min': min_rouges,
                                     'max': max_rouges, 'std': std_rouges}.items():
                    batch_rouge_strs.append('batch avg {} rouges: '.format(stat) + evaluator.to_str(rouges))
                epoch_rouge_strs = []
                for stat, rouges in evaluator.get_avg_stats_dicts().items():
                    epoch_rouge_strs.append('epoch avg {} rouges: '.format(stat) + evaluator.to_str(rouges))

                print_str = ' --- '.join([main_str, stats_str, stats_avgs_str, gn_str] +
                                         batch_rouge_strs + epoch_rouge_strs)
                print(print_str)

                # Example summary to get qualitative sense
                print('\n', '-' * 100)
                print('ORIGINAL REVIEWS: ', texts[0].encode('utf8'))
                print('-' * 100)
                print('SUMMARY: ', summ_texts[0].encode('utf8'))
                print('-' * 100, '\n')


                print('\n', '#' * 100, '\n')

                # Write to tensorboard
                if tb_writer:
                    for k, v in stats.items():
                        tb_writer.add_scalar('stats/{}'.format(k), v, step)
                    for k, v in {'sum_gn': sum_gn, 'discrim_gn': discrim_gn, 'clf_gn': clf_gn}.items():
                        tb_writer.add_scalar('grad_norm/{}'.format(k), v, step)

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

        return stats_avgs, evaluator, summaries




    def test(self):
        """
        Run trained model on test set
        """
        self.dataset = SummDatasetFactory.get(self.opt.dataset)
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
            test_iter = self.dataset.get_data_loader(split='test', sample_reviews=False, n_docs=self.hp.n_docs,
                                                     category=self.opt.az_cat,
                                                     batch_size=self.hp.batch_size, shuffle=False)
            test_iter_len = test_iter.__len__()


        self.tb_val_sub_writer = None

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
            stats_avgs, evaluator, summaries = self.run_epoch(test_iter, test_iter_len, 0, 'test',
                                                              save_intermediate=False, run_val_subset=False,
                                                              store_all_rouges=True, store_all_summaries=True)
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
        results = []
        if self.hp.sum_clf:
            
            accuracy = 0.0
            true_rating_dist = defaultdict(int)  # used to track distribution of mean ratings
            per_rating_counts = defaultdict(int)  # these are predicted ratnigs
            per_rating_acc = defaultdict(int)

            clf_model = self.sum_model.module.clf_model if self.ngpus > 1 else self.sum_model.clf_model
            if self.opt.test_group_ratings:
                test_iter  = grouped_reviews_iter(self.hp.n_docs)

            for i, (texts, ratings_batch, metadata) in enumerate(test_iter):
                summaries_batch = summaries[i * self.hp.batch_size: i * self.hp.batch_size + len(texts)]
                acc, per_rating_counts, per_rating_acc, pred_ratings, pred_probs = \
                    classify_summ_batch(clf_model, summaries_batch, ratings_batch, self.dataset,
                                        per_rating_counts, per_rating_acc)

                for rating in ratings_batch:
                    true_rating_dist[rating.item()] += 1

                if acc is None:
                    print('Summary was too short to classify')
                    pred_ratings = [None for _ in range(len(summaries_batch))]
                    pred_probs = [None for _ in range(len(summaries_batch))]
                else:
                    accuracy = update_moving_avg(accuracy, acc.item(), i + 1)

                for j in range(len(summaries_batch)):
                    dic = {'docs': texts[j],
                           'summary': summaries_batch[j],
                           'rating': ratings_batch[j].item(),
                           'pred_rating': pred_ratings[j].item(),
                           'pred_prob': pred_probs[j].item()}
                    for k, values in metadata.items():
                        dic[k] = values[j]
                    results.append(dic)
        else:
            for i, (texts, ratings_batch, metadata) in enumerate(test_iter):
                summaries_batch = summaries[i * self.hp.batch_size: i * self.hp.batch_size + len(texts)]

                for j in range(len(summaries_batch)):
                    dic = {'docs': texts[j],
                           'summary': summaries_batch[j],
                           'rating': ratings_batch[j].item(),
                           'pred_rating': None,    #changed
                           'pred_prob': None}      #changed
                    for k, values in metadata.items():
                        dic[k] = values[j]
                    results.append(dic)


        # Save summaries, rouge scores, and rouge distributions figures
        dataset_dir = self.opt.dataset if self.opt.az_cat is None else 'amazon_{}'.format(self.opt.az_cat)
        out_dir = os.path.join(OUTPUTS_EVAL_DIR, dataset_dir, 'n_docs_{}'.format(self.hp.n_docs),
                               'unsup_{}'.format(self.opt.notes))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        summs_out_fp = os.path.join(out_dir, 'summaries.json')
        save_file(results, summs_out_fp)


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
            print('-' * 50)
            print(stat.upper())
            print(evaluator.to_str(rouge_dict))

            out_fp = os.path.join(out_dir, 'avg_{}-rouges.json'.format(stat))
            save_file(rouge_dict, out_fp)
            out_fp = os.path.join(out_dir, 'avg_{}-rouges.csv'.format(stat))
            evaluator.to_csv(rouge_dict, out_fp)

        out_fp = os.path.join(out_dir, '{}-rouges.pdf')
        evaluator.plot_rouge_distributions(show=self.opt.show_figs, out_fp=out_fp)














class compariser:
	def __init__(self, dataset1, dataset2,save_dir):
		self.dataset1 = dataset1
      	self.dataset2 = dataset2
      	self.save_dir = save_dir
   	def compare(self):
    	test_iter = self.dataset.get_data_loader(split='test', sample_reviews=False, n_docs=self.hp.n_docs,
                                                     category=self.opt.az_cat,
                                                     batch_size=self.hp.batch_size, shuffle=False)
    	test_iter_len = test_iter.__len__()
   			with torch.no_grad():    
   		stats_avgs, evaluator, summaries = self.run_epoch(test_iter, test_iter_len, 0, 'test',
                                                              save_intermediate=False, run_val_subset=False,
                                                              store_all_rouges=True, store_all_summaries=True)






 
   	def displayEmployee(self):
      	print "Name : ", self.name,  ", Salary: ", self.salary
 









if __name__ == '__main__':
	hp = HParams()
	hp, run_name, parser = create_argparse_and_update_hp(hp)
	opt = parser.parse_args()
    setup_gpus(opt.gpus, hp.seed)

	dataset1='hotel_mask'
	dataset2='hotel'
	ds_conf1 = DatasetConfig(dataset1)
	ds_conf2 = DatasetConfig(dataset2)


	output_base_path = os.path.join(OUTPUTS_DIR, '/compar/')
	if not os.path.exists(output_path):
	    os.mkdir(output_path)

	#save_dir = os.path.join(opt.save_model_basedir.format(hp.model_type, opt.dataset), run_name)

	# text = input("Enter your text to summarize: ") 
	# print(text)

	summarizer = Summarizer(hp, opt, save_dir)
    summarizer.test()




