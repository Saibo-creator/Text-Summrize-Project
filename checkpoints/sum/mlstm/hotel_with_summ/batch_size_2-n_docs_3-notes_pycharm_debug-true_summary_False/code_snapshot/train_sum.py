# train_sum.py

"""
Train and test unsupervised summarization model

Usage:
Train model:
1. python train_sum.py --gpus=0,1,2,3 --batch_size=16 \
--tau=2.0 --bs_dir=tmp

Test model:
2. python train_sum.py --mode=test --gpus=0 --batch_size=4 \
--tau=2.0 --notes=tmp

"""
import os
from collections import defaultdict
from models.summarizer import Summarizer

from project_settings import HParams, SAVED_MODELS_DIR, \
    EDOC_ID, RESERVED_TOKENS, WORD2VEC_PATH, EDOC_TOK, DatasetConfig, OUTPUTS_EVAL_DIR
    
from utils import create_argparse_and_update_hp, save_run_data, update_moving_avg, sync_run_data_to_bigstore, save_file
from models.nn_utils import setup_gpus
from pretrain_classifier import TextClassifier

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
from data_loaders.yelp_dataset import YelpDataset
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




if __name__ == '__main__':
    # Get hyperparams
    hp = HParams()
    hp, run_name, parser = create_argparse_and_update_hp(hp)

    parser.add_argument('--dataset', default='hotel_mask_asp_1',
                        help='yelp,amazon,hotel,hote_mask')

    parser.add_argument('--test_on_another_dataset', default=None,
                        help='yelp,amazon,hotel,hote_mask')


    parser.add_argument('--az_cat', default=None,
                        help='"Movies_and_TV" or "Electronics"'
                             'Only train on one category')

    parser.add_argument('--save_model_basedir', default=os.path.join(SAVED_MODELS_DIR, 'sum', '{}', '{}'),
                        help="Base directory to save different runs' checkpoints to")
    parser.add_argument('--save_model_fn', default='sum',
                        help="Model filename to save")
    parser.add_argument('--bs_dir', default='',
                        help="Subdirectory on bigstore to save to,"
                             "i.e. <bigstore-bucket>/checkpoints/sum/mlstm/yelp/<bs-dir>/<name-of-experiment>/"
                             "This way related experiments can be grouped together")

    parser.add_argument('--load_lm', default=None,
                        help="Path to pretrained language model")
    parser.add_argument('--load_clf', default=None,
                        help="Path to pretrained classifier")
    parser.add_argument('--load_autoenc', default=None,
                        help="Path to pretrained document autoencoder")
    parser.add_argument('--load_discrim', default='',
                        help="Path to discriminator")

    parser.add_argument('--mode', default='train',
                        help="train or test")
    parser.add_argument('--load_test_sum', default=None,
                        help="Path to trained model, run on test set")
    parser.add_argument('--show_figs', action='store_true')
    parser.add_argument('--test_group_ratings', action='store_true',
                        help='Run on subset of test set, grouping together reviews by rating.'
                             'This is so we can show examples of summaries on the same store'
                             'when given different reviews.')

    parser.add_argument('--print_every_nbatches', default=1,
                        help="Print stats every n batches")
    parser.add_argument('--gpus', default='0',
                        help="CUDA visible devices, e.g. 2,3")
    parser.add_argument('--no_bigstore', action='store_true',
                        help="Do not sync results to bigstore")

    parser.add_argument('--cpu', default=False,
                        help="if want to run on cpu, set --cpu=True")

    parser.add_argument('--skip_clf', default=False,
                        help="set True if want to skip test/train clf")

    parser.add_argument('--lr', default=None, #1e-4
                        help="learning rate")
    parser.add_argument('--len_loss',default=False,
                        help="include length diff loss between input reviews and generated summries")

    parser.add_argument('--len_loss_coef',default=False,
                        help="set coefficient of lenth loss ,effect similar to stepsize")

    parser.add_argument('--sum_combine',default=False,
                        help="mean,ff,gru")

    parser.add_argument('--sscoef',default=False,
                        help="add a coef to summ_shortness")

    parser.add_argument('--true_summ',default=False,
                        help="set True if calculate rouge w.r.t  true summary")



    opt = parser.parse_args()
    #opt:Namespace(autoenc_docs=None, autoenc_docs_tie_dec=None, autoenc_only=None, az_cat=None, batch_size=2, bs_dir='', clf_clip=None, clf_lr=None, clf_mse=None, clf_onehot=None, cnn_dropout=None, cnn_filter_sizes=None, cnn_n_feat_maps=None, combine_encs=None, combine_encs_gru_bi=None, combine_encs_gru_dropout=None, combine_encs_gru_nlayers=None, combine_tie_hc=None, concat_docs=None, cos_honly=None, cos_wgt=None, cpu='1', cycle_loss=None, dataset='hotel_mask_asp_1', debug=None, decay_interval_size=None, decay_tau=None, decay_tau_alpha=None, decay_tau_method=None, discrim_clip=None, discrim_lr=None, discrim_model=None, discrim_onehot=None, docs_attn=None, docs_attn_hidden_size=None, docs_attn_learn_alpha=None, early_cycle=None, emb_size=None, extract_loss=None, freeze_embed=None, g_eps=None, gpus='0', hidden_size=None, len_loss='True', len_loss_coef='1e2', length_loss=None, length_loss_coef=None, lm_clip=None, lm_lr=None, lm_lr_decay=None, lm_lr_decay_method=None, lm_seq_len=None, load_ae_freeze=None, load_autoenc=None, load_clf=None, load_discrim='', load_lm=None, load_test_sum=None, lr=None, lstm_dropout=None, lstm_layers=None, lstm_ln=None, max_nepochs=None, min_tau=None, mode='train', model_type=None, n_docs=3, n_docs_max=None, n_docs_min=None, no_bigstore=False, noam_warmup=None, notes='pycharm_debug', optim=None, print_every_nbatches=1, remove_stopwords=None, save_model_basedir='checkpoints/sum/{}/{}', save_model_fn='sum', seed=None, show_figs=False, skip_clf='1', sum_clf=None, sum_clf_lr=None, sum_clip=None, sum_combine='ff', sum_cycle=None, sum_discrim=None, sum_label_smooth=None, sum_label_smooth_val=None, sum_lr=None, tau=None, test_group_ratings=False, test_on_another_dataset=None, tie_enc=None, track_ppl=None, train_subset=None, tsfr_blocks=None, tsfr_dropout=None, tsfr_ff_size=None, tsfr_label_smooth=None, tsfr_nheads=None, tsfr_tie_embs=None, use_stemmer=None, wgan_lam=None)
    # majority of opt.items are none


    # Hardcoded at the moment
    opt.no_bigstore = True

    setup_gpus(opt.gpus, hp.seed)

    if opt.skip_clf:
        hp.sum_clf=False
    if opt.lr:
        hp.sum_lr=float(opt.lr)
    if opt.len_loss:
        hp.length_loss=True
    if opt.len_loss_coef:
        hp.length_loss_coef=float(opt.len_loss_coef)
    if opt.sum_combine:
        hp.combine_encs = opt.sum_combine
    if opt.sscoef:
        hp.summ_short_coef = float(opt.sscoef)
    if opt.true_summ:
        hp.true_summary = opt.true_summ


    # Set some default paths. It's dataset dependent, which is why we do it here, as dataset is also a
    # command line argument
    ds_conf = DatasetConfig(opt.dataset)
    if opt.load_lm is None:
        opt.load_lm = ds_conf.lm_path
    if opt.load_clf is None:
        opt.load_clf = ds_conf.clf_path
    if opt.load_autoenc is None:
        opt.load_autoenc = ds_conf.autoenc_path
    if opt.load_test_sum is None:
        opt.load_test_sum = ds_conf.sum_path

    # Run
    if opt.mode == 'train':
        # create directory to store results and save run info
        save_dir = os.path.join(opt.save_model_basedir.format(hp.model_type, opt.dataset), run_name)
        save_run_data(save_dir, hp=hp)
        if (not hp.debug) and (not opt.no_bigstore):
            sync_run_data_to_bigstore(save_dir, exp_sub_dir=opt.bs_dir, method='cp')


        summarizer = Summarizer(hp, opt, save_dir)

        summarizer.train()
    elif opt.mode == 'test':
        # Get directory model was saved in. Will be used to save tensorboard test results to
        save_dir = os.path.dirname(opt.load_test_sum)

        # Run
        opt.no_bigstore = True
        if len(hp.notes) == 0:
            raise ValueError('The --notes flag is used for directory name of outputs'
                             ' (e.g. outputs/eval/yelp/n_docs_8/unsup_<notes>). '
                             'Pass in something identifying about this model.')

        summarizer = Summarizer(hp, opt, save_dir)
        summarizer.test()
