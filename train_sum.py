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




if __name__ == '__main__':
    # Get hyperparams
    hp = HParams()
    hp, run_name, parser = create_argparse_and_update_hp(hp)

    parser.add_argument('--dataset', default='hotel',
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


    opt = parser.parse_args()

    # Hardcoded at the moment
    opt.no_bigstore = True

    setup_gpus(opt.gpus, hp.seed)

    if opt.skip_clf:
        hp.sum_clf=False

    if hp.sum_clf==False:
        raise ValueError('A very very specific bad thing happened')

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
