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


text="Easy access. Large clean room. Quiet and the beds were comfortable. Have stayed here twice and will again. </DOC> Was not impressed by this property, it's across from the Tacoma Dome. Seems to be located in an uninhabited area. Our room over looked the railroad tracks and a large parking structure. I heard a train whistle a couple of times. The room needed some maintenance, hair dryer was burned out, toilet seat was falling off, and they put a floor mounted door stop right where you step out of the shower. The exhaust fan in the bathroom was very noisy, if I wasn't so tired it probably would have kept me awake. The bed was okay, but the small pillows have seen a lot of use and only one thin blanket. The breakfast was the best part of the experience. We've stayed at better Best Westerns for a lower cost than this one. </DOC> This Best Western Plus hotel is very nice. It has a great view of the Tacoma Dome and LeMay, America's Car Museum and is within walking distance of both venues. The room was comfortable and quiet. The glass elevator was fun to use. The breakfast was very good. The location makes it easy to get to restaurants and other things in the area. The parking is in a secure parking structure which is locked at night. </DOC> This hotel is located off the main freeway...so location for us was good. The hotel was ok and clean and had friendly staff too when checking in. But the room was so small even though we booked for 4 (2 adults and 2 children). It had a queen bed and a pull out sofa. The 2 kids and I slept in the main bed and my husband slept in the pullout sofa bed as it would definitely not sleep 2. The mattress on the sofa bed needs to be replaced. Room was clean but not the appropriate size for a family. Breakfast was great and even have a staff member to make fresh omelettes! That ended our experience on a high note! </DOC> We stayed here while attending a show at the Tacoma Dome and enjoyed our stay. There are not a lot of amenities here but it was clean and comfortable. The continental breakfast had a wide selection of good food. </DOC> We highly recommend the Best Western Plus-Tacoma Dome. We were afraid that there would be traffic noise because of the proximity to I-5, but the hotel was quiet and we couldn't hear the road noise at all. Our room was ample and very clean with the usual amenities and a comfortable bed. The breakfast had a variety of choices and was very good. We would definitely stay again. </DOC> This is a very easy to reach Best Western. Being right next to the Tacoma Dome, you never will get lost finding it. Also with LaMay's Automotive Museum also within walking distance it makes this a great and quiet place to stay. Also, Brown and Hailey's is 2 blocks down the road. The Museum district is a few blocks down the road and the Transit Center (rail etc) is a few blocks away. Made to order Omelets was a great treat too. Location, quiet and good location. </DOC> Not up to the usual BW standard we have come to expect. We kept getting the interconnecting door bumped and thought what inconsiderate people next door until a huge dog started barking just as we went to be. We complained and the reception staff said they would ask the owners to keep the dog quiet. However the dog barked early again the next morning. We have nothing against pets but the hotel should confine them to a separate are of the hotel. Breakfast was awful, eggs like old boot leather, no bacon, juice reconstituted stuff, very bitter. Also there was a cat wandering around the dining are when we went down and then it was on the tables. Overall a very overpriced disappointing stay and would not go there again."


def main():
	print(SummDataset.split_docs(text))
	return 0

