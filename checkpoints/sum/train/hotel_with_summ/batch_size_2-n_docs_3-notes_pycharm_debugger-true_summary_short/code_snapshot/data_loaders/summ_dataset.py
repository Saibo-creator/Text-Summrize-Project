# summ_dataset.py

"""
Base class for summarization datasets
"""

import numpy as np
import torch

from models.nn_utils import move_to_cuda
from project_settings import EDOC_TOK

# EDOC_TOK='</DOC>'

class SummDataset(object):
    def __init__(self):
        self.name = None  # yelp, amazon; set in dataset-specific constructor
        self.conf = None  # dataset-specific config
        self.subwordenc = None

    @staticmethod
    def concat_docs(docs, edok_token=True):
        """
        Concatenate a list of strs by joining them with the end of doc token
        """
        if edok_token:
            return ' {} '.format(EDOC_TOK).join(docs)
        else:
            return ' '.join(docs)

    @staticmethod
    def split_docs(docs):
        """
        Return a list of strs (docs is one str)
        """
        return docs.split(' {} '.format(EDOC_TOK))


class SummReviewDataset(SummDataset):
    def __init__(self):
        super(SummReviewDataset, self).__init__()

    def prepare_batch(self,
                      texts_batch, ratings_batch,g_summ_loss=False, gold_summary_batch=None,
                      global_prepend_id=None, global_append_id=None,
                      doc_prepend_id=None, doc_append_id=None):
        #TODO,
        #only doc_append_id=EDOC_ID
        """
        Prepare batch of texts and labels from DataLoader as input into nn.

        Args:
            g_summ_loss: list of string, if True, prepare batch of gold summries as well
            texts_batch: list of str's
                - length batch_size
                - each str is a concatenated group of document
            ratings_batch: list of size-1 LongTensor's
                - length batch_size

            global_prepend_id: int (prepend GO)
            global_append_id: int (append EOS)
            doc_prepend_id: int (prepend DOC before start of each review)
            doc_append_id: int (append /DOC after end of each review)

        Returns: (cuda)
            x: LongTensor of size [batch_size, max_seq_len]
            lengths: LongTensor (length of each text in subtokens)
            labels: LongTensor of size [batch_size]
        """
        # Original ratings go from 1-5
        labels_of_batch = [rating - 1 for rating in ratings_batch]

        batch = []
        subtoken_ids_list=[]
        seq_len_list=[]
        summ_subtoken_ids_list=[]

        for i, text in enumerate(texts_batch):
            # Split apart by docs and potentially add delimiters
            docs = SummDataset.split_docs(text)  # list of strs  or list containing only one str if text contains only one review
            if doc_prepend_id or doc_append_id:
                docs_ids = [self.subwordenc.encode(doc) for doc in docs]
                if doc_prepend_id:
                    for doc_ids in docs_ids:
                        doc_ids.insert(0, doc_prepend_id)
                if doc_append_id:
                    for doc_ids in docs_ids:
                        doc_ids.append(doc_append_id)
                docs_ids = [id for doc_ids in docs_ids for id in doc_ids]  # flatten, merge 3 reviews into one docs_ids
                subtoken_ids = docs_ids
            else:
                subtoken_ids = self.subwordenc.encode(' '.join(docs))

            # Add start and end token for concatenated set of documents
            if global_prepend_id:
                subtoken_ids.insert(0, global_prepend_id)
            if global_append_id:
                subtoken_ids.append(global_append_id)

            seq_len = len(subtoken_ids) #number of tokens the review contains
            subtoken_ids_list.append(subtoken_ids)
            seq_len_list.append(seq_len)


        if g_summ_loss:
            # map gold summries to a collection of token index
            for i, text in enumerate(gold_summary_batch):
                summs = gold_summary_batch
                # Split apart by summs and potentially add delimiters
                if doc_prepend_id or doc_append_id:
                    summs_ids = [self.subwordenc.encode(doc) for doc in summs]
                    if doc_prepend_id:
                        for doc_ids in summs_ids:
                            doc_ids.insert(0, doc_prepend_id)
                    if doc_append_id:
                        for doc_ids in summs_ids:
                            doc_ids.append(doc_append_id)
                    summs_ids = [id for doc_ids in summs_ids for id in doc_ids]  # flatten
                    summ_subtoken_ids = summs_ids
                else:
                    raise NotImplementedError('this case is not considered, please add code')
                summ_subtoken_ids_list.append(summ_subtoken_ids)

        for i in range(len(texts_batch)):
            if g_summ_loss:
                batch.append((subtoken_ids_list[i], seq_len_list[i], labels_of_batch[i], summ_subtoken_ids_list[i]))
                #TODO
            else:
                batch.append((subtoken_ids_list[i], seq_len_list[i], labels_of_batch[i]))

        if g_summ_loss:#TODO
            texts_ids, lengths, labels, gold_summaries_ids = zip(*batch)
        else:
            texts_ids, lengths, labels, = zip(*batch)
        lengths = torch.LongTensor(lengths)
        labels = torch.stack(labels)

        # Pad each text
        max_seq_len = max(lengths)
        batch_size = len(batch)
        x = np.zeros((batch_size, max_seq_len))
        for i, text_ids in enumerate(texts_ids):
            padded = np.zeros(max_seq_len)
            padded[:len(text_ids)] = text_ids
            x[i, :] = padded
        x = torch.from_numpy(x.astype(int))
        x = move_to_cuda(x)

        x_ = np.zeros((batch_size, max_seq_len))
        if g_summ_loss:
            for i, gold_summary_ids in enumerate(gold_summaries_ids):
                padded = np.zeros(max_seq_len)
                padded[:len(gold_summary_ids)] = gold_summary_ids
                x_[i, :] = padded
        x_ = torch.from_numpy(x_.astype(int))
        x_ = move_to_cuda(x_)


        lengths = move_to_cuda(lengths)
        labels = move_to_cuda(labels)


        return x, lengths, labels, x_
