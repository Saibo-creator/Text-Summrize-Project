# summarization.py

"""
Unsupervised summarization model
"""
import pdb
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nn_utils import move_to_cuda, calc_clf_acc, convert_to_onehot, LabelSmoothing
from project_settings import PAD_ID, EDOC_ID


#defined only one Class here: SummarizationModel(nn.Module)


class SummarizationModel(nn.Module):

    def __init__(self, docs_enc, docs_autodec,
                 combine_encs_h_net, combine_encs_c_net, summ_dec,
                 summ_enc, docs_dec,
                 discrim_model, clf_model,
                 fixed_lm,
                 hp, dataset):
        super(SummarizationModel, self).__init__()

        self.docs_enc = docs_enc #StackedLSTMEncoder
        self.docs_autodec = docs_autodec #StackedLSTMDecoder
        self.combine_encs_h_net = combine_encs_h_net
        self.combine_encs_c_net = combine_encs_c_net
        self.summ_dec = summ_dec #StackedLSTMDecoder
        self.summ_enc = summ_enc #StackedLSTMEncoder
        self.docs_dec = docs_dec #StackedLSTMDecoder
        self.discrim_model = discrim_model
        self.clf_model = clf_model
        self.fixed_lm = fixed_lm

        self.hp = hp
        self.dataset = dataset

        if self.hp.sum_label_smooth:#False
            self.rec_crit = LabelSmoothing(size=self.dataset.subwordenc.vocab_size,
                                           smoothing=self.hp.sum_label_smooth_val)
        else:
            self.rec_crit = nn.NLLLoss(ignore_index=PAD_ID)
        self.cos_crit = nn.CosineSimilarity(dim=1)
        self.clf_crit = nn.CrossEntropyLoss()

        self.stats = {}

    def forward(self, docs_ids, labels,
                cycle_tgt_ids=None,
                extract_summ_ids=None,
                tau=None,gold_summaries_ids=None,
                adv_step=None, real_ids=None,
                minibatch_idx=None, print_every_nbatches=None,
                tb_writer=None, tb_step=None,
                wass_loss=None, grad_pen_loss=None, adv_gen_loss=None, clf_loss=None, clf_acc=None, clf_avg_diff=None):
        """
        Args:
            docs_ids: [batch, max_len (concatenated reviews)] when concat_docs=True 
                      [batch, n_docs, max_len] when concat_docs=False (present case)
            labels: [batch]
                - ratings for classification
            cycle_tgt_ids: [batch, n_docs, seq_len]
            extract_summ_ids: [batch, max_sum_len]
                - summaries from extractive model
            gold_summaries_ids: having the same shape as docs_ids, used only in the supervised mode to compute
            gold_summary_loss
            tau: float
                Passed in instead of using self.hp.tau because tau may be
                obtained from a StepAnnealer if there is a scheduled decay.
            adv_step: str ('discrim' or 'gen')
                - whether to compute discriminator step (with detach to only train Discriminator), or
                just the generator step (pass in generated summaries and return .mean())
            real_ids: [batch, max_rev_len]
                - reviews used for Discriminator

            minibatch_idx: int (how many minibatches in current epoch)
            print_every_nbatches: int
            tb_writer: Tensorboard SummaryWriter
            tb_step: int (used for writer)

            The remaining are 0-D float Tensors to handle an edge case where the summary is too short for the
            TextCNN. The current average is passed in.

        Returns:
            stats: dict (str to 0-D tensors)
                - contains losses
            summ_texts: list of strs
        """
        batch_size = docs_ids.size(0)
        n_docs = docs_ids.size(1)
        max_len = docs_ids.size(2)

        ##########################################################
        # CALCULATE AVERAGE LENGTH OF INPUT REVIEWS
        ##########################################################
        docs_ids_unstacked=torch.squeeze(docs_ids.view(1, -1, max_len)) # [batchsize*ndocs:max_len]
        input_txts=[self.dataset.subwordenc.decode(x_i).replace('<pad>','') for x_i in docs_ids_unstacked]#len(input_txts)=batchsize*ndocs
        input_txts_lengths=torch.tensor([len(txt.split(' ')) for txt in input_txts], dtype=torch.float).view(batch_size,n_docs) #  [batchsize*ndocs:lenth_tokens] ,length_tokens is always <150
        # print(input_txts_lengths)
        # print('\n')
        input_txts_mean_lengths=torch.mean(input_txts_lengths,1) # get mean length per batch; len(input_txts_mean_lengths)=batch_size

        # print(input_txts_mean_lengths)


        ##########################################################
        # ENCODE DOCUMENTS
        ##########################################################
        # Print a review if we're autoencoding or using cycle reconstruction loss so that we can
        # check how well the reconstruction is
        if self.hp.autoenc_docs or (self.hp.cycle_loss == 'rec'):
            if minibatch_idx % print_every_nbatches == 0:
                if docs_ids.get_device() == 0:
                    orig_rev_text = self.dataset.subwordenc.decode(docs_ids[0][0])
                    print('\n', '-' * 100)
                    print('ORIGINAL REVIEW: ', orig_rev_text.encode('utf8'))
                    print('-' * 100)
                    if tb_writer:
                        tb_writer.add_text('auto_or_rec/orig_review', orig_rev_text, tb_step)

        if not self.hp.concat_docs:
            n_docs = docs_ids.size(1)  # : need to get data loader to choose items with same n_docs
            docs_ids = docs_ids.view(-1, docs_ids.size(-1))  # [batch * n_docs, len]

        h_init, c_init = self.docs_enc.rnn.state0(docs_ids.size(0))# batch * n_docs,  
        # def state0(self, batch_size):
        # h_0 = torch.zeros(batch_size, self.num_layers, self.hidden_size, requires_grad=False)
        # c_0 = torch.zeros(batch_size, self.num_layers, self.hidden_size, requires_grad=False)
        # return h_0, c_0
        
        h_init, c_init = move_to_cuda(h_init), move_to_cuda(c_init)
        hiddens, cells, outputs = self.docs_enc(docs_ids, h_init, c_init)
        docs_enc_h, docs_enc_c = hiddens[-1], cells[-1]  # [_, n_layers, hidden]

        ##########################################################
        # Demonstrate autoencoder behavior
        ##########################################################

        # Autoencoder - decode into original reviews
        if self.hp.autoenc_docs:
            assert (self.hp.concat_docs == False), \
                'Docs must be encoded individually for autoencoder. Set concat_docs=False'
            init_input = torch.LongTensor([EDOC_ID for _ in range(docs_enc_h.size(0))])  # batch * n_docs
            init_input = move_to_cuda(init_input)
            docs_autodec_probs, _, docs_autodec_texts, extra = self.docs_autodec(docs_enc_h, docs_enc_c, init_input,
                                                                             targets=docs_ids,
                                                                             eos_id=EDOC_ID, non_pad_prob_val=1e-14,
                                                                             softmax_method='softmax',
                                                                             sample_method='greedy',
                                                                             tau=tau,
                                                                             subwordenc=self.dataset.subwordenc)


            shortness=torch.squeeze(extra['shortness']).reshape(-1,batch_size)
            input_shortness=torch.mean(shortness, dim=0)

            docs_autodec_logprobs = torch.log(docs_autodec_probs)
            # print(docs_autodec_logprobs.shape) #torch.Size([32, 179, 31688]) 32=n_docs(8)*batch_size(4),31688= vocab size, 179 = token number
            # print((docs_ids.view(-1)).shape)# torch.Size([5728])
            autoenc_loss = 100*self.rec_crit(docs_autodec_logprobs.view(-1, docs_autodec_logprobs.size(-1)),
                                         docs_ids.view(-1))#if torch.view(-1)，then the original tensor will become dimension 1。
            # print(docs_autodec_probs)
            # print(docs_ids.view(-1))
            if self.hp.sum_label_smooth:
                autoenc_loss /= (docs_ids != move_to_cuda(torch.tensor(PAD_ID))).sum().float()
            self.stats['autoenc_loss'] = autoenc_loss

            if minibatch_idx % print_every_nbatches == 0:
                if docs_ids.get_device() == 0:
                    dec_text = docs_autodec_texts[0]
                    print('DECODED REVIEW: ', dec_text.encode('utf8'))
                    print('-' * 100, '\n')
                    if tb_writer:
                        tb_writer.add_text('auto_or_rec/auto_dec_review', dec_text, tb_step)

            # Early return if we're only computing auto-encoder (don't have to decode into summaries)
            if self.hp.autoenc_only:
                dummy_summ_texts = ['a dummy review' for _ in range(batch_size)]
                return self.stats, dummy_summ_texts


        ##########################################################
        # Meansum 
        ##########################################################

        # Decode into summary
        if not self.hp.concat_docs:
            _, n_layers, hidden_size = docs_enc_h.size()
            docs_enc_h = docs_enc_h.view(batch_size, n_docs, n_layers, hidden_size)
            docs_enc_c = docs_enc_c.view(batch_size, n_docs, n_layers, hidden_size)
            if self.hp.combine_encs == 'mean':
                docs_enc_h_comb = docs_enc_h.mean(dim=1)
                docs_enc_c_comb = docs_enc_c.mean(dim=1)
            elif self.hp.combine_encs == 'ff':
                docs_enc_h_comb = docs_enc_h.transpose(1, 2).view(batch_size, n_layers, -1)
                # [batch, n_layers, n_docs * hidden]
                docs_enc_c_comb = docs_enc_c.transpose(1, 2).view(batch_size, n_layers, -1)
                docs_enc_h_comb = self.combine_encs_h_net(docs_enc_h_comb)  # [batch, n_layers, hidden]
                docs_enc_c_comb = self.combine_encs_c_net(docs_enc_c_comb)
            elif self.hp.combine_encs == 'gru':
                n_directions = 2 if self.hp.combine_encs_gru_bi else 1
                init_h = torch.zeros(self.hp.combine_encs_gru_nlayers * n_directions, batch_size, hidden_size)
                init_c = torch.zeros(self.hp.combine_encs_gru_nlayers * n_directions, batch_size, hidden_size)
                init_h = move_to_cuda(init_h)
                init_c = move_to_cuda(init_c)
                docs_enc_h_comb = docs_enc_h.view(batch_size, n_docs * n_layers, hidden_size)
                docs_enc_c_comb = docs_enc_h.view(batch_size, n_docs * n_layers, hidden_size)
                # [batch, n_directions * gru_nlayers, hidden]
                # self.combine_encs_h_net.flatten_parameters()
                # self.combine_encs_c_net.flatten_parameters()
                _, docs_enc_h_comb = self.combine_encs_h_net(docs_enc_h_comb, init_h)
                _, docs_enc_c_comb = self.combine_encs_c_net(docs_enc_c_comb, init_c)
                # [n_directions * gru_nlayers, batch, hidden]
                docs_enc_h_comb = docs_enc_h_comb[-1, :, :].unsqueeze(0).transpose(0,1)  # last layer : last or combine?
                docs_enc_c_comb = docs_enc_c_comb[-1, :, :].unsqueeze(0).transpose(0,1)  # last layer


        softmax_method = 'gumbel'
        sample_method = 'greedy'
        if self.hp.early_cycle:
            softmax_method = 'softmax'
            # sample_method = 'sample'
        init_input = torch.LongTensor([EDOC_ID for _ in range(batch_size)])
        init_input = move_to_cuda(init_input)
        # Backwards compatibility with models trained before dataset refactoring
        # We could use the code_snapshot saved at the time the model was trained, but I've added some
        # useful things (e.g. tracking NLL of summaries)
        tgt_summ_seq_len = self.dataset.conf.review_max_len if hasattr(self.dataset, 'conf') else \
            self.hp.yelp_review_max_len
        summ_probs, _, summ_texts, extra = self.summ_dec(docs_enc_h_comb, docs_enc_c_comb, init_input,
                                                     seq_len=tgt_summ_seq_len, eos_id=EDOC_ID,
                                                     # seq_len=self.dataset.conf.review_max_len, eos_id=EDOC_ID,
                                                     softmax_method=softmax_method, sample_method=sample_method,
                                                     tau=tau, eps=self.hp.g_eps, gumbel_hard=True,non_pad_prob_val=1e-14,
                                                     attend_to_embs=docs_enc_h,
                                                     subwordenc=self.dataset.subwordenc)
        # summ_texts:2 texts
        # summ_prob.shape:torch.Size([2, 180, 23852])


        ##########################################################
        # SUPERVISED
        ##########################################################
        try:
            if self.hp.gold_summ_loss:
                # if not self.hp.autoenc_docs:
                # default_prob = move_to_cuda(torch.zeros(summ_probs.shape)).fill_(1e-14)  # k=1
                # summ_probs = torch.where((summ_probs == 0), default_prob, summ_probs)
                summ_probs+=1e-14
                summ_logprobs = torch.log(summ_probs)
                # print(docs_autodec_logprobs.shape) #torch.Size([32, 179, 31688]) 32=n_docs(8)*batch_size(4),31688= vocab size, 179 = token number
                # print((docs_ids.view(-1)).shape)# torch.Size([5728])
                try:
                    gold_summaries_ids=gold_summaries_ids[:, :tgt_summ_seq_len]
                except Exception as e:
                    print('A possible error here in supervised loss calculation')
                    for i, gold_summary_ids in enumerate(gold_summaries_ids):
                        padded = np.zeros(tgt_summ_seq_len)
                        padded[:len(gold_summary_ids)] = gold_summary_ids
                        gold_summaries_ids[i, :] = padded

                gold_summ_loss = self.rec_crit(summ_logprobs.view(-1, summ_logprobs.size(-1)),
                                             gold_summaries_ids.reshape(-1))  # if torch.view(-1)，then the original tensor will become dimension 1。
                print(gold_summ_loss)
                # print(summ_probs)
                # print(summ_texts)
                # print(gold_summaries_ids.reshape(-1))
                self.stats['gold_summ_loss'] = gold_summ_loss
        except AttributeError as e:
            print("The model loaded model was trained on dataset without supervised mode")

                #TODO
            # else:
            #     raise ValueError('hp.autoenc_docs(unsupervised) not recommanded with gold summary loss(supervised)')






        ##########################################################
        # LENGTH DIFF  LOSS
        ##########################################################
        ###########################  summ_texts_lengths  ###############################
        # [batch, max_summ_len, vocab];  [batch] of str's
        # summ.extra['shortness']
        summ_shortness=extra['shortness']

        try:
            if self.hp.length_loss:
                # length_cos = nn.CosineSimilarity(dim=0, eps=1e-08)
                # length_loss = length_cos(ideal_length,
                #                          summ_texts_lengths )#input_txts_mean_lengths  # 0<=loss<=1( not strictly <1 but summ_text should merely exceed twice the input length in practice)

                #self.stats['length_loss'] = 1e4*move_to_cuda(torch.mean(move_to_cuda(torch.ones(extra['shortness'].shape))-move_to_cuda(extra['shortness'])))#10*[6.4844e-01, 4.2873e-06, 8.9063e-01, 7.0317e-02, 5.4688e-01, 7.1737e-06],
                self.stats['length_loss'] = move_to_cuda(self.hp.length_loss_coef*torch.mean(torch.norm(summ_shortness*self.hp.summ_short_coef-input_shortness)))  # 10*[6.4844e-01, 4.2873e-06, 8.9063e-01, 7.0317e-02, 5.4688e-01, 7.1737e-06], shortness of summ should be small
                print('*' * 20)
                print('length_loss=:', self.stats['length_loss'])
                print('*' * 20)
        except AttributeError  as e:
            pass

        # Compute a cosine similarity loss between the (mean) summary representation that's fed to the
        # summary decoder and each of the original encoded reviews.
        # With this setup, there's no need for the summary encoder or back propagating through the summary.
        if self.hp.early_cycle:
            # Repeat each summary representation n_docs times to match shape of tensor with individual reviews
            docs_enc_h_comb_rep = docs_enc_h_comb.repeat(1, n_docs, 1) \
                .view(batch_size * n_docs, docs_enc_h_comb.size(1), docs_enc_h_comb.size(2))
            docs_enc_c_comb_rep = docs_enc_c_comb.repeat(1, n_docs, 1) \
                .view(batch_size * n_docs, docs_enc_c_comb.size(1), docs_enc_c_comb.size(2))

            loss = -self.cos_crit(docs_enc_h_comb_rep.view(batch_size, -1),
                                  docs_enc_h.view(batch_size, -1).detach()).mean()
            if not self.hp.cos_honly:
                   loss -= self.cos_crit(docs_enc_c_comb_rep.view(batch_size, -1),
                                 docs_enc_c.view(batch_size, -1).detach()).mean()
            self.stats['early_cycle_loss'] = loss * self.hp.cos_wgt

        ##########################################################
        # CYCLE LOSS and / or  EXTRACTIVE SUMMARY LOSS
        ##########################################################

        # Encode summaries
        if self.hp.sum_cycle or self.hp.extract_loss:
            init_h, init_c = self.summ_enc.rnn.state0(batch_size)
            init_h, init_c = move_to_cuda(init_h), move_to_cuda(init_c)
            hiddens, cells, outputs = self.summ_enc(summ_probs, init_h, init_c)
            summ_enc_h, summ_enc_c = hiddens[-1], cells[-1]  # [batch, n_layers, hidden], ''

        # Extractive vs. abstractive summary loss
        if self.hp.extract_loss:
            # Encode extractive summary
            init_h, init_c = self.summ_enc.rnn.state0(batch_size)
            init_h, init_c = move_to_cuda(init_h), move_to_cuda(init_c)
            ext_hiddens, ext_cells, ext_outputs = self.summ_enc(extract_summ_ids, init_h, init_c)
            ext_enc_h, ext_enc_c = ext_hiddens[-1], ext_cells[-1]  # [batch, n_layers, hidden], ''
            loss = -self.cos_crit(summ_enc_h.view(batch_size, -1),
                                  ext_enc_h.view(batch_size, -1).detach()).mean()
            if not self.hp.cos_honly:
                   loss -= self.cos_crit(summ_enc_c.view(batch_size, -1),
                                 ext_enc_c.view(batch_size, -1).detach()).mean()
            self.stats['extract_loss'] = loss

        # Reconstruction or encoder cycle loss
        if self.hp.sum_cycle:
            # Repeat each summary representation n_docs times to match shape of tensor with individual reviews
            summ_enc_h_rep = summ_enc_h.repeat(1, n_docs, 1) \
                .view(batch_size * n_docs, summ_enc_h.size(1), summ_enc_h.size(2))
            summ_enc_c_rep = summ_enc_c.repeat(1, n_docs, 1) \
                .view(batch_size * n_docs, summ_enc_c.size(1), summ_enc_c.size(2))

            if self.hp.cycle_loss == 'enc':#defualt case, compare summary
                assert (self.hp.concat_docs == False), \
                    'Docs must have been encoded individually for autoencoder. Set concat_docs=False'
                # (It's possible to have cycle_loss=enc and concat_docs=False, you just have to als encode them
                # separately. Didn't add that b/c I think I'll always have concat_docs=False from now on)
                # docs_enc_h, docs_enc_c: [batch, n_docs, n_layers, hidden]
                loss = -self.cos_crit(summ_enc_h_rep.view(batch_size, -1),
                                      docs_enc_h.view(batch_size, -1).detach()).mean()
                if not self.hp.cos_honly:
                    loss -= self.cos_crit(summ_enc_c_rep.view(batch_size, -1),
                                     docs_enc_c.view(batch_size, -1).detach()).mean()
                self.stats['cycle_loss'] = loss * self.hp.cos_wgt
            elif self.hp.cycle_loss == 'rec':
                init_input = move_to_cuda(torch.LongTensor([EDOC_ID for _ in range(batch_size * n_docs)]))
                probs, ids, texts, extra = self.docs_dec(summ_enc_h_rep, summ_enc_c_rep, init_input,
                                                         targets=cycle_tgt_ids.view(-1, cycle_tgt_ids.size(-1)),
                                                         eos_id=EDOC_ID, non_pad_prob_val=1e-14,
                                                         softmax_method='softmax', sample_method='sample',
                                                         tau=tau,
                                                         subwordenc=self.dataset.subwordenc)
                vocab_size = probs.size(-1)
                logprobs = torch.log(probs).view(-1, vocab_size)
                loss = self.rec_crit(logprobs, cycle_tgt_ids.view(-1))
                if self.hp.sum_label_smooth:
                    loss /= (cycle_tgt_ids != move_to_cuda(torch.tensor(PAD_ID))).sum().float()
                self.stats['cycle_loss'] = loss * self.hp.cos_wgt

                if minibatch_idx % print_every_nbatches == 0:
                    if docs_ids.get_device() == 0:
                        print('DECODED REVIEW: ', texts[0].encode('utf8'))
                        print('-' * 100, '\n')
                        if tb_writer:
                            tb_writer.add_text('auto_or_rec/rec_review', texts[0], tb_step)





        ##########################################################
        # CLASSIFIER
        ##########################################################

        if self.hp.sum_clf:
            if summ_probs.size(1) < 5:  # conv filters are 3,4,5
                print('Summary length is less than 5... skipping classification model because it uses a CNN '
                      'with a convolution kernel of size 5')
            else:
                logits = self.clf_model(summ_probs.long())
                clf_loss = self.clf_crit(logits, labels)

                _, indices = torch.max(logits, dim=1)
                clf_avg_diff = (labels - indices).float().mean()
                clf_acc = torch.eq(indices, labels).sum().float() / batch_size

            self.stats.update({'clf_loss': clf_loss, 'clf_acc': clf_acc, 'clf_avg_diff': clf_avg_diff})

        return self.stats, summ_texts
