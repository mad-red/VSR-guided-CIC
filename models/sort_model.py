import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import defaultdict

from .transformer_modules import TransformerEmbedding
from .transformer_modules import LabelSmoothingKLDivLoss
from .sort_modules import TransformerEncoder, TransformerDecoder

class S_SSP(nn.Module):

    def __init__(self, pos_enc=False, add_fc=True, dataset='coco'):
        super(S_SSP, self).__init__()
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        if dataset == 'coco':
            self._verb_size = 2662
        else:
            self._verb_size = 2926
        self.encoder_layers = 3
        self.decoder_layers = 3
        self.max_len = 10
        self.beam_size = 1
        self.hidden_size = 512
        self.embed_size = 512
        self.prepare(pos_enc=pos_enc, add_fc=add_fc)
        self.initialize_parameters()

    # in init, call function prepare and initialize_parameters
    def prepare(self, pos_enc=False, add_fc=False):
        """Define the modules
        """
        # Embedding layers
        self.sr_embed_layer = TransformerEmbedding(26, self.embed_size)
        self.v_embed_layer = TransformerEmbedding(self._verb_size+1, self.embed_size)
        # Encoder
        self.encoder = TransformerEncoder(self.sr_embed_layer, self.v_embed_layer, self.hidden_size, self.encoder_layers, pos_enc=pos_enc, add_fc=add_fc)
        # Decoder
        self.decoder = TransformerDecoder(self.sr_embed_layer, self.hidden_size, self.decoder_layers, skip_connect=False)
        # sr probability estimator
        self.expander_nn = nn.Linear(self.hidden_size, 26)
        self.label_smooth = LabelSmoothingKLDivLoss(0.1, 26) # not to ignore 0
        self.softmax = nn.Softmax(dim = -1)

    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_shard_loss(self, decoder_states, tgt_seq, tgt_mask, denominator=None, ignore_first_token=False):
        if denominator is None:
            if ignore_first_token:
                denom = tgt_mask[:, 1:].sum()
            else:
                denom = tgt_mask.sum()
        else:
            denom = denominator
        # Compute loss
        logits = self.expander_nn(decoder_states)
        loss = self.compute_loss(logits, tgt_seq, tgt_mask, denominator=denom, ignore_first_token=ignore_first_token)
        return loss
    
    def compute_loss(self, logits, tgt_seq, tgt_mask, denominator=None, ignore_first_token=False):
        B, T, _ = logits.shape
        logits = F.log_softmax(logits, dim=2)
        flat_logits = logits.contiguous().view(B * T, 26)
        if ignore_first_token:
            tgt_seq = tgt_seq[:, 1:]
            tgt_mask = tgt_mask[:, 1:]
        flat_targets = tgt_seq.contiguous().view(B * T)
        flat_mask = tgt_mask.contiguous().view(B * T)
        if denominator is None:
            denominator = flat_mask.sum()
        loss = self.label_smooth(flat_logits, flat_targets, flat_mask) / denominator
        return loss

    def forward(self, this_verb, det_seqs_sr, gt_seqs_sr):
        this_verb = (this_verb % 10000).long()  # (1, 1)
        det_seqs_sr = det_seqs_sr.long()  # (1, fixed_len)
        gt_seqs_sr = gt_seqs_sr.long()

        sr_mask = torch.ne(gt_seqs_sr, 0).float()

        # process decoder input
        decoder_input = gt_seqs_sr.new_zeros((gt_seqs_sr.shape[0], gt_seqs_sr.shape[1] + 2))
        decoder_input[:, 1:-1] = gt_seqs_sr  # 第一行置为0，0表示bos
        decoder_mask = sr_mask.new_zeros((gt_seqs_sr.shape[0], gt_seqs_sr.shape[1] + 2))
        decoder_mask[:, 1:-1] = sr_mask
        decoder_mask[:, 0] = 1  # 第一行置为1

        # ----------- Encoder -------------#
        prior_states = self.encoder(this_verb, det_seqs_sr)
        # ----------- Decoder -------------#
        decoder_states = self.decoder(decoder_input[:, :-1], decoder_mask[:, :-1], prior_states, None)
        # ----------- Compute losses ------------------------#
        loss = self.compute_shard_loss(decoder_states, decoder_input[:, 1:], decoder_mask[:, :-1], ignore_first_token=False)

        if torch.isnan(loss) or torch.isinf(loss):
            import pdb;pdb.set_trace()
        return loss

    def generate(self, this_verb, det_seqs_sr, mode='normal'):
        """autoregressively generate the semantic role
        """
        this_verb = (this_verb % 10000).long()
        det_seqs_sr = det_seqs_sr.long()
        if self.beam_size != 1:
            return self.sample_beam(this_verb, det_seqs_sr)
        sr_remain = torch.ne(det_seqs_sr, 0) # (batch_size, fixed_len), denote the sr remaining to be found
        batch_size = this_verb.shape[0]

        # ----------- Encoder -------------#
        prior_states = self.encoder(this_verb, det_seqs_sr)

        # Run decoder to predict the semantic role
        pred = det_seqs_sr.new_zeros((batch_size, self.max_len), dtype=torch.long)
        seqLogprobs = det_seqs_sr.new_zeros(batch_size, self.max_len)
        if mode == 'normal':
            for t in range(self.max_len+1):
                if t == 0: # input <bos>
                    it = det_seqs_sr.new_zeros((batch_size, 1)).long()
                    x = it
                else:
                    x = torch.cat((x, it), 1) # (b_s, seq_len)
                decoder_states = self.decoder(x, None, prior_states, None) # decoder_states size(batch_size, seq_len, hidden_size)
                logits = self.expander_nn(decoder_states[:, -1]) # hidden_size -> vocab_size+1 | size: (b_s, vob_s)
                logprobs = F.log_softmax(logits, dim=-1) # (b_s, vob_s)
                sampleLogprobs, it = torch.max(logprobs, -1) # (b_s, vob_s) --> (b_s)
                it = it.long()

                # skip if we achieve maximum length
                if t == self.max_len: 
                    break
                # stop when all finished
                if t == 0:
                    unfinished = it > 0 # (b_s)
                else:
                    unfinished = unfinished * (it > 0)
                it = it * unfinished.type_as(it)
                pred[:, t] = it # (b_s)
                seqLogprobs[:, t] = sampleLogprobs
                it = it.unsqueeze(1)
                # quit loop if all sequences have finished
                if unfinished.sum() == 0:
                    break
        else:
            for t in range(self.max_len+1):
                remain_len = torch.sum(sr_remain[0].long(), -1) # (b_s) the remain sr len
                if remain_len.item() == 0:
                    break
                if t == 0: # input <bos>
                    it = det_seqs_sr.new_zeros((batch_size, 1)).long()
                    x = it
                else:
                    x = torch.cat((x, it), 1) # (b_s, seq_len)
                decoder_states = self.decoder(x, None, prior_states, None) # decoder_states size(batch_size, seq_len, hidden_size)
                logits = self.expander_nn(decoder_states[:, -1]) # hidden_size -> vocab_size+1 | size: (b_s, vob_s)
                logprobs = F.log_softmax(logits, dim=-1) # (b_s, vob_s)

                seqs_sr = torch.masked_select(det_seqs_sr, sr_remain) # remain_len, 存着semantic role
                sr_logprobs = torch.index_select(logprobs, -1, seqs_sr)
                sampleLogprobs, it = torch.max(sr_logprobs, -1) # it只是在剩余中的顺序，而不是vocab中的顺序

                k = 0
                for i in range(sr_remain.shape[1]):
                    if sr_remain[0][i] == 1:
                        if it.squeeze().item() == k:
                            it = det_seqs_sr[0][i]
                            sr_remain[0][i] = 0
                            break
                        k += 1

                # sampleLogprobs, it = torch.max(logprobs, -1) # (b_s, vob_s) --> (b_s)
                it = it.long().unsqueeze(0)

                pred[:, t] = it # (b_s)
                seqLogprobs[:, t] = sampleLogprobs.unsqueeze(0)
                it = it.unsqueeze(1)

        return pred, seqLogprobs, None

    def sample_beam(self, verb, det_seqs_sr):
        beam_size = self.beam_size
        batch_size = verb.size(0)
        device = verb.device

        seq = torch.LongTensor(self.max_len, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.max_len, batch_size)

        prior_states = self.prior_encoder(verb, det_seqs_sr)

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):  # for each data
            prior_state = prior_states[k, :, :]  # (feat_len, feat_size)
            prior_state = prior_state.expand(beam_size, prior_state.size(0), prior_state.size(1)) # (beam_size, feat_len, feat_size)

            # the first input, <bos>
            it = verb.new_zeros((beam_size, 1)).long()
            decoder_states = self.decoder(it, None, prior_state, None) # decoder_states: (beam_size, 1, hidden_size)
            logprobs = F.log_softmax(self.expander_nn(decoder_states.squeeze()), dim=-1)  # (beam_size, vocab_s)

            # other inputs
            self.done_beams[k] = self.beam_search(logprobs, prior_states[k])  # beam_search()
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']

        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1), None  # seq/seqLogprobs: (batch_size, seq_len)

    def beam_search(self, logprobs, prior_state): # logprobs:(beam_size, vocab_s), prior_state:(36, 2048)
        # args are the miscelleous inputs to the core in addition to embedded word and state
        # kwargs only accept opt

        def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum):
            #INPUTS:
            #logprobsf: probabilities augmented after diversity  (beam_size, vocab_s)
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams  (seq_len, beam_size)
            #beam_seq_logprobs: tensor contanining the beam logprobs  (seq_len, beam_size)
            #beam_logprobs_sum: tensor contanining joint logprobs  (beam_size)
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions
            #beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            #beam_logprobs_sum : joint log-probability of each beam
            ys, ix = torch.sort(logprobsf, 1, True) # descending sort, ys:(beam_size, vocab_s), ix:(beam_size, vocab_s)
            candidates = []
            cols = min(beam_size, ys.size(1))  # ys.size(1) = vocab_s
            rows = beam_size
            if t == 0: # from <bos>
                rows = 1
            for c in range(cols): # for each column --which word
                for q in range(rows): # for each beam expansion --which beam
                    #compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q, c].item()  # c代表概率第c大的概率，应修改为满足不和之前重复的第c大的概率，且在输入的semantic role的范围之内
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': local_logprob})
            candidates = sorted(candidates, key=lambda x: -x['p']) # probs descening 

            # beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1: # save the previous beam_seq and beam_seq_logporbs
            # we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()   # beam_seq_prev (t, beam_size)
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()   # beam_seq_logprobs_prev (t, beam_size)
            for vix in range(beam_size):  # for each beam
                v = candidates[vix]  # take the top beam_size information, as the topper, the better
                # fork beam index q into index vix
                if t >= 1: # update previous beam_seq
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                # append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation.
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum

        # start beam search
        beam_size = self.beam_size
        # device = prior_state.device

        beam_seq = torch.LongTensor(self.max_len, beam_size).zero_()  # include <eos>?
        beam_seq_logprobs = torch.FloatTensor(self.max_len, beam_size).zero_() # (seq_len, beam_size)
        beam_logprobs_sum = torch.zeros(beam_size) # running sum of logprobs for each beam, it decides the best sequence
        done_beams = []
        bos = beam_seq.new_zeros((beam_size, 1))

        for t in range(self.max_len):
            logprobsf = logprobs.data.float() # lets go to CPU for more efficiency in indexing operations (beam_size, vocab_s)

            beam_seq,\
            beam_seq_logprobs,\
            beam_logprobs_sum = beam_step(logprobsf,
                                        beam_size,
                                        t,
                                        beam_seq,
                                        beam_seq_logprobs,
                                        beam_logprobs_sum)

            for vix in range(beam_size): # for each beam
                # if time's up... or if end token is reached then copy beams
                if beam_seq[t, vix] == 0 or t == self.max_len - 1:
                    final_beam = {
                        'seq': beam_seq[:, vix].clone(),
                        'logps': beam_seq_logprobs[:, vix].clone(),
                        'p': beam_logprobs_sum[vix].item()
                    }
                    done_beams.append(final_beam)
                    # don't continue beams from finished sequences
                    beam_logprobs_sum[vix] = -1000

            # encode as vectors
            
            it = beam_seq[:(t+1)].t() # (t, beam_size) --> (beam_size, t)
            it = torch.cat((bos, it), dim=1)
            
            # itp = beam_seq_logprobs[:(t+1)].t()
            prior_state_ = prior_state.expand(beam_size, prior_state.size(0), prior_state.size(1)) # (beam_size, 36, 512)
            decoder_states = self.decoder(it.cuda(), None, prior_state_, None) # decoder_states: (beam_size, 1, hidden_size)
            logprobs = F.log_softmax(self.expander_nn(decoder_states[:, -1]), dim=-1)  # (beam_size, vocab_s)

        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size] # choose the top beam_size results
        return done_beams
