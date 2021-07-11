from __future__ import division
from __future__ import absolute_import
import torch
from torch import nn
import torch.nn.functional as F
from models import _CaptioningModel
import json, os


class ControllableCaptioningModel(_CaptioningModel):
    def __init__(self, seq_len, vocab_size, bos_idx, det_feat_size=2048, input_encoding_size=1000, rnn_size=1000, att_size=512,
                 h2_first_lstm=True, img_second_lstm=False, dataset='coco'):
        super(ControllableCaptioningModel, self).__init__(seq_len)
        self.vocab_size = vocab_size
        self.bos_idx = bos_idx
        self.det_feat_size = det_feat_size
        self.input_encoding_size = input_encoding_size
        self.rnn_size = rnn_size
        self.att_size = att_size
        self.h2_first_lstm = h2_first_lstm
        self.img_second_lstm = img_second_lstm

        self.embed = nn.Embedding(vocab_size, input_encoding_size)

        if dataset == 'coco':
            with open(os.path.join('datasets/coco', 'verb_2_vob_all_refine.json')) as f:
                self.verb_2_vob_all = json.load(f)
            with open(os.path.join('datasets/coco', 'verb_2_vob.json')) as f:
                self.verb_2_vob = json.load(f)
        else:
            with open(os.path.join('datasets/flickr', 'verb_2_vob_all_refine_flickr.json')) as f:
                self.verb_2_vob_all = json.load(f)
            with open(os.path.join('datasets/flickr', 'verb_2_vob_flickr.json')) as f:
                self.verb_2_vob = json.load(f)

        if self.h2_first_lstm:
            self.W1_is = nn.Linear(det_feat_size + rnn_size + input_encoding_size, rnn_size)
        else:
            self.W1_is = nn.Linear(det_feat_size + input_encoding_size, rnn_size)
        self.W1_hs = nn.Linear(rnn_size, rnn_size)

        self.att_va = nn.Linear(det_feat_size, att_size, bias=False)
        self.att_ha = nn.Linear(rnn_size, att_size, bias=False)
        self.att_a = nn.Linear(att_size, 1, bias=False)

        self.att_sa = nn.Linear(rnn_size, att_size, bias=False)
        self.att_s = nn.Linear(att_size, 1, bias=False)

        if self.h2_first_lstm:
            self.lstm_cell_1 = nn.LSTMCell(det_feat_size + rnn_size + input_encoding_size, rnn_size)
        else:
            self.lstm_cell_1 = nn.LSTMCell(det_feat_size + input_encoding_size, rnn_size)

        if self.img_second_lstm:
            self.lstm_cell_2 = nn.LSTMCell(rnn_size + det_feat_size + det_feat_size, rnn_size)
        else:
            self.lstm_cell_2 = nn.LSTMCell(rnn_size + det_feat_size, rnn_size)

        self.out_fc = nn.Linear(rnn_size, vocab_size)
        self.s_fc = nn.Linear(rnn_size, det_feat_size)

        if self.h2_first_lstm:
            self.W1_ig = nn.Linear(det_feat_size + rnn_size + input_encoding_size, rnn_size)
        else:
            self.W1_ig = nn.Linear(det_feat_size + input_encoding_size, rnn_size)
        self.W1_hg = nn.Linear(rnn_size, rnn_size)
        self.att_ga = nn.Linear(rnn_size, att_size, bias=False)
        self.att_g = nn.Linear(att_size, 1, bias=False)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.out_fc.weight)
        nn.init.constant_(self.out_fc.bias, 0)

        nn.init.xavier_normal_(self.s_fc.weight)
        nn.init.constant_(self.s_fc.bias, 0)

        nn.init.xavier_normal_(self.W1_is.weight)
        nn.init.constant_(self.W1_is.bias, 0)

        nn.init.xavier_normal_(self.W1_hs.weight)
        nn.init.constant_(self.W1_hs.bias, 0)

        nn.init.xavier_normal_(self.att_va.weight)
        nn.init.xavier_normal_(self.att_ha.weight)
        nn.init.xavier_normal_(self.att_a.weight)
        nn.init.xavier_normal_(self.att_sa.weight)
        nn.init.xavier_normal_(self.att_s.weight)

        nn.init.xavier_normal_(self.lstm_cell_1.weight_ih)
        nn.init.orthogonal_(self.lstm_cell_1.weight_hh)
        nn.init.constant_(self.lstm_cell_1.bias_ih, 0)
        nn.init.constant_(self.lstm_cell_1.bias_hh, 0)

        nn.init.xavier_normal_(self.lstm_cell_2.weight_ih)
        nn.init.orthogonal_(self.lstm_cell_2.weight_hh)
        nn.init.constant_(self.lstm_cell_2.bias_ih, 0)
        nn.init.constant_(self.lstm_cell_2.bias_hh, 0)

        nn.init.xavier_normal_(self.W1_ig.weight)
        nn.init.constant_(self.W1_ig.bias, 0)
        nn.init.xavier_normal_(self.W1_hg.weight)
        nn.init.constant_(self.W1_hg.bias, 0)
        nn.init.xavier_normal_(self.att_ga.weight)
        nn.init.xavier_normal_(self.att_g.weight)

    def init_state(self, b_s, device):
        h0_1 = torch.zeros((b_s, self.rnn_size), requires_grad=True).to(device)
        c0_1 = torch.zeros((b_s, self.rnn_size), requires_grad=True).to(device)
        h0_2 = torch.zeros((b_s, self.rnn_size), requires_grad=True).to(device)
        c0_2 = torch.zeros((b_s, self.rnn_size), requires_grad=True).to(device)
        ctrl_det_idxs = torch.zeros((b_s, ), requires_grad=True).long().to(device)
        return (h0_1, c0_1), (h0_2, c0_2), ctrl_det_idxs

    def step(self, t, state, prev_outputs, statics, seqs, *args, mode='teacher_forcing'):
        ''' statics[0]: (b_s, det_len, feat_dim), 
            statics[1]: (b_s, fixed_len, max_det, feat_dim)
        '''
        assert (mode in ['teacher_forcing', 'feedback'])
        bos_idx = self.bos_idx
        detections = statics[0]
        b_s = detections.size(0)

        detections_mask = (torch.sum(detections, -1, keepdim=True) != 0).float()
        # image_descriptor是pooled的结果
        image_descriptor = torch.sum(detections, 1) / torch.sum(detections_mask, 1)
        state_1, state_2, ctrl_det_idxs = state

        if mode == 'teacher_forcing':
            it = seqs[0][:, t]
            det_curr = seqs[1][:, t]
        elif mode == 'feedback':
            if t == 0:
                it = detections.data.new_full((b_s,), bos_idx).long()
            else:
                it = prev_outputs[0]
                ctrl_det_idxs = ctrl_det_idxs + prev_outputs[1]
                ctrl_det_idxs = torch.clamp(ctrl_det_idxs, 0, statics[1].shape[1]-1)

            det_curr = torch.gather(statics[1], 1, ctrl_det_idxs.view((b_s, 1, 1, 1)).expand((b_s, 1) + statics[1].shape[2:])).squeeze(1)

        xt = self.embed(it)

        if self.h2_first_lstm:
            input_1 = torch.cat([state_2[0], image_descriptor, xt], 1)
        else:
            input_1 = torch.cat([image_descriptor, xt], 1)

        s_gate = torch.sigmoid(self.W1_is(input_1) + self.W1_hs(state_1[0]))
        state_1 = self.lstm_cell_1(input_1, state_1)

        s_t = s_gate * torch.tanh(state_1[1])
        fc_sentinel = self.s_fc(s_t).unsqueeze(1)
        
        # fc_sentinel: (b_s, 1, 2048)
        regions = torch.cat([fc_sentinel, det_curr], 1)
        regions_mask = (torch.sum(regions, -1, keepdim=True) != 0).float()

        det_weights = torch.tanh(self.att_va(det_curr) + self.att_ha(state_1[0]).unsqueeze(1))
        det_weights = self.att_a(det_weights)
        sent_weights = torch.tanh(self.att_sa(s_t) + self.att_ha(state_1[0])).unsqueeze(1)
        sent_weights = self.att_s(sent_weights)
        att_weights = torch.cat([sent_weights, det_weights], 1)

        att_weights = F.softmax(att_weights, 1)  # (b_s, n_regions, 1)
        att_weights = regions_mask * att_weights
        att_weights = att_weights / torch.sum(att_weights, 1, keepdim=True)
        # 最后加权打分，得到(b_s, 2048)作为需要attend的feature
        att_detections = torch.sum(regions * att_weights, 1)

        if self.img_second_lstm:
            input_2 = torch.cat([state_1[0], att_detections, image_descriptor], 1)
        else:
            input_2 = torch.cat([state_1[0], att_detections], 1)
        state_2 = self.lstm_cell_2(input_2, state_2)
        out = F.log_softmax(self.out_fc(state_2[0]), dim=-1)

        # (b_s, rnn_size)
        g_gate = torch.sigmoid(self.W1_ig(input_1) + self.W1_hg(state_1[0]))
        g_t = g_gate * torch.tanh(state_1[1])
        # (b_s, att_size)
        gate_weights = torch.tanh(self.att_ga(g_t) + self.att_ha(state_1[0])).unsqueeze(1)
        # (b_s, 1)
        gate_weights = self.att_g(gate_weights)
        gate_weights = torch.cat([gate_weights, torch.sum(regions_mask[:, 1:] * det_weights, 1, keepdim=True)], 1)
        gate_weights = F.log_softmax(gate_weights, 1).squeeze(-1)  # (b_s, 2)

        return (out, gate_weights), (state_1, state_2, ctrl_det_idxs)

    def step_v(self, t, state, prev_outputs, statics, seqs, *args, mode='teacher_forcing', gt=False):
        ''' statics[0]: (b_s, det_len, feat_dim), 
            statics[1]: (b_s, fixed_len, max_det, feat_dim), 
            statics[2]: (b_s, fixed_len)
        '''
        assert (mode in ['teacher_forcing', 'feedback'])
        bos_idx = self.bos_idx
        detections = statics[0]
        b_s = detections.size(0)
        device = statics[0].device

        detections_mask = (torch.sum(detections, -1, keepdim=True) != 0).float()
        # image_descriptor是pooled的结果
        image_descriptor = torch.sum(detections, 1) / torch.sum(detections_mask, 1)
        state_1, state_2, ctrl_det_idxs = state

        if mode == 'teacher_forcing':
            it = seqs[0][:, t]
            det_curr = seqs[1][:, t]
        elif mode == 'feedback':
            if t == 0:
                it = detections.data.new_full((b_s,), bos_idx).long()
            else:
                it = prev_outputs[0]
                ctrl_det_idxs = ctrl_det_idxs + prev_outputs[1]
                ctrl_det_idxs = torch.clamp(ctrl_det_idxs, 0, statics[1].shape[1]-1)

            det_curr = torch.gather(statics[1], 1, ctrl_det_idxs.view((b_s, 1, 1, 1)).expand((b_s, 1) + statics[1].shape[2:])).squeeze(1)
            # (b_s)
            verb_curr = torch.gather(statics[2], 1, ctrl_det_idxs.unsqueeze(-1)).squeeze(1).long()
            # verb_mask: (b_s, 1), 表示需要被替换的词
            verb_mask = (verb_curr != -1).int().unsqueeze(-1)

        xt = self.embed(it)

        if self.h2_first_lstm:
            input_1 = torch.cat([state_2[0], image_descriptor, xt], 1)
        else:
            input_1 = torch.cat([image_descriptor, xt], 1)

        s_gate = torch.sigmoid(self.W1_is(input_1) + self.W1_hs(state_1[0]))
        state_1 = self.lstm_cell_1(input_1, state_1)

        s_t = s_gate * torch.tanh(state_1[1])
        fc_sentinel = self.s_fc(s_t).unsqueeze(1)
        
        # fc_sentinel: (b_s, 1, 2048)
        regions = torch.cat([fc_sentinel, det_curr], 1)
        regions_mask = (torch.sum(regions, -1, keepdim=True) != 0).float()

        det_weights = torch.tanh(self.att_va(det_curr) + self.att_ha(state_1[0]).unsqueeze(1))
        det_weights = self.att_a(det_weights)
        sent_weights = torch.tanh(self.att_sa(s_t) + self.att_ha(state_1[0])).unsqueeze(1)
        sent_weights = self.att_s(sent_weights)
        att_weights = torch.cat([sent_weights, det_weights], 1)

        att_weights = F.softmax(att_weights, 1)  # (b_s, n_regions, 1)
        att_weights = regions_mask * att_weights
        att_weights = att_weights / torch.sum(att_weights, 1, keepdim=True)
        # 最后加权打分，得到(b_s, 2048)作为需要attend的feature
        att_detections = torch.sum(regions * att_weights, 1)

        if self.img_second_lstm:
            input_2 = torch.cat([state_1[0], att_detections, image_descriptor], 1)
        else:
            input_2 = torch.cat([state_1[0], att_detections], 1)
        state_2 = self.lstm_cell_2(input_2, state_2)

        # (b_s, rnn_size)
        g_gate = torch.sigmoid(self.W1_ig(input_1) + self.W1_hg(state_1[0]))
        g_t = g_gate * torch.tanh(state_1[1])
        # (b_s, att_size)
        gate_weights = torch.tanh(self.att_ga(g_t) + self.att_ha(state_1[0])).unsqueeze(1)
        # (b_s, 1)
        gate_weights = self.att_g(gate_weights)
        gate_weights = torch.cat([gate_weights, torch.sum(regions_mask[:, 1:] * det_weights, 1, keepdim=True)], 1)
        gate_weights = F.log_softmax(gate_weights, 1).squeeze(-1)  # (b_s, 2)

        # update the out and gate_weights
        change_gate = torch.tensor([-1e3, 0]).to(device)

        out = F.log_softmax(self.out_fc(state_2[0]), dim=-1)

        # (b_s, vob_size)
        verb_out = torch.ones(out.shape[:]).to(device) * -1e6
        for i in range(b_s):
            if verb_curr[i] != -1:
                if gt:
                    verb_out[i][verb_curr[i]] = 0
                else:
                    # only find verb from different tenses of the controlled <verb>
                    if str(verb_curr[i].item()) in self.verb_2_vob_all and len(self.verb_2_vob_all[str(verb_curr[i].item())]) != 0:
                        pro_max = -1e6
                        max_idx = -1
                        for idx in self.verb_2_vob_all[str(verb_curr[i].item())]:
                            if out[i][idx] > pro_max:
                                pro_max = out[i][idx]
                                max_idx = idx
                        verb_out[i][max_idx] = 0
                    else:
                        verb_out[i][0] = 0

        out = (1 - verb_mask) * out + verb_mask * verb_out
        gate_weights = (1 - verb_mask) * gate_weights + verb_mask * change_gate.unsqueeze(0)

        return (out, gate_weights), (state_1, state_2, ctrl_det_idxs)

    def test(self, detections, ctrl_det_seqs_test):
        return super(ControllableCaptioningModel, self).test((detections, ctrl_det_seqs_test))

    def sample_rl(self, detections, ctrl_det_seqs_test):
        return super(ControllableCaptioningModel, self).sample_rl((detections, ctrl_det_seqs_test))
