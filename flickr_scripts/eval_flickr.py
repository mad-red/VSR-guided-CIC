from speaksee.data import TextField
import os, sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from data import FlickrDetectionField, FlickrControlSetField_Verb, FlickrDetSetField_Verb
from data.dataset import FlickrEntities
from models import ControllableCaptioningModel
from speaksee.data import DataLoader, DictionaryDataset, RawField
from speaksee.evaluation import Bleu, Meteor, Rouge, Cider, Spice
from speaksee.evaluation import PTBTokenizer
from models import SinkhornNet, S_SSP
from config import *
import torch
import random
import numpy as np
import itertools
import argparse
import munkres
from tqdm import tqdm
from utils import verb_rank_merge

random.seed(1234)
torch.manual_seed(1234)
device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='flickr', type=str, help='dataset: coco | flickr')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--nb_workers', default=10, type=int, help='number of workers')
parser.add_argument('--checkpoint_path', type=str, default="res")
parser.add_argument('--start_from', type=str, default=None)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--det', action='store_true')
parser.add_argument('--gt', action='store_true')
parser.add_argument('--verb', action='store_true')

opt = parser.parse_args()
print(opt)

saved_data = torch.load('saved_model/flickr_cap/ours_flickr_rl.pth')
opt_cap = saved_data['opt']

# define the field
image_field = FlickrDetectionField(detections_path=os.path.join(flickr_root, 'flickr30k_detections.hdf5'), diverse=True)

if not opt.det:
    det_field = FlickrControlSetField_Verb(detections_path=os.path.join(flickr_root, 'flickr30k_detections.hdf5'),
                                        classes_path=os.path.join(flickr_root, 'object_class_list.txt'),
                                        img_shapes_path=os.path.join(flickr_root, 'flickr_img_shapes.json'),
                                        precomp_glove_path=os.path.join(flickr_root, 'object_class_glove.pkl'),
                                        verb_idx_path=os.path.join(flickr_root, 'flickr_verb_idx.json'),
                                        idx_vs_path=os.path.join(flickr_root, 'idx_2_vs_flickr.json'),
                                        cap_verb_path=os.path.join(flickr_root, 'cap_2_verb_flickr.json'),
                                        cap_classes_path=os.path.join(flickr_root, 'cap_2_classes_flickr.json'),
                                        idx_v_og_path=os.path.join(flickr_root, 'idx_2_v_og_flickr.json'),
                                        vocab_list_path=os.path.join(flickr_root, 'vocab_tv_flickr.json'),
                                        fix_length=10, gt_verb=opt.gt)
else:
    det_field = FlickrDetSetField_Verb(detections_path=os.path.join(flickr_root, 'flickr30k_detections.hdf5'),
                                        classes_path=os.path.join(flickr_root, 'object_class_list.txt'),
                                        img_shapes_path=os.path.join(flickr_root, 'flickr_img_shapes.json'),
                                        precomp_glove_path=os.path.join(flickr_root, 'object_class_glove.pkl'),
                                        verb_idx_path=os.path.join(flickr_root, 'flickr_verb_idx.json'),
                                        verb_vob_path=os.path.join(flickr_root, 'verb_2_vob_flickr.json'),
                                        idbox_seq_path=os.path.join('saved_data/flickr', 'img_cap_v_2_idbox_flickr.json'),
                                        vocab_list_path=os.path.join(flickr_root, 'vocab_tv_flickr.json'),
                                        vlem_2_verb_og_path=os.path.join(flickr_root, 'vlem_2_vog_flickr.json'),
                                        fix_length=10, gt_verb=opt.gt)

text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, remove_punctuation=True, fix_length=20)

# define the datasets
dataset = FlickrEntities(image_field, text_field, det_field,
                        img_root='',
                        ann_file=os.path.join(flickr_root, 'flickr30k_annotations.json'),
                        entities_root=flickr_entities_root)

if not opt.det:
    test_dataset = FlickrEntities(image_field, RawField(), det_field,
                                img_root='',
                                ann_file=os.path.join(flickr_root, 'flickr30k_annotations.json'),
                                entities_root=flickr_entities_root,
                                verb_filter=True)
else:
    test_dataset = FlickrEntities(image_field, RawField(), det_field,
                                img_root='',
                                ann_file=os.path.join(flickr_root, 'flickr30k_annotations.json'),
                                entities_root=flickr_entities_root,
                                det_filter=True)

train_dataset, val_dataset, _ = dataset.splits
text_field.build_vocab(train_dataset, val_dataset, min_freq=5)

# define the dataloader
_, _, test_dataset = test_dataset.splits
test_dataset = DictionaryDataset(test_dataset.examples, test_dataset.fields, 'image')
dataloader_test = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=opt.nb_workers)

# captioning model
model = ControllableCaptioningModel(20, len(text_field.vocab), text_field.vocab.stoi['<bos>'], \
        h2_first_lstm=opt_cap.h2_first_lstm, img_second_lstm=opt_cap.img_second_lstm, dataset='flickr').to(device)
model.eval()
model.load_state_dict(saved_data['state_dict'])


# region sort model
re_sort_net = S_SSP(dataset='flickr').cuda()
re_sort_net.load_state_dict(torch.load(os.path.join('saved_model/flickr_npos_fc_v', 'model-tr.pth')))
re_sort_net.eval()

sinkhorn_len = 10
sinkhorn_net = SinkhornNet(sinkhorn_len, 20, 0.1).cuda()
sinkhorn_net.load_state_dict(torch.load(os.path.join('saved_model/flickr_sinkhorn', 'model-sh.pth')))
sinkhorn_net.eval()

fixed_len = 10
predictions = []
gt_captions = []

# Evaluate
with tqdm(desc='Test', unit='it', ncols=110, total=len(iter(dataloader_test))) as pbar:
    with torch.no_grad():
        for it, (keys, values) in enumerate(iter(dataloader_test)):
            detections, image_ids = keys  # b_s, 100, feat
            if not opt.det:
                det_seqs_txt, det_seqs_vis, det_seqs_pos, det_seqs_all, det_seqs_v, det_seqs_sr, control_verb, \
                _, _, _, verb_list, captions = values
            else:
                det_seqs_txt, det_seqs_vis, det_seqs_pos, det_seqs_all, det_seqs_v, det_seqs_sr, control_verb, \
                verb_list, captions = values
            
            for i in range(detections.size(0)):  # batch
                # add a region sort model
                det_seqs_recons = np.zeros(det_seqs_all[i].shape)
                img_verb_list = np.zeros(verb_list[i].shape)

                for idx in range(len(control_verb[i])):  # caption数目
                    # visual feature
                    this_seqs_vis = det_seqs_vis[i][idx]
                    this_seqs_txt = det_seqs_txt[i][idx]
                    this_seqs_pos = det_seqs_pos[i][idx]  # pos是position信息
                    this_seqs_all = det_seqs_all[i][idx]

                    # semantic role and verb
                    this_control_verb = control_verb[i][idx]  # (max_verb)
                    this_det_seqs_v = det_seqs_v[i][idx]  # (fixed_len, max_verb)
                    this_det_seqs_sr = det_seqs_sr[i][idx]  # (fixed_len, max_sr)
                    
                    this_verb_list = verb_list[i][idx]

                    # visual feature concat
                    this_seqs_perm = torch.cat((this_seqs_vis, this_seqs_txt, this_seqs_pos), -1)
                    
                    verb_ranks = []

                    for verb in this_control_verb:
                        # 找到某个verb对应的semantic role序列
                        if verb == 0:
                            break
                        verb_det_seqs_sr = this_det_seqs_sr.new_zeros(this_det_seqs_sr.shape[0])
                        find_sr = 0
                        sr_find = {}
                        need_re_rank = set()
                        for j, vs in enumerate(this_det_seqs_v): # fixed_len
                            for k, v in enumerate(vs):  # max_verb
                                if verb == v and find_sr < 10:
                                    if int(this_det_seqs_sr[j][k].item()) not in sr_find:
                                        sr_find[int(this_det_seqs_sr[j][k].item())] = []
                                        sr_find[int(this_det_seqs_sr[j][k].item())].append(j)
                                        verb_det_seqs_sr[find_sr] = this_det_seqs_sr[j][k].item()
                                        find_sr += 1
                                    else:
                                        sr_find[int(this_det_seqs_sr[j][k].item())].append(j)
                                        need_re_rank.add(int(this_det_seqs_sr[j][k].item()))
                        
                        if find_sr == 0:
                            continue
                        this_verb = verb.unsqueeze(0).to(device)
                        verb_det_seqs_sr = verb_det_seqs_sr.unsqueeze(0).to(device)
                        
                        output = re_sort_net.generate(this_verb, verb_det_seqs_sr, mode='not-normal')
                        sr_rank = {}
                        if len(need_re_rank) != 0:
                            for sr in need_re_rank:
                                this_sr_perm = torch.zeros(sinkhorn_len, this_seqs_perm.shape[1])
                                tr_locs = torch.ones(sinkhorn_len) * 10
                                for j, loc in enumerate(sr_find[sr]):
                                    if j >= sinkhorn_len:
                                        continue
                                    tr_locs[j] = loc
                                    this_sr_perm[j, :] = this_seqs_perm[loc]
                                
                                tr_matrix = sinkhorn_net(this_sr_perm.unsqueeze(0).to(device))
                                
                                mx = torch.transpose(tr_matrix, 1, 2).squeeze()
                                if isinstance(mx, torch.Tensor):
                                    mx = mx.detach().cpu().numpy()
                                
                                m = munkres.Munkres()
                                ass = m.compute(munkres.make_cost_matrix(mx))
                                sr_re = []
                                for idx_ in range(len(sr_find[sr])):
                                    for a in ass:
                                        if a[0] == idx_:
                                            sr_re.append(a[1])

                                sr_re = np.array(sr_re)
                                sr_idx = np.argsort(sr_re)
                                output_idx = np.zeros(len(sr_find[sr]))
                                for j, idx_ in enumerate(sr_idx):
                                    output_idx[j] = sr_find[sr][idx_]
                                sr_rank[sr] = output_idx

                        verb_rank = []
                        for sr_ in output[0].squeeze().cpu().numpy():
                            if sr_ == 0:
                                break
                            if len(sr_find[sr_]) != 1:
                                verb_rank += list(sr_rank[sr_])
                            else:
                                verb_rank += sr_find[sr_]
                        verb_ranks.append(verb_rank)

                    final_rank = []
                    if len(verb_ranks) == 1:
                        final_rank = verb_ranks[0]
                    else:
                        final_rank = verb_ranks[0]
                        for j in range(len(verb_ranks) - 1):
                            final_rank = verb_rank_merge(final_rank, verb_ranks[j+1])
                    
                    perm_matrix = np.zeros((fixed_len, fixed_len))
                    for j, rk in enumerate(final_rank):
                        if j < 10:
                            perm_matrix[j, int(rk)] = 1

                    perm = np.reshape(this_seqs_all, (this_seqs_all.shape[0], -1))  # fixed_len, -1
                    recons = np.dot(perm_matrix, perm)
                    recons = np.reshape(recons, this_seqs_all.shape[0:])
                    recons = recons[np.sum(recons, (1, 2)) != 0]

                    last = recons.shape[0] - 1
                    det_seqs_recons[idx, :recons.shape[0]] = recons
                    det_seqs_recons[idx, last + 1:] = recons[last:last+1]
                    
                    # permute the verb_list
                    perm_mask = (np.sum(perm_matrix, -1) == 0).astype(int)
                    img_verb_list[idx] = -1 * perm_mask[:, np.newaxis] + np.dot(perm_matrix, this_verb_list)
                
                # detections_i: (1, det_len, feat_dim), det_seqs_recons: (1, fixed_len, max_det, feat_dim)
                img_verb_list = torch.tensor(img_verb_list).to(device).squeeze(-1)
                detections_i, det_seqs_recons = detections[i].to(device), torch.tensor(det_seqs_recons).float().to(device)
                detections_i = detections_i.unsqueeze(0).expand(det_seqs_recons.size(0), detections_i.size(0), detections_i.size(1))
                out, _ = model.beam_search_v((detections_i, det_seqs_recons, img_verb_list), 
                                                eos_idxs=[text_field.vocab.stoi['<eos>'], -1], beam_size=5, out_size=1, gt=opt.gt)

                out = out[0].data.cpu().numpy()
                
                for o, caps in zip(out, captions[i]):
                    predictions.append(np.expand_dims(o, axis=0))
                    gt_captions.append(caps)
                
            pbar.update()

# Compute the metric scores
predictions = np.concatenate(predictions, axis=0)
gen = {}
gts = {}

print("Computing accuracy performance.")
for i, cap in enumerate(predictions):
    pred_cap = text_field.decode(cap, join_words=False)
    pred_cap = ' '.join([k for k, g in itertools.groupby(pred_cap)])
    gts[i] = [gt_captions[i]]
    gen[i] = [pred_cap]

gts_t = PTBTokenizer.tokenize(gts)
gen_t = PTBTokenizer.tokenize(gen)

val_bleu, _ = Bleu(n=4).compute_score(gts_t, gen_t)
method = ['Blue_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']
for metric, score in zip(method, val_bleu):
    print(metric, score)

val_meteor, _ = Meteor().compute_score(gts_t, gen_t)
print('METEOR', val_meteor)

val_rouge, _ = Rouge().compute_score(gts_t, gen_t)
print('ROUGE_L', val_rouge)

val_cider, _ = Cider().compute_score(gts_t, gen_t)
print('CIDEr', val_cider)

val_spice, _ = Spice().compute_score(gts_t, gen_t)
print('SPICE', val_spice)
