from speaksee.data import TextField
import os, sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from data import COCOControlSetField, ImageDetectionsField
from data.dataset import COCOEntities
from speaksee.data import DataLoader, DictionaryDataset, RawField
from models import SinkhornNet
from config import *
import torch
import torch.optim as optim
import torch.nn as nn
import random
import argparse
from tqdm import tqdm
from utils import set_lr, add_summary_value, pickle_load, pickle_dump

random.seed(1234)
torch.manual_seed(1234)
device = torch.device('cuda')

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='coco', type=str, help='dataset: coco | flickr')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--nb_workers', default=0, type=int, help='number of workers')
parser.add_argument('--learning_rate', default=1e-4, type=float, help='the init learning rate')
parser.add_argument('--learning_rate_decay_every', type=int, default=3)
parser.add_argument('--learning_rate_decay_rate', type=float, default=0.6)
parser.add_argument('--checkpoint_path', type=str, default="res")
parser.add_argument('--start_from', type=str, default=None)
parser.add_argument('--load_best', type=bool, default=False)
parser.add_argument('--eval', action='store_true')

opt = parser.parse_args()
print(opt)

# define the dataloader
image_field = ImageDetectionsField(detections_path=os.path.join(coco_root, 'coco_detections.hdf5'), load_in_tmp=False)

det_field = COCOControlSetField(detections_path=os.path.join(coco_root, 'coco_detections.hdf5'),
                                classes_path=os.path.join(coco_root, 'object_class_list.txt'),
                                img_shapes_path=os.path.join(coco_root, 'coco_img_shapes.json'),
                                precomp_glove_path=os.path.join(coco_root, 'object_class_glove.pkl'),
                                verb_idx_path=os.path.join(coco_root, 'verb_2_idx.json'),
                                idx_vs_path=os.path.join(coco_root, 'idx_2_vs.json'),
                                cap_classes_path=os.path.join(coco_root, 'cap_2_classes.json'),
                                cap_verb_path=os.path.join(coco_root, 'cap_2_verb.json'),
                                fix_length=10, max_detections=20)

text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, remove_punctuation=True, fix_length=20)

test_dataset = COCOEntities(image_field, det_field, RawField(),
                            img_root='',
                            ann_root=os.path.join(coco_root, 'annotations'),
                            entities_file=os.path.join(coco_root, 'coco_entities.json'),
                            id_root=os.path.join(coco_root, 'annotations'),
                            filtering=True)

train_dataset, val_dataset, test_dataset = test_dataset.splits
test_dataset = DictionaryDataset(test_dataset.examples, test_dataset.fields, 'image')
dataloader_test = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=opt.nb_workers)
train_dataset = DictionaryDataset(train_dataset.examples, train_dataset.fields, 'image')
dataloader_train = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.nb_workers)

tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

# region sort model define
sinkhorn_len = 10
sinkhorn_net = SinkhornNet(sinkhorn_len, 20, 0.1).cuda()

# set the optimizer
optimizer = optim.Adam(sinkhorn_net.parameters(), opt.learning_rate, (0.9, 0.999), 1e-8)

infos = {}

if vars(opt).get('start_from', None) is not None:
    if opt.load_best:
        if os.path.isfile(os.path.join(opt.start_from, "model-best.pth")):
            sinkhorn_net.load_state_dict(torch.load(
                os.path.join(opt.start_from, 'model-best.pth')))
        if os.path.isfile(os.path.join(opt.start_from, "optimizer-best.pth")):
            optimizer.load_state_dict(torch.load(
                os.path.join(opt.start_from, 'optimizer-best.pth')))
    else:
        if os.path.isfile(os.path.join(opt.start_from, "model.pth")):
            sinkhorn_net.load_state_dict(torch.load(
                os.path.join(opt.start_from, 'model.pth')))
        if os.path.isfile(os.path.join(opt.start_from, "optimizer.pth")):
            optimizer.load_state_dict(torch.load(
                os.path.join(opt.start_from, 'optimizer.pth')))
        if os.path.isfile(os.path.join(opt.start_from, 'infos.pkl')):
            with open(os.path.join(opt.start_from, 'infos.pkl'), 'rb') as f:
                infos = pickle_load(f)
else:
    infos['iter'] = 0
    infos['epoch'] = 0

def save_checkpoint(model, infos, optimizer, append='sh'):
    if len(append) > 0:
        append = '-' + append
    # if checkpoint_path doesn't exist
    if not os.path.isdir(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)
    checkpoint_path = os.path.join(
        opt.checkpoint_path, 'model%s.pth' % (append))
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))
    with open(os.path.join(opt.checkpoint_path, 'infos%s.pkl' % (append)), 'wb') as f:
        pickle_dump(infos, f)
    optimizer_path = os.path.join(
        opt.checkpoint_path, 'optimizer%s.pth' % (append))
    torch.save(optimizer.state_dict(), optimizer_path)

iteration = infos['iter']
start_epoch = infos['epoch']
for e in range(start_epoch, start_epoch + 100):
    # Training with cross-entropy
    # learning rate decay
    if e >= 3:
        current_lr = opt.learning_rate * (opt.learning_rate_decay_rate ** int((e - 3) // opt.learning_rate_decay_every + 1))
    else:
        current_lr = opt.learning_rate
    if e == 20:
        break
    set_lr(optimizer, current_lr)
    running_loss = 0.
    max_len = 0
    criterion = nn.MSELoss()
    if opt.eval is False:
        sinkhorn_net.train()
        with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(iter(dataloader_train))) as pbar:
            for it, (keys, values) in enumerate(iter(dataloader_train)):
                detections, _ = keys  # b_s, 100, feat
                det_seqs_txt, det_seqs_vis, det_seqs_pos, det_seqs_all, det_seqs_v, det_seqs_sr, control_verb, \
                    gt_seqs_v, gt_seqs_sr, idx_list, _ = values

                optimizer.zero_grad()
                loss = 0.
                for i in range(detections.size(0)):  # batch
                    # add a region sort model
                    for idx in range(len(control_verb[i])):
                        this_seqs_vis = det_seqs_vis[i][idx]
                        this_seqs_txt = det_seqs_txt[i][idx]
                        this_seqs_pos = det_seqs_pos[i][idx]  # pos是position信息
                        this_seqs_all = det_seqs_all[i][idx]
                        this_control_verb = control_verb[i][idx]  # (max_verb)
                        # (fixed_len, max_verb)
                        this_det_seqs_v = det_seqs_v[i][idx]
                        # (fixed_len, max_sr)
                        this_det_seqs_sr = det_seqs_sr[i][idx]
                        this_gt_seqs_v = gt_seqs_v[i][idx]
                        this_gt_seqs_sr = gt_seqs_sr[i][idx]
                        this_idx_list = idx_list[i][idx].squeeze()
                        fix_length = this_det_seqs_v.shape[0]
                        this_seqs_perm = torch.cat(
                            (this_seqs_vis, this_seqs_txt, this_seqs_pos), -1)
                        for verb in this_control_verb:
                            # 找到某个verb对应的semantic role序列
                            if verb == 0:
                                break
                            verb_gt_seqs_sr = this_gt_seqs_sr.new_zeros(
                                this_det_seqs_sr.shape[0])
                            verb_det_seqs_sr = this_det_seqs_sr.new_zeros(
                                this_det_seqs_sr.shape[0])
                            find_sr = 0
                            find_gt_sr = 0
                            sr_find = {}
                            need_re_rank = set()
                            for j, vs in enumerate(this_det_seqs_v):
                                for k, v in enumerate(vs):
                                    if verb == v and find_sr < 10:
                                        if this_det_seqs_sr[j][k].item() not in sr_find:
                                            sr_find[this_det_seqs_sr[j][k].item()] = []
                                            sr_find[this_det_seqs_sr[j][k].item()].append(j)
                                            verb_det_seqs_sr[find_sr] = this_det_seqs_sr[j][k].item()
                                            find_sr += 1
                                        else:
                                            sr_find[this_det_seqs_sr[j][k].item()].append(j)
                                            need_re_rank.add(this_det_seqs_sr[j][k].item())

                            if find_sr == 0:
                                continue
                            if len(need_re_rank) != 0:
                                for sr in need_re_rank:
                                    this_sr_perm = torch.zeros(sinkhorn_len, this_seqs_perm.shape[1])
                                    tr_locs = torch.ones(sinkhorn_len) * 10
                                    gt_locs = torch.ones(sinkhorn_len) * 10
                                    gt_locs_ = torch.ones(sinkhorn_len) * 10
                                    for j, loc in enumerate(sr_find[sr]):
                                        tr_locs[j] = loc
                                        gt_locs[j] = this_idx_list[loc]
                                        this_sr_perm[j, :] = this_seqs_perm[loc]
                                    change_ = torch.argsort(gt_locs, -1)
                                    matrix = torch.zeros((sinkhorn_len, sinkhorn_len))
                                    for j in range(matrix.shape[0]):
                                        if j < len(sr_find[sr]):
                                            matrix[j, change_[j]] = 1
                                            gt_locs_[j] = change_[j]
                                        else:
                                            matrix[j, j] = 1

                                    tr_matrix = sinkhorn_net(this_sr_perm.unsqueeze(0).to(device)).squeeze()
                                    resort_locs = torch.mm(tr_locs.unsqueeze(0).to(device), tr_matrix).squeeze()
                                    loss += criterion(resort_locs, gt_locs_.to(device))

                loss = loss / detections.size(0)
                if loss != 0.:
                    add_summary_value(tb_summary_writer, 'train_loss', loss, iteration)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                iteration += 1
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
        infos['epoch'] = e
        infos['iter'] = iteration
        save_checkpoint(sinkhorn_net, infos, optimizer)
