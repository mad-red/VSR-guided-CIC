import os
import warnings
import shutil
import numpy as np
import h5py
import pickle as pkl
import warnings
import json
import random
import torch
from itertools import groupby
from speaksee.data import RawField


class COCOControlSequenceField(RawField):
    def __init__(self, postprocessing=None, detections_path=None, classes_path=None,
                 padding_idx=0, fix_length=None, all_boxes=True, pad_init=True, pad_eos=True, dtype=torch.float32,
                 max_detections=20, max_length=100, sorting=False):
        self.max_detections = max_detections
        self.max_length = max_length
        self.detections_path = detections_path
        self.padding_idx = padding_idx
        self.fix_length = fix_length
        self.all_boxes = all_boxes
        self.sorting = sorting
        self.eos_token = padding_idx if pad_eos else None
        self.dtype = dtype

        self.classes = ['__background__']
        with open(classes_path, 'r') as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())

        super(COCOControlSequenceField, self).__init__(None, postprocessing)

    def get_detections_inside(self, det_boxes, query):
        cond1 = det_boxes[:, 0] >= det_boxes[query, 0]
        cond2 = det_boxes[:, 1] >= det_boxes[query, 1]
        cond3 = det_boxes[:, 2] <= det_boxes[query, 2]
        cond4 = det_boxes[:, 3] <= det_boxes[query, 3]
        cond = cond1 & cond2 & cond3 & cond4
        return np.nonzero(cond)[0]

    def _fill(self, cls_seq, det_features, det_boxes, selected_classes, most_probable_dets, max_len):
        det_sequences = np.zeros((self.fix_length, self.max_detections, det_features.shape[-1]))
        for j, cls in enumerate(cls_seq[:max_len]):
            if cls == '_':
                det_sequences[j, :det_features.shape[0]] = most_probable_dets
            else:
                seed_detections = [i for i, c in enumerate(selected_classes) if c == cls]
                if self.all_boxes:
                    det_ids = np.unique(np.concatenate([self.get_detections_inside(det_boxes, d) for d in seed_detections]))
                else:
                    det_ids = np.unique(seed_detections)
                det_sequences[j, :len(det_ids)] = np.take(det_features, det_ids, axis=0)[:self.max_detections]

        if not self.sorting:
            last = len(cls_seq[:max_len])
            det_sequences[last:] = det_sequences[last-1]

        return det_sequences.astype(np.float32)

    def preprocess(self, x):
        image = x[0][0]
        det_classes = x[1]
        max_len = self.fix_length + (self.eos_token, self.eos_token).count(None) - 2

        id_image = int(image.split('/')[-1].split('_')[-1].split('.')[0])
        try:
            f = h5py.File(self.detections_path, 'r')
            det_cls_probs = f['%s_cls_prob' % id_image][()]
            det_features = f['%s_features' % id_image][()]
            det_boxes = f['%s_boxes' % id_image][()]
        except KeyError:
            warnings.warn('Could not find detections for %d' % id_image)
            det_cls_probs = np.random.rand(10, 2048)
            det_features = np.random.rand(10, 2048)
            det_boxes = np.random.rand(10, 4)

        most_probable_idxs = np.argsort(np.max(det_cls_probs, -1))[::-1][:self.max_detections]  # 按概率从大到小的max_detections个
        most_probable_dets = det_features[most_probable_idxs]

        selected_classes = [self.classes[np.argmax(det_cls_probs[i][1:])+1] for i in range(len(det_cls_probs))]

        cls_seq = []
        for i, cls in enumerate(det_classes):
            if cls is not None:
                cls_seq.append(cls)
            else:
                cls_ok = next((c for c in det_classes[i+1:] if c is not None), '_')
                cls_seq.append(cls_ok)

        cls_seq_gt = np.asarray([int(a != b) for (a, b) in zip(cls_seq[:-1], cls_seq[1:])] + [0, ])
        cls_seq_gt = cls_seq_gt[:max_len]
        cls_seq_gt = np.concatenate([cls_seq_gt, [self.eos_token, self.eos_token]])
        cls_seq_gt = np.concatenate([cls_seq_gt, [self.padding_idx]*max(0, self.fix_length - len(cls_seq_gt))])
        cls_seq_gt = cls_seq_gt.astype(np.float32)

        cls_seq_test = [x[0] for x in groupby(det_classes) if x[0] is not None]
        if self.sorting:
            cls_seq_test.sort()
            det_sequences_test = self._fill(cls_seq_test, det_features, det_boxes, selected_classes, most_probable_dets, max_len)
            return det_sequences_test
        else:
            det_sequences = self._fill(cls_seq, det_features, det_boxes, selected_classes, most_probable_dets, max_len)
            det_sequences_test = self._fill(cls_seq_test, det_features, det_boxes, selected_classes, most_probable_dets, max_len)

            cls_seq_test = ' '.join(cls_seq_test)

            return det_sequences, cls_seq_gt, det_sequences_test, cls_seq_test # , id_image


# MSCOCO
class ImageDetectionsField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, detections_path=None, max_detections=100,
                 sort_by_prob=False, load_in_tmp=True):
        self.max_detections = max_detections
        self.detections_path = detections_path
        self.sort_by_prob = sort_by_prob

        tmp_detections_path = os.path.join('/tmp', os.path.basename(detections_path))
        if not os.path.isfile(tmp_detections_path):
            if shutil.disk_usage("/tmp")[-1] < os.path.getsize(detections_path):
                warnings.warn('Loading from %s, because /tmp has no enough space.' % detections_path)
            elif load_in_tmp:
                warnings.warn("Copying detection file to /tmp")
                shutil.copyfile(detections_path, tmp_detections_path)
                self.detections_path = tmp_detections_path
                warnings.warn("Done.")
        else:
            self.detections_path = tmp_detections_path

        super(ImageDetectionsField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x, avoid_precomp=False):
        image_id = int(x.split('_')[-1].split('.')[0])
        try:
            f = h5py.File(self.detections_path, 'r')
            precomp_data = f['%d_features' % image_id][()]
            if self.sort_by_prob:
                precomp_data = precomp_data[np.argsort(np.max(f['%d_cls_prob' % image_id][()], -1))[::-1]]
        except KeyError:
            warnings.warn('Could not find detections for %d' % image_id)
            precomp_data = np.random.rand(10,2048)

        delta = self.max_detections - precomp_data.shape[0]
        if delta > 0:
            precomp_data = np.concatenate([precomp_data, np.zeros((delta, precomp_data.shape[1]))], axis=0)
        elif delta < 0:
            precomp_data = precomp_data[:self.max_detections]

        return precomp_data.astype(np.float32), image_id


# MSCOCO detection field
class COCOControlSetField(RawField):
    def __init__(self, postprocessing=None, classes_path=None, img_shapes_path=None, 
                precomp_glove_path=None, verb_idx_path=None, idx_vs_path=None, cap_classes_path=None, 
                cap_verb_path=None, detections_path=None, fix_length=20, max_detections=20):
        self.fix_length = fix_length
        self.detections_path = detections_path
        self.max_detections = max_detections

        self.classes = ['__background__']
        with open(classes_path, 'r') as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())

        with open(precomp_glove_path, 'rb') as fp:
            self.vectors = pkl.load(fp)

        with open(img_shapes_path, 'r') as fp:
            self.img_shapes = json.load(fp)
        
        with open(verb_idx_path, 'r') as fp:
            self.verb_2_idx = json.load(fp)

        with open(idx_vs_path, 'r') as fp:
            self.idx_2_vs = json.load(fp)

        with open(cap_classes_path, 'r') as fp:
            self.cap_2_classes = json.load(fp)
        
        with open(cap_verb_path, 'r') as fp:
            self.cap_2_verb = json.load(fp)

        self.sr_2_idx = {'ARG0': 1, 'ARG1': 2, 'ARG2': 3, 'ARG3': 4, 'ARG4': 5, 'ARG5': 6, 'LOC': 7, 'DIR': 8, 'GOL': 9, 
                        'MNR': 10, 'TMP': 11, 'EXT': 12, 'REC': 13, 'PRD': 14, 'PRP': 15, 'CAU': 16, 'DIS': 17, 'ADV': 18, 
                        'ADJ': 19, 'MOD': 20, 'NEG': 21, 'LVB': 22, 'PNC': 23, 'COM': 24, 'V': 25}  # END is for predict END of the output

        super(COCOControlSetField, self).__init__(None, postprocessing)

    def preprocess(self, x):
        image = x[0][0]
        caption = x[0][1]
        id_image = int(image.split('/')[-1].split('_')[-1].split('.')[0])
        try:
            f = h5py.File(self.detections_path, 'r')
            det_cls_probs = f['%s_cls_prob' % id_image][()]
            det_features = f['%s_features' % id_image][()]
            det_boxes = f['%s_boxes' % id_image][()]
        except KeyError:
            warnings.warn('Could not find detections for %d' % id_image)
            det_cls_probs = np.random.rand(10, 2048)
            det_features = np.random.rand(10, 2048)
            det_boxes = np.random.rand(10, 4)

        idx_2_verb = self.idx_2_vs[str(id_image)][caption]['verb']
        idx_2_sr = self.idx_2_vs[str(id_image)][caption]['sr']
        cap_2_verb = self.cap_2_verb[str(id_image)][caption]
        cls_seq = self.cap_2_classes[str(id_image)][caption]

        selected_classes = [self.classes[np.argmax(det_cls_probs[i][1:]) + 1] for i in range(len(det_cls_probs))]
        width, height = self.img_shapes[str(id_image)]
        det_sequences_visual_all = np.zeros((self.fix_length, self.max_detections, det_features.shape[-1]))
        det_sequences_visual = np.zeros((self.fix_length, det_features.shape[-1]))
        det_sequences_word = np.zeros((self.fix_length, 300))
        det_sequences_position = np.zeros((self.fix_length, 4))

        # why set as 8?
        det_sequences_sr = np.zeros((self.fix_length, 8))
        det_sequences_verb = np.zeros((self.fix_length, 8))
        gt_det_sequences_sr = np.zeros((self.fix_length, 8))
        gt_det_sequences_verb = np.zeros((self.fix_length, 8))
        idx_list = np.zeros((self.fix_length, 1))
        idx_list[idx_list==0] = -1

        # why set as 8?
        control_verb = np.zeros(8)
        for j, verb in enumerate(cap_2_verb):
            control_verb[j] = self.verb_2_idx[verb] + 1  # 0代表没有verb

        cls_seq = cls_seq[:self.fix_length]  # 只保留前10个
        for j, cls in enumerate(cls_seq):
            for k, sr in enumerate(idx_2_sr[j]):
                if k == 8:
                    break
                gt_det_sequences_sr[j, k] = self.sr_2_idx[sr.split('-')[-1]]  # 0代表pad
                gt_det_sequences_verb[j, k] = self.verb_2_idx[idx_2_verb[j][k]] + 1 # 0代表pad

        idx_list_ = np.array(cls_seq).argsort()
        idx_list[:len(idx_list_), :] = idx_list_[:, np.newaxis]

        cls_seq.sort()  # 按字母序排列class, 相当于打乱排序，shuffle sequence
        for j, cls in enumerate(cls_seq):
            cls_w = cls.split(',')[0].split(' ')[-1]  # 当前class，为啥这么处理?
            if cls_w in self.vectors:
                det_sequences_word[j] = self.vectors[cls_w]
            seed_detections = [i for i, c in enumerate(selected_classes) if c == cls]  # 从detections里面找到class是选中class的detections序号
            det_ids = np.unique(seed_detections)

            # 当前class所有region的feature
            det_sequences_visual_all[j, :len(det_ids)] = np.take(det_features, det_ids, axis=0)[:self.max_detections]
            det_sequences_visual[j] = det_features[det_ids[0]]  # 当前class第一个region的feature
            bbox = det_boxes[det_ids[0]]  # 当前class第一个region的框 (x1, y1, x2, y2)
            det_sequences_position[j, 0] = (bbox[2] - bbox[0] / 2) / width  # 中心横坐标
            det_sequences_position[j, 1] = (bbox[3] - bbox[1] / 2) / height  # 中心纵坐标
            det_sequences_position[j, 2] = (bbox[2] - bbox[0]) / width  # 宽
            det_sequences_position[j, 3] = (bbox[3] - bbox[1]) / height  # 长

            for k, sr in enumerate(idx_2_sr[int(idx_list[j][0])]):
                if k >= 8:
                    continue
                det_sequences_sr[j, k] = self.sr_2_idx[sr.split('-')[-1]]  # 0代表pad
                det_sequences_verb[j, k] = self.verb_2_idx[idx_2_verb[int(idx_list[j][0])][k]] + 1 # 0代表pad
            
        return det_sequences_word.astype(np.float32), det_sequences_visual.astype(np.float32), \
               det_sequences_position.astype(np.float32), det_sequences_visual_all.astype(np.float32), \
               det_sequences_verb.astype(np.float32), det_sequences_sr.astype(np.float32), control_verb.astype(np.float32), \
               gt_det_sequences_verb.astype(np.float32), gt_det_sequences_sr.astype(np.float32), idx_list


class COCODetSetField(RawField):
    def __init__(self, postprocessing=None, verb_idx_path=None, detections_path=None, classes_path=None, 
                img_shapes_path=None, precomp_glove_path=None, cls_seq_path=None, fix_length=20, max_detections=20):
        self.fix_length = fix_length
        self.detections_path = detections_path
        self.max_detections = max_detections

        self.classes = ['__background__']

        with open(classes_path, 'r') as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())

        with open(precomp_glove_path, 'rb') as fp:
            self.vectors = pkl.load(fp)

        with open(img_shapes_path, 'r') as fp:
            self.img_shapes = json.load(fp)

        with open(verb_idx_path, 'r') as fp:
            self.verb_2_idx = json.load(fp)

        with open(cls_seq_path, 'r') as f:
            self.img_cap_v_2_class = json.load(f)

        self.sr_2_idx = {'ARG0': 1, 'ARG1': 2, 'ARG2': 3, 'ARG3': 4, 'ARG4': 5, 'ARG5': 6, 'LOC': 7, 'DIR': 8, 'GOL': 9, 
                        'MNR': 10, 'TMP': 11, 'EXT': 12, 'REC': 13, 'PRD': 14, 'PRP': 15, 'CAU': 16, 'DIS': 17, 'ADV': 18, 
                        'ADJ': 19, 'MOD': 20, 'NEG': 21, 'LVB': 22, 'PNC': 23, 'COM': 24, 'V': 25}  # END is for predict END of the output

        super(COCODetSetField, self).__init__(None, postprocessing)

    def preprocess(self, x):
        image = x[0][0]
        caption = x[0][1]
        id_image = int(image.split('/')[-1].split('_')[-1].split('.')[0])
        try:
            f = h5py.File(self.detections_path, 'r')
            det_cls_probs = f['%s_cls_prob' % id_image][()]
            det_features = f['%s_features' % id_image][()]
            det_boxes = f['%s_boxes' % id_image][()]
        except KeyError:
            warnings.warn('Could not find detections for %d' % id_image)
            det_cls_probs = np.random.rand(10, 2048)
            det_features = np.random.rand(10, 2048)
            det_boxes = np.random.rand(10, 4)

        v_2_class = self.img_cap_v_2_class[str(id_image)][caption]
        
        classes_seq = []
        loc_2_verb = {}
        loc_2_sr = {}
        loc = 0
        
        cap_2_verb = []
        for verb in v_2_class:
            for sr in v_2_class[verb]:
                for class_idx in v_2_class[verb][sr]:
                    if verb not in cap_2_verb:
                        cap_2_verb.append(verb)
                    classes_seq.append(class_idx)
                    loc_2_verb.setdefault(loc, []).append(verb)
                    loc_2_sr.setdefault(loc, []).append(sr)
                    loc += 1

        control_verb = np.zeros(8)
        for j, verb in enumerate(cap_2_verb):
            control_verb[j] = self.verb_2_idx[verb] + 1  # 0代表没有verb

        cls_seq = [self.classes[class_idx] for class_idx in classes_seq]
        selected_classes = [self.classes[np.argmax(det_cls_probs[i][1:]) + 1] for i in range(len(det_cls_probs))] # class从1开始，0可能表示没有关系
        width, height = self.img_shapes[str(id_image)]
        
        det_sequences_visual_all = np.zeros((self.fix_length, self.max_detections, det_features.shape[-1]))
        det_sequences_visual = np.zeros((self.fix_length, det_features.shape[-1]))
        det_sequences_word = np.zeros((self.fix_length, 300))
        det_sequences_position = np.zeros((self.fix_length, 4))

        det_sequences_sr = np.zeros((self.fix_length, 8))
        det_sequences_verb = np.zeros((self.fix_length, 8))
        idx_list = np.zeros((self.fix_length, 1))
        idx_list[idx_list==0] = -1

        cls_seq = cls_seq[:self.fix_length]  # 只保留前10个
        idx_list_ = np.array(cls_seq).argsort()
        idx_list[:len(idx_list_), :] = idx_list_[:, np.newaxis]

        cls_seq.sort()  # 按字母序排列class, 相当于打乱排序
        for j, cls in enumerate(cls_seq):
            cls_w = cls.split(',')[0].split(' ')[-1] # 当前class，为啥这么处理?
            if cls_w in self.vectors:
                det_sequences_word[j] = self.vectors[cls_w]
            seed_detections = [i for i, c in enumerate(selected_classes) if c == cls]  # 从detections里面找到class是选中class的detections序号
            det_ids = np.unique(seed_detections)
            det_sequences_visual_all[j, :len(det_ids)] = np.take(det_features, det_ids, axis=0)[:self.max_detections]  # 当前class所有region的feature
            det_sequences_visual[j] = det_features[det_ids[0]]  # 当前class第一个region的feature
            bbox = det_boxes[det_ids[0]]  # 当前class第一个region的框 (x1, y1, x2, y2)
            det_sequences_position[j, 0] = (bbox[2] - bbox[0] / 2) / width  # 中心横坐标
            det_sequences_position[j, 1] = (bbox[3] - bbox[1] / 2) / height  # 中心纵坐标
            det_sequences_position[j, 2] = (bbox[2] - bbox[0]) / width  # 宽
            det_sequences_position[j, 3] = (bbox[3] - bbox[1]) / height  # 长
            for k, sr in enumerate(loc_2_sr[int(idx_list[j][0])]):
                if k >= 8:
                    continue
                det_sequences_sr[j, k] = self.sr_2_idx[sr.split('-')[-1]]  # 0代表pad
                det_sequences_verb[j, k] = self.verb_2_idx[loc_2_verb[int(idx_list[j][0])][k]] + 1 # 0代表pad
            
        return det_sequences_word.astype(np.float32), det_sequences_visual.astype(np.float32), \
               det_sequences_position.astype(np.float32), det_sequences_visual_all.astype(np.float32), \
               det_sequences_verb.astype(np.float32), det_sequences_sr.astype(np.float32), control_verb.astype(np.float32), idx_list


class COCOControlSetField_Verb(RawField):
    def __init__(self, postprocessing=None, idx_vs_path=None, cap_classes_path=None, cap_verb_path=None, 
                verb_idx_path=None, detections_path=None, classes_path=None, img_shapes_path=None, 
                precomp_glove_path=None, vocab_path=None, idx_2_verb_og_path=None, verb_vob_path=None, 
                fix_length=20, max_detections=20, gt_verb=False):
        self.fix_length = fix_length
        self.detections_path = detections_path
        self.max_detections = max_detections
        self.gt_verb = gt_verb

        self.classes = ['__background__']
        with open(classes_path, 'r') as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())

        with open(precomp_glove_path, 'rb') as fp:
            self.vectors = pkl.load(fp)

        with open(img_shapes_path, 'r') as fp:
            self.img_shapes = json.load(fp)

        with open(cap_classes_path, 'r') as fp:
            self.cap_2_classes = json.load(fp)

        with open(idx_vs_path, 'r') as fp:
            self.idx_2_vs = json.load(fp)
        
        with open(verb_idx_path, 'r') as fp:
            self.verb_2_idx = json.load(fp)
        
        with open(cap_verb_path, 'r') as fp:
            self.cap_2_verb = json.load(fp)
        
        with open(vocab_path) as f:
            vocab_list = json.load(f)
        self.vocab_2_idx = {}
        for idx, vocab in enumerate(vocab_list):
            self.vocab_2_idx[vocab] = idx

        with open(idx_2_verb_og_path) as f:
            self.idx_2_v_og = json.load(f)

        with open(verb_vob_path) as f:
            self.verb_2_vob = json.load(f)

        self.sr_2_idx = {'ARG0': 1, 'ARG1': 2, 'ARG2': 3, 'ARG3': 4, 'ARG4': 5, 'ARG5': 6, 'LOC': 7, 'DIR': 8, 'GOL': 9, 
                        'MNR': 10, 'TMP': 11, 'EXT': 12, 'REC': 13, 'PRD': 14, 'PRP': 15, 'CAU': 16, 'DIS': 17, 'ADV': 18, 
                        'ADJ': 19, 'MOD': 20, 'NEG': 21, 'LVB': 22, 'PNC': 23, 'COM': 24, 'V': 25}

        super(COCOControlSetField_Verb, self).__init__(None, postprocessing)

    def preprocess(self, x):
        image = x[0][0]
        caption = x[0][1]
        id_image = int(image.split('/')[-1].split('_')[-1].split('.')[0])
        try:
            f = h5py.File(self.detections_path, 'r')
            det_cls_probs = f['%s_cls_prob' % id_image][()]
            det_features = f['%s_features' % id_image][()]
            det_boxes = f['%s_boxes' % id_image][()]
        except KeyError:
            warnings.warn('Could not find detections for %d' % id_image)
            det_cls_probs = np.random.rand(10, 2048)
            det_features = np.random.rand(10, 2048)
            det_boxes = np.random.rand(10, 4)

        idx_2_verb = self.idx_2_vs[str(id_image)][caption]['verb']
        idx_2_sr = self.idx_2_vs[str(id_image)][caption]['sr']
        idx_2_v_og = self.idx_2_v_og[str(id_image)][caption]
        cap_2_verb = self.cap_2_verb[str(id_image)][caption]
        cls_seq = self.cap_2_classes[str(id_image)][caption]

        selected_classes = [self.classes[np.argmax(det_cls_probs[i][1:]) + 1] for i in range(len(det_cls_probs))] # class从1开始，0可能表示没有关系
        width, height = self.img_shapes[str(id_image)]

        # pooled_feat
        pooled_feat = np.mean(det_features, axis=0)
        det_sequences_visual_all = np.zeros((self.fix_length, self.max_detections, det_features.shape[-1]))
        det_sequences_visual = np.zeros((self.fix_length, det_features.shape[-1]))
        det_sequences_word = np.zeros((self.fix_length, 300))
        det_sequences_position = np.zeros((self.fix_length, 4))

        det_sequences_sr = np.zeros((self.fix_length, 8))
        det_sequences_verb = np.zeros((self.fix_length, 8))
        gt_det_sequences_sr = np.zeros((self.fix_length, 8))
        gt_det_sequences_verb = np.zeros((self.fix_length, 8))
        
        # 不是verb的位置为0，是verb的位置为verb的index
        verb_list = np.zeros((self.fix_length, 1))
        verb_list[verb_list==0] = -1
        verb_list_og = np.zeros((self.fix_length, 1))
        verb_list_og[verb_list_og==0] = -1
        idx_list = np.zeros((self.fix_length, 1))
        idx_list[idx_list==0] = -1

        control_verb = np.zeros(8)
        for j, verb in enumerate(cap_2_verb):
            control_verb[j] = self.verb_2_idx[verb] + 1  # 0代表没有verb

        cls_seq = cls_seq[:self.fix_length]  # 只保留前10个
        for j, cls in enumerate(cls_seq):
            for k, sr in enumerate(idx_2_sr[j]):
                if k >= 8:
                    continue
                gt_det_sequences_sr[j, k] = self.sr_2_idx[sr.split('-')[-1]]  # 0代表pad
                gt_det_sequences_verb[j, k] = self.verb_2_idx[idx_2_verb[j][k]] + 1 # 0代表pad

        # 随机排序
        idx_rank = [x for x in range(self.fix_length)]
        rank_use = list(zip(cls_seq, idx_rank))
        random.shuffle(rank_use)
        cls_seq, idx_list_ = zip(*rank_use)
        idx_list_ = np.array(idx_list_)
        idx_list[:len(idx_list_), :] = idx_list_[:, np.newaxis]

        for j, cls in enumerate(cls_seq):
            if cls == '_':
                continue
            if cls != 'verb':  # class为verb不用feature排序
                cls_w = cls.split(',')[0].split(' ')[-1] # 当前class，为啥这么处理?
                if cls_w in self.vectors:
                    det_sequences_word[j] = self.vectors[cls_w]
                seed_detections = [i for i, c in enumerate(selected_classes) if c == cls]  # 从detections里面找到class是选中class的detections序号
                det_ids = np.unique(seed_detections)
                
                det_sequences_visual_all[j, :len(det_ids)] = np.take(det_features, det_ids, axis=0)[:self.max_detections]  # 当前class所有region的feature
                det_sequences_visual[j] = det_features[det_ids[0]]  # 当前class第一个region的feature
                bbox = det_boxes[det_ids[0]]  # 当前class第一个region的框 (x1, y1, x2, y2)
                det_sequences_position[j, 0] = (bbox[2] - bbox[0] / 2) / width  # 中心横坐标
                det_sequences_position[j, 1] = (bbox[3] - bbox[1] / 2) / height  # 中心纵坐标
                det_sequences_position[j, 2] = (bbox[2] - bbox[0]) / width  # 宽
                det_sequences_position[j, 3] = (bbox[3] - bbox[1]) / height  # 长
            else:
                det_sequences_visual_all[j, 0] = pooled_feat
                # 后面这串是verb_idx
                if idx_2_verb[int(idx_list[j][0])] != []:
                    if idx_2_v_og[int(idx_list[j][0])][0] in self.vocab_2_idx:
                        verb_list_og[j, :] = self.vocab_2_idx[idx_2_v_og[int(idx_list[j][0])][0]]
                    else:
                        verb_list_og[j, :] = 0
                    verb_list[j, :] = self.verb_2_idx[idx_2_verb[int(idx_list[j][0])][0]] + 1
            
            for k, sr in enumerate(idx_2_sr[int(idx_list[j][0])]):
                if k >= 8:
                    continue
                det_sequences_sr[j, k] = self.sr_2_idx[sr.split('-')[-1]]  # 0代表pad
                det_sequences_verb[j, k] = self.verb_2_idx[idx_2_verb[int(idx_list[j][0])][k]] + 1 # 0代表pad

        if self.gt_verb:
            return det_sequences_word.astype(np.float32), det_sequences_visual.astype(np.float32), \
               det_sequences_position.astype(np.float32), det_sequences_visual_all.astype(np.float32), \
               det_sequences_verb.astype(np.float32), det_sequences_sr.astype(np.float32), control_verb.astype(np.float32), \
               gt_det_sequences_verb.astype(np.float32), gt_det_sequences_sr.astype(np.float32), idx_list, verb_list_og
        else:
            return det_sequences_word.astype(np.float32), det_sequences_visual.astype(np.float32), \
                det_sequences_position.astype(np.float32), det_sequences_visual_all.astype(np.float32), \
                det_sequences_verb.astype(np.float32), det_sequences_sr.astype(np.float32), control_verb.astype(np.float32), \
                gt_det_sequences_verb.astype(np.float32), gt_det_sequences_sr.astype(np.float32), idx_list, verb_list


class COCODetSetField_Verb(RawField):
    def __init__(self, postprocessing=None, cls_seq_path=None, vocab_path=None, 
                vlem_2_v_og_path=None, verb_idx_path=None, detections_path=None, classes_path=None, 
                img_shapes_path=None, precomp_glove_path=None, fix_length=20, max_detections=20, 
                gt_verb=False):
        self.fix_length = fix_length
        self.detections_path = detections_path
        self.max_detections = max_detections
        self.gt_verb = gt_verb

        self.classes = ['__background__']

        with open(classes_path, 'r') as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())

        with open(precomp_glove_path, 'rb') as fp:
            self.vectors = pkl.load(fp)

        with open(img_shapes_path, 'r') as fp:
            self.img_shapes = json.load(fp)

        with open(verb_idx_path, 'r') as fp:
            self.verb_2_idx = json.load(fp)
        
        with open(vocab_path, 'r') as f:
            vocab_list = json.load(f)
        self.vocab_2_idx = {}
        for idx, vocab in enumerate(vocab_list):
            self.vocab_2_idx[vocab] = idx

        # match verb_idx to vocab_idx
        with open(vlem_2_v_og_path) as f:
            self.vlem_2_verb = json.load(f)

        with open(cls_seq_path) as f:
            self.img_cap_v_2_class = json.load(f)

        self.sr_2_idx = {'ARG0': 1, 'ARG1': 2, 'ARG2': 3, 'ARG3': 4, 'ARG4': 5, 'ARG5': 6, 'LOC': 7, 'DIR': 8, 'GOL': 9, 
                        'MNR': 10, 'TMP': 11, 'EXT': 12, 'REC': 13, 'PRD': 14, 'PRP': 15, 'CAU': 16, 'DIS': 17, 'ADV': 18,
                        'ADJ': 19, 'MOD': 20, 'NEG': 21, 'LVB': 22, 'PNC': 23, 'COM': 24, 'V': 25}  # END is for predict END of the output

        super(COCODetSetField_Verb, self).__init__(None, postprocessing)

    def preprocess(self, x, rand=True):
        image = x[0][0]
        caption = x[0][1]
        id_image = int(image.split('/')[-1].split('_')[-1].split('.')[0])
        try:
            f = h5py.File(self.detections_path, 'r')
            det_cls_probs = f['%s_cls_prob' % id_image][()]
            det_features = f['%s_features' % id_image][()]
            det_boxes = f['%s_boxes' % id_image][()]
        except KeyError:
            warnings.warn('Could not find detections for %d' % id_image)
            det_cls_probs = np.random.rand(10, 2048)
            det_features = np.random.rand(10, 2048)
            det_boxes = np.random.rand(10, 4)

        # get the det region info.
        v_2_class = self.img_cap_v_2_class[str(id_image)][caption]
        
        classes_seq = []
        loc_2_verb = {}
        loc_2_sr = {}
        loc = 0
        
        cap_2_verb = []
        vlem_2_verb = self.vlem_2_verb[str(id_image)][caption]
        for verb in v_2_class:
            for sr in v_2_class[verb]:
                for class_idx in v_2_class[verb][sr]:
                    if verb not in cap_2_verb:
                        cap_2_verb.append(verb)
                    classes_seq.append(class_idx)  # the class idx (1600分类中的某一类)

        control_verb = np.zeros(8)
        for j, verb in enumerate(cap_2_verb):
            control_verb[j] = self.verb_2_idx[verb] + 1  # 0代表没有verb

        # append the cls_seq with "verb" in the beginning
        cls_seq = []
        for verb in cap_2_verb:
            cls_seq.append('verb')
            loc_2_verb.setdefault(loc, []).append(verb)
            loc_2_sr.setdefault(loc, []).append('V')
            loc += 1
        cls_seq += [self.classes[class_idx] for class_idx in classes_seq]

        for verb in v_2_class:
            for sr in v_2_class[verb]:
                for class_idx in v_2_class[verb][sr]:
                    loc_2_verb.setdefault(loc, []).append(verb)
                    loc_2_sr.setdefault(loc, []).append(sr)
                    loc += 1

        selected_classes = [self.classes[np.argmax(det_cls_probs[i][1:]) + 1] for i in range(len(det_cls_probs))] # class从1开始，0可能表示没有关系
        width, height = self.img_shapes[str(id_image)]

        # pooled_feat
        pooled_feat = np.mean(det_features, axis=0)
        
        det_sequences_visual_all = np.zeros((self.fix_length, self.max_detections, det_features.shape[-1]))
        det_sequences_visual = np.zeros((self.fix_length, det_features.shape[-1]))
        det_sequences_word = np.zeros((self.fix_length, 300))
        det_sequences_position = np.zeros((self.fix_length, 4))

        det_sequences_sr = np.zeros((self.fix_length, 8))
        det_sequences_verb = np.zeros((self.fix_length, 8))
        idx_list = np.zeros((self.fix_length, 1))
        idx_list[idx_list==0] = -1

        cls_seq = cls_seq[:self.fix_length]  # 只保留前10个

        # 不是verb的位置为0，是verb的位置为verb的index
        verb_list = np.zeros((self.fix_length, 1))
        verb_list[verb_list==0] = -1

        # 随机排序
        idx_rank = [x for x in range(self.fix_length)]
        rank_use = list(zip(cls_seq, idx_rank))
        random.shuffle(rank_use)
        cls_seq, idx_list_ = zip(*rank_use)
        idx_list_ = np.array(idx_list_)
        idx_list[:len(idx_list_), :] = idx_list_[:, np.newaxis]

        for j, cls in enumerate(cls_seq):
            if cls != 'verb':
                cls_w = cls.split(',')[0].split(' ')[-1] # 当前class，为啥这么处理?
                if cls_w in self.vectors:
                    det_sequences_word[j] = self.vectors[cls_w]
                seed_detections = [i for i, c in enumerate(selected_classes) if c == cls]  # 从detections里面找到class是选中class的detections序号
                if seed_detections != []:
                    det_ids = np.unique(seed_detections)
                else:
                    det_ids = np.array([]).astype(np.int64)
                if len(det_ids) == 0:
                    det_ids = [1]
                    print(caption)
                det_sequences_visual_all[j, :len(det_ids)] = np.take(det_features, det_ids, axis=0)[:self.max_detections]  # 当前class所有region的feature
                det_sequences_visual[j] = det_features[det_ids[0]]  # 当前class第一个region的feature
                bbox = det_boxes[det_ids[0]]  # 当前class第一个region的框 (x1, y1, x2, y2)
                det_sequences_position[j, 0] = (bbox[2] - bbox[0] / 2) / width  # 中心横坐标
                det_sequences_position[j, 1] = (bbox[3] - bbox[1] / 2) / height  # 中心纵坐标
                det_sequences_position[j, 2] = (bbox[2] - bbox[0]) / width  # 宽
                det_sequences_position[j, 3] = (bbox[3] - bbox[1]) / height  # 长
            else:
                det_sequences_visual_all[j, 0] = pooled_feat
                # 后面这串是verb_idx
                if loc_2_verb[int(idx_list[j][0])] != []:
                    if self.gt_verb is False:
                        verb_list[j, :] = self.verb_2_idx[loc_2_verb[int(idx_list[j][0])][0]] + 1
                    else:
                        for v_lem, verb_og in vlem_2_verb:
                            if v_lem == loc_2_verb[int(idx_list[j][0])][0]:
                                if verb_og in self.vocab_2_idx:
                                    verb_list[j, :] = self.vocab_2_idx[verb_og]
                                else:
                                    verb_list[j, :] = 0
                                break

            for k, sr in enumerate(loc_2_sr[int(idx_list[j][0])]):
                if k >= 8:
                    continue
                det_sequences_sr[j, k] = self.sr_2_idx[sr.split('-')[-1]]  # 0代表pad
                det_sequences_verb[j, k] = self.verb_2_idx[loc_2_verb[int(idx_list[j][0])][k]] + 1 # 0代表pad
            
        return det_sequences_word.astype(np.float32), det_sequences_visual.astype(np.float32), \
               det_sequences_position.astype(np.float32), det_sequences_visual_all.astype(np.float32), \
               det_sequences_verb.astype(np.float32), det_sequences_sr.astype(np.float32), \
               control_verb.astype(np.float32), idx_list, verb_list


# Flickr
class FlickrDetectionField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, detections_path=None, diverse=False):
        self.max_detections = 100
        self.detections_path = detections_path
        self.diverse = diverse

        super(FlickrDetectionField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x, avoid_precomp=False):
        image_id = int(x.split('/')[-1].split('.')[0])
        try:
            precomp_data = h5py.File(self.detections_path, 'r')['%d_features' % image_id][()]
        except KeyError:
            warnings.warn('Could not find detections for %d' % image_id)
            precomp_data = np.random.rand(10, 2048)

        delta = self.max_detections - precomp_data.shape[0]
        if delta > 0:
            precomp_data = np.concatenate([precomp_data, np.zeros((delta, precomp_data.shape[1]))], axis=0)
        elif delta < 0:
            precomp_data = precomp_data[:self.max_detections]

        if self.diverse:
            return precomp_data.astype(np.float32), image_id
        return precomp_data.astype(np.float32)


# Flickr detection field
class FlickrControlSetField(RawField):
    def __init__(self, postprocessing=None, idx_vs_path=None, cap_verb_path=None, cap_classes_path=None, 
                verb_idx_path=None, detections_path=None, classes_path=None, img_shapes_path=None, 
                precomp_glove_path=None, fix_length=20, max_detections=20, visual=True):
        self.fix_length = fix_length
        self.detections_path = detections_path
        self.max_detections = max_detections
        self.visual = visual

        self.classes = ['__background__']
        with open(classes_path, 'r') as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())

        with open(precomp_glove_path, 'rb') as fp:
            self.vectors = pkl.load(fp)

        with open(img_shapes_path, 'r') as fp:
            self.img_shapes = json.load(fp)

        with open(verb_idx_path) as f:
            self.flickr_verb_idx = json.load(f)

        with open(idx_vs_path) as f:
            self.idx_2_vs = json.load(f)
        
        with open(cap_verb_path) as f:
            self.cap_2_verb = json.load(f)

        with open(cap_classes_path) as f:
            self.cap_2_classes = json.load(f)
        
        self.sr_2_idx = {'ARG0': 1, 'ARG1': 2, 'ARG2': 3, 'ARG3': 4, 'ARG4': 5, 'ARG5': 6, 'LOC': 7, 'DIR': 8, 'GOL': 9, 
                        'MNR': 10, 'TMP': 11, 'EXT': 12, 'REC': 13, 'PRD': 14, 'PRP': 15, 'CAU': 16, 'DIS': 17, 'ADV': 18, 
                        'ADJ': 19, 'MOD': 20, 'NEG': 21, 'LVB': 22, 'PNC': 23, 'COM': 24, 'V': 25}

        super(FlickrControlSetField, self).__init__(None, postprocessing)

    @staticmethod
    def _bb_intersection_over_union(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / (boxAArea + boxBArea - interArea)
        return iou

    def preprocess(self, x):
        image = x[0][0]
        caption = x[0][1]
        gt_bboxes = x[1]

        id_image = image.split('/')[-1].split('.')[0]
        if self.visual:
            try:
                f = h5py.File(self.detections_path, 'r')
                det_cls_probs = f['%s_cls_prob' % id_image][()]
                det_features = f['%s_features' % id_image][()]
                det_bboxes = f['%s_boxes' % id_image][()]
            except KeyError:
                warnings.warn('Could not find detections for %d' % id_image)
                det_cls_probs = np.random.rand(10, 2048)
                det_features = np.random.rand(10, 2048)
                det_bboxes = np.random.rand(10, 4)

        idx_2_verb = self.idx_2_vs[id_image][caption]['verb'] # verb&num
        idx_2_sr = self.idx_2_vs[id_image][caption]['sr'] # sr_idx
        cap_2_verb = self.cap_2_verb[id_image][caption] # verb

        cls_seq = self.cap_2_classes[id_image][caption]
        cls_seq = [x-1 for x in cls_seq] # verb由-1变为了-2

        if self.visual:
            selected_classes = [self.classes[np.argmax(det_cls_probs[i][1:]) + 1] for i in range(len(det_cls_probs))]
            width, height = self.img_shapes[str(id_image)]

            # visual feature
            pooled_feat = np.mean(det_features, axis=0)
            det_sequences_visual_all = np.zeros((self.fix_length, self.max_detections, det_features.shape[-1]))
            det_sequences_visual = np.zeros((self.fix_length, det_features.shape[-1]))
            det_sequences_word = np.zeros((self.fix_length, 300))
            det_sequences_position = np.zeros((self.fix_length, 4))

        # semantic role feature
        det_sequences_sr = np.zeros((self.fix_length, 8))
        det_sequences_verb = np.zeros((self.fix_length, 8))
        gt_det_sequences_sr = np.zeros((self.fix_length, 8))
        gt_det_sequences_verb = np.zeros((self.fix_length, 8))

        control_verb = np.zeros(8)
        for j, verb in enumerate(cap_2_verb):
            if j >= 8:
                continue
            control_verb[j] = self.flickr_verb_idx[verb.split('_')[0]] + 1 \
                                + 10000 * int(verb.split('_')[-1]) # 0代表没有verb

        # 不是verb的位置为0，是verb的位置为verb的index
        idx_list = np.zeros((self.fix_length, 1))
        idx_list[idx_list==0] = -1

        cls_seq = cls_seq[:self.fix_length] # 只保留前10个
        for j, cls_ in enumerate(cls_seq):
            for k, sr in enumerate(idx_2_sr[j]):
                if idx_2_verb[j][k] in cap_2_verb:
                    gt_det_sequences_sr[j, k] = sr # 0代表pad
                    gt_det_sequences_verb[j, k] = self.flickr_verb_idx[idx_2_verb[j][k].split('_')[0]] + 1 \
                                                    + 10000 * int(idx_2_verb[j][k].split('_')[-1]) # 0代表没有verb
        
        # 随机排序
        idx_rank = [x for x in range(self.fix_length)]
        rank_use = list(zip(cls_seq, idx_rank))
        random.shuffle(rank_use)
        cls_seq, idx_list_ = zip(*rank_use)
        idx_list_ = np.array(idx_list_)
        idx_list[:len(idx_list_), :] = idx_list_[:, np.newaxis]

        for j, cls in enumerate(cls_seq):
            if self.visual:
                id_boxes = []
                for k, bbox in enumerate(gt_bboxes[cls]):
                    id_bbox = -1
                    iou_max = 0
                    for ii, det_bbox in enumerate(det_bboxes):
                        iou = self._bb_intersection_over_union(bbox, det_bbox)
                        if iou_max < iou:
                            id_bbox = ii
                            iou_max = iou
                    id_boxes.append(id_bbox)

                id_boxes.sort()

                cls_w = selected_classes[id_boxes[0]].split(',')[0].split(' ')[-1]
                if cls_w in self.vectors:
                    det_sequences_word[j] = self.vectors[cls_w]

                det_sequences_visual_all[j, :len(id_boxes)] = np.take(det_features, id_boxes, axis=0)[:self.max_detections]
                det_sequences_visual[j] = det_features[id_boxes[0]]
                
                bbox = det_bboxes[id_boxes[0]]
                det_sequences_position[j, 0] = (bbox[2] - bbox[0] / 2) / width
                det_sequences_position[j, 1] = (bbox[3] - bbox[1] / 2) / height
                det_sequences_position[j, 2] = (bbox[2] - bbox[0]) / width
                det_sequences_position[j, 3] = (bbox[3] - bbox[1]) / height
        
            for k, sr in enumerate(idx_2_sr[int(idx_list[j][0])]):
                if idx_2_verb[int(idx_list[j][0])][k] in cap_2_verb:
                    det_sequences_sr[j, k] = sr
                    det_sequences_verb[j, k] = self.flickr_verb_idx[idx_2_verb[int(idx_list[j][0])][k].split('_')[0]] + 1 \
                                                + 10000 * int(idx_2_verb[int(idx_list[j][0])][k].split('_')[-1])
                                                
        if self.visual:
            return det_sequences_word.astype(np.float32), det_sequences_visual.astype(np.float32), \
                det_sequences_position.astype(np.float32), det_sequences_visual_all.astype(np.float32), \
                det_sequences_verb.astype(np.float32), det_sequences_sr.astype(np.float32), control_verb.astype(np.float32), \
                gt_det_sequences_verb.astype(np.float32), gt_det_sequences_sr.astype(np.float32), idx_list
        else:
            return det_sequences_verb.astype(np.float32), det_sequences_sr.astype(np.float32), control_verb.astype(np.float32), \
                gt_det_sequences_verb.astype(np.float32), gt_det_sequences_sr.astype(np.float32), idx_list


class FlickrDetSetField(RawField):
    def __init__(self, postprocessing=None, verb_idx_path=None, verb_vob_path=None, idbox_seq_path=None, 
                detections_path=None, classes_path=None, img_shapes_path=None, precomp_glove_path=None, 
                fix_length=20, max_detections=20, visual=True):
        self.fix_length = fix_length
        self.detections_path = detections_path
        self.max_detections = max_detections
        self.visual = visual

        self.classes = ['__background__']
        with open(classes_path, 'r') as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())

        with open(precomp_glove_path, 'rb') as fp:
            self.vectors = pkl.load(fp)

        with open(img_shapes_path, 'r') as fp:
            self.img_shapes = json.load(fp)

        with open(verb_idx_path) as f:
            self.flickr_verb_idx = json.load(f)
        
        with open(verb_vob_path) as f:
            self.verb_2_vob = json.load(f)
        
        with open(idbox_seq_path) as f:
            self.img_cap_v_2_idbox = json.load(f)
        
        self.sr_2_idx = {'ARG0': 1, 'ARG1': 2, 'ARG2': 3, 'ARG3': 4, 'ARG4': 5, 'ARG5': 6, 'LOC': 7, 'DIR': 8, 'GOL': 9, 
                        'MNR': 10, 'TMP': 11, 'EXT': 12, 'REC': 13, 'PRD': 14, 'PRP': 15, 'CAU': 16, 'DIS': 17, 'ADV': 18, 
                        'ADJ': 19, 'MOD': 20, 'NEG': 21, 'LVB': 22, 'PNC': 23, 'COM': 24, 'V': 25}

        super(FlickrDetSetField, self).__init__(None, postprocessing)

    def preprocess(self, x):
        image = x[0][0]
        caption = x[0][1]
        gt_bboxes = x[1]
        det_classes = x[2]

        id_image = image.split('/')[-1].split('.')[0]
        try:
            f = h5py.File(self.detections_path, 'r')
            det_cls_probs = f['%s_cls_prob' % id_image][()]
            det_features = f['%s_features' % id_image][()]
            det_bboxes = f['%s_boxes' % id_image][()]
        except KeyError:
            warnings.warn('Could not find detections for %d' % id_image)
            det_cls_probs = np.random.rand(10, 2048)
            det_features = np.random.rand(10, 2048)
            det_bboxes = np.random.rand(10, 4)

        v_2_class = self.img_cap_v_2_idbox[id_image][caption]

        loc_2_verb = {}
        loc_2_sr = {}
        loc = 0
        
        idbox_seq = {}

        cap_2_verb = []
        for verb in v_2_class:
            for sr in v_2_class[verb]:
                for id_box in v_2_class[verb][sr]:
                    if verb not in cap_2_verb:
                        cap_2_verb.append(verb)
                    if id_box not in idbox_seq:
                        idbox_seq[id_box] = loc
                        loc += 1

        control_verb = np.zeros(8)
        for j, verb in enumerate(cap_2_verb):
            if j >= 8:
                continue
            control_verb[j] = self.flickr_verb_idx[verb.split('_')[0]] + 1  # 0代表没有verb
        
        for verb in v_2_class:
            for sr in v_2_class[verb]:
                for id_box in v_2_class[verb][sr]:
                    loc_ = idbox_seq[id_box]
                    loc_2_verb.setdefault(loc_, []).append(verb)
                    loc_2_sr.setdefault(loc_, []).append(sr)

        width, height = self.img_shapes[str(id_image)]
        
        # visual feature
        selected_classes = [self.classes[np.argmax(det_cls_probs[i][1:]) + 1] for i in range(len(det_cls_probs))]
        det_sequences_visual_all = np.zeros((self.fix_length, self.max_detections, det_features.shape[-1]))
        det_sequences_visual = np.zeros((self.fix_length, det_features.shape[-1]))
        det_sequences_word = np.zeros((self.fix_length, 300))
        det_sequences_position = np.zeros((self.fix_length, 4))

        # semantic role feature
        det_sequences_sr = np.zeros((self.fix_length, 8))
        det_sequences_verb = np.zeros((self.fix_length, 8))

        for j, idbox in enumerate(idbox_seq):
            if j == 10: break
            det_sequences_visual_all[j, 0] = det_features[idbox]  # 当前class所有region的feature
            det_sequences_visual[j] = det_features[idbox]  # 当前class第一个region的feature
            cls_w = selected_classes[idbox].split(',')[0].split(' ')[-1]
            if cls_w in self.vectors:
                det_sequences_word[j] = self.vectors[cls_w]
            bbox = det_bboxes[idbox]  # 当前class第一个region的框 (x1, y1, x2, y2)
            det_sequences_position[j, 0] = (bbox[2] - bbox[0] / 2) / width  # 中心横坐标
            det_sequences_position[j, 1] = (bbox[3] - bbox[1] / 2) / height  # 中心纵坐标
            det_sequences_position[j, 2] = (bbox[2] - bbox[0]) / width  # 宽
            det_sequences_position[j, 3] = (bbox[3] - bbox[1]) / height  # 长

            for k, sr in enumerate(loc_2_sr[j]):
                if k >= 8:
                    continue
                det_sequences_sr[j, k] = sr  # 0代表pad
                det_sequences_verb[j, k] = self.flickr_verb_idx[loc_2_verb[j][k].split('_')[0]] + 1 # 0代表pad
        
        return det_sequences_word.astype(np.float32), det_sequences_visual.astype(np.float32), \
            det_sequences_position.astype(np.float32), det_sequences_visual_all.astype(np.float32), \
            det_sequences_verb.astype(np.float32), det_sequences_sr.astype(np.float32), control_verb.astype(np.float32)


class FlickrControlSetField_Verb(RawField):
    def __init__(self, postprocessing=None, idx_vs_path=None, cap_verb_path=None, cap_classes_path=None, 
                verb_idx_path=None, idx_v_og_path=None, vocab_list_path=None, detections_path=None, 
                classes_path=None, img_shapes_path=None, precomp_glove_path=None, fix_length=20, 
                max_detections=20, visual=True, gt_verb=False):
        self.fix_length = fix_length
        self.detections_path = detections_path
        self.max_detections = max_detections
        self.visual = visual
        self.gt_verb = gt_verb

        self.classes = ['__background__']
        with open(classes_path, 'r') as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())

        with open(precomp_glove_path, 'rb') as fp:
            self.vectors = pkl.load(fp)

        with open(img_shapes_path, 'r') as fp:
            self.img_shapes = json.load(fp)

        with open(idx_vs_path) as f:
            self.idx_2_vs = json.load(f)
        
        with open(cap_verb_path) as f:
            self.cap_2_verb = json.load(f)

        with open(cap_classes_path) as f:
            self.cap_2_classes = json.load(f)

        with open(verb_idx_path) as f:
            self.flickr_verb_idx = json.load(f)
        
        with open(idx_v_og_path) as f:
            self.idx_2_v_og = json.load(f)
        
        with open(vocab_list_path) as f:
            vocab_list = json.load(f)
        self.vocab_2_idx = {}
        for idx, vocab in enumerate(vocab_list):
            self.vocab_2_idx[vocab] = idx
        
        self.sr_2_idx = {'ARG0': 1, 'ARG1': 2, 'ARG2': 3, 'ARG3': 4, 'ARG4': 5, 'ARG5': 6, 'LOC': 7, 'DIR': 8, 'GOL': 9, 
                        'MNR': 10, 'TMP': 11, 'EXT': 12, 'REC': 13, 'PRD': 14, 'PRP': 15, 'CAU': 16, 'DIS': 17, 'ADV': 18, 
                        'ADJ': 19, 'MOD': 20, 'NEG': 21, 'LVB': 22, 'PNC': 23, 'COM': 24, 'V': 25}

        super(FlickrControlSetField_Verb, self).__init__(None, postprocessing)

    @staticmethod
    def _bb_intersection_over_union(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / (boxAArea + boxBArea - interArea)
        return iou

    def preprocess(self, x):
        image = x[0][0]
        caption = x[0][1]
        gt_bboxes = x[1]

        id_image = image.split('/')[-1].split('.')[0]
        if self.visual:
            try:
                f = h5py.File(self.detections_path, 'r')
                det_cls_probs = f['%s_cls_prob' % id_image][()]
                det_features = f['%s_features' % id_image][()]
                det_bboxes = f['%s_boxes' % id_image][()]
            except KeyError:
                warnings.warn('Could not find detections for %d' % id_image)
                det_cls_probs = np.random.rand(10, 2048)
                det_features = np.random.rand(10, 2048)
                det_bboxes = np.random.rand(10, 4)

        idx_2_verb = self.idx_2_vs[id_image][caption]['verb'] # verb&num
        idx_2_v_og = self.idx_2_v_og[id_image][caption]
        idx_2_sr = self.idx_2_vs[id_image][caption]['sr'] # sr_idx
        cap_2_verb = self.cap_2_verb[id_image][caption] # verb

        cls_seq = self.cap_2_classes[id_image][caption]
        cls_seq = [x-1 for x in cls_seq] # verb由-1变为了-2

        if self.visual:
            selected_classes = [self.classes[np.argmax(det_cls_probs[i][1:]) + 1] for i in range(len(det_cls_probs))]
            width, height = self.img_shapes[str(id_image)]

            # visual feature
            pooled_feat = np.mean(det_features, axis=0)
            det_sequences_visual_all = np.zeros((self.fix_length, self.max_detections, det_features.shape[-1]))
            det_sequences_visual = np.zeros((self.fix_length, det_features.shape[-1]))
            det_sequences_word = np.zeros((self.fix_length, 300))
            det_sequences_position = np.zeros((self.fix_length, 4))

        # semantic role feature
        det_sequences_sr = np.zeros((self.fix_length, 8))
        det_sequences_verb = np.zeros((self.fix_length, 8))
        gt_det_sequences_sr = np.zeros((self.fix_length, 8))
        gt_det_sequences_verb = np.zeros((self.fix_length, 8))

        control_verb = np.zeros(8)
        for j, verb in enumerate(cap_2_verb):
            if j >= 8:
                continue
            control_verb[j] = self.flickr_verb_idx[verb.split('_')[0]] + 1 \
                                + 10000 * int(verb.split('_')[-1]) # 0代表没有verb

        # 不是verb的位置为0，是verb的位置为verb的index
        verb_list = np.zeros((self.fix_length, 1))
        verb_list[verb_list==0] = -1
        verb_list_og = np.zeros((self.fix_length, 1))
        verb_list_og[verb_list_og==0] = -1
        idx_list = np.zeros((self.fix_length, 1))
        idx_list[idx_list==0] = -1

        cls_seq = cls_seq[:self.fix_length] # 只保留前10个
        for j, cls_ in enumerate(cls_seq):
            for k, sr in enumerate(idx_2_sr[j]):
                if idx_2_verb[j][k] in cap_2_verb:
                    gt_det_sequences_sr[j, k] = sr # 0代表pad
                    gt_det_sequences_verb[j, k] = self.flickr_verb_idx[idx_2_verb[j][k].split('_')[0]] + 1 \
                                                    + 10000 * int(idx_2_verb[j][k].split('_')[-1]) # 0代表没有verb
        
        # 随机排序
        idx_rank = [x for x in range(self.fix_length)]
        rank_use = list(zip(cls_seq, idx_rank))
        random.shuffle(rank_use)
        cls_seq, idx_list_ = zip(*rank_use)
        idx_list_ = np.array(idx_list_)
        idx_list[:len(idx_list_), :] = idx_list_[:, np.newaxis]

        for j, cls in enumerate(cls_seq):
            if self.visual:
                if cls >= 0:
                    iou_max_max = 0
                    only_box = -1
                    id_boxes = []
                    for k, bbox in enumerate(gt_bboxes[cls]):
                        id_bbox = -1
                        iou_max = 0
                        # 只在前56个里面找iou最大的
                        for ii, det_bbox in enumerate(det_bboxes):
                            iou = self._bb_intersection_over_union(bbox, det_bbox)
                            if iou_max < iou:
                                id_bbox = ii
                                iou_max = iou
                        if iou_max_max < iou_max:
                            only_box = id_bbox
                            iou_max_max = iou_max
                        id_boxes.append(id_bbox)
                    id_boxes.sort()

                    det_sequences_visual_all[j, 0] = det_features[only_box]
                    det_sequences_visual[j] = det_features[only_box]
                    bbox = det_bboxes[only_box]
                    det_sequences_position[j, 0] = (bbox[2] - bbox[0] / 2) / width
                    det_sequences_position[j, 1] = (bbox[3] - bbox[1] / 2) / height
                    det_sequences_position[j, 2] = (bbox[2] - bbox[0]) / width
                    det_sequences_position[j, 3] = (bbox[3] - bbox[1]) / height
                else:
                    det_sequences_visual_all[j, 0] = pooled_feat
                    if idx_2_verb[int(idx_list[j][0])] != [] and idx_2_verb[int(idx_list[j][0])][0].split('_')[0] in self.flickr_verb_idx:
                        verb_list[j, :] = \
                            self.flickr_verb_idx[idx_2_verb[int(idx_list[j][0])][0].split('_')[0]] + 1
                        if idx_2_v_og[int(idx_list[j][0])][0] in self.vocab_2_idx:
                            verb_list_og[j, :] = \
                                self.vocab_2_idx[idx_2_v_og[int(idx_list[j][0])][0]]
                        else:
                            verb_list_og[j, :] = 0
        
            for k, sr in enumerate(idx_2_sr[int(idx_list[j][0])]):
                if idx_2_verb[int(idx_list[j][0])][k] in cap_2_verb:
                    det_sequences_sr[j, k] = sr
                    det_sequences_verb[j, k] = self.flickr_verb_idx[idx_2_verb[int(idx_list[j][0])][k].split('_')[0]] + 1 \
                                                + 10000 * int(idx_2_verb[int(idx_list[j][0])][k].split('_')[-1])

        if self.gt_verb:
            if self.visual:
                return det_sequences_word.astype(np.float32), det_sequences_visual.astype(np.float32), \
                    det_sequences_position.astype(np.float32), det_sequences_visual_all.astype(np.float32), \
                    det_sequences_verb.astype(np.float32), det_sequences_sr.astype(np.float32), control_verb.astype(np.float32), \
                    gt_det_sequences_verb.astype(np.float32), gt_det_sequences_sr.astype(np.float32), idx_list, verb_list_og
            else:
                return det_sequences_verb.astype(np.float32), det_sequences_sr.astype(np.float32), control_verb.astype(np.float32), \
                    gt_det_sequences_verb.astype(np.float32), gt_det_sequences_sr.astype(np.float32), idx_list, verb_list_og
        else:
            if self.visual:
                return det_sequences_word.astype(np.float32), det_sequences_visual.astype(np.float32), \
                    det_sequences_position.astype(np.float32), det_sequences_visual_all.astype(np.float32), \
                    det_sequences_verb.astype(np.float32), det_sequences_sr.astype(np.float32), control_verb.astype(np.float32), \
                    gt_det_sequences_verb.astype(np.float32), gt_det_sequences_sr.astype(np.float32), idx_list, verb_list
            else:
                return det_sequences_verb.astype(np.float32), det_sequences_sr.astype(np.float32), control_verb.astype(np.float32), \
                    gt_det_sequences_verb.astype(np.float32), gt_det_sequences_sr.astype(np.float32), idx_list, verb_list


class FlickrDetSetField_Verb(RawField):
    def __init__(self, postprocessing=None, verb_idx_path=None, verb_vob_path=None, idbox_seq_path=None, 
                vocab_list_path=None, vlem_2_verb_og_path=None, detections_path=None, classes_path=None, 
                img_shapes_path=None, precomp_glove_path=None, fix_length=20, max_detections=20, 
                visual=True, gt_verb=False):
        self.fix_length = fix_length
        self.detections_path = detections_path
        self.max_detections = max_detections
        self.visual = visual
        self.gt_verb = gt_verb

        self.classes = ['__background__']
        with open(classes_path, 'r') as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())

        with open(precomp_glove_path, 'rb') as fp:
            self.vectors = pkl.load(fp)

        with open(img_shapes_path, 'r') as fp:
            self.img_shapes = json.load(fp)

        with open(verb_idx_path) as f:
            self.flickr_verb_idx = json.load(f)
        
        with open(verb_vob_path) as f:
            self.verb_2_vob = json.load(f)
        
        with open(idbox_seq_path) as f:
            self.img_cap_v_2_idbox = json.load(f)

        with open(vocab_list_path) as f:
            vocab_list = json.load(f)
        self.vocab_2_idx = {}
        for idx, vocab in enumerate(vocab_list):
            self.vocab_2_idx[vocab] = idx
        
        with open(vlem_2_verb_og_path) as f:
            self.vlem_2_verb = json.load(f)
        
        self.sr_2_idx = {'ARG0': 1, 'ARG1': 2, 'ARG2': 3, 'ARG3': 4, 'ARG4': 5, 'ARG5': 6, 'LOC': 7, 'DIR': 8, 'GOL': 9, 
                        'MNR': 10, 'TMP': 11, 'EXT': 12, 'REC': 13, 'PRD': 14, 'PRP': 15, 'CAU': 16, 'DIS': 17, 'ADV': 18, 
                        'ADJ': 19, 'MOD': 20, 'NEG': 21, 'LVB': 22, 'PNC': 23, 'COM': 24, 'V': 25}

        super(FlickrDetSetField_Verb, self).__init__(None, postprocessing)

    def preprocess(self, x):
        image = x[0][0]
        caption = x[0][1]
        gt_bboxes = x[1]

        id_image = image.split('/')[-1].split('.')[0]
        try:
            f = h5py.File(self.detections_path, 'r')
            det_cls_probs = f['%s_cls_prob' % id_image][()]
            det_features = f['%s_features' % id_image][()]
            det_bboxes = f['%s_boxes' % id_image][()]
        except KeyError:
            warnings.warn('Could not find detections for %d' % id_image)
            det_cls_probs = np.random.rand(10, 2048)
            det_features = np.random.rand(10, 2048)
            det_bboxes = np.random.rand(10, 4)

        v_2_class = self.img_cap_v_2_idbox[id_image][caption]
        vlem_2_verb = self.vlem_2_verb[id_image][caption]

        loc_2_verb = {}
        loc_2_sr = {}
        loc = 0
        
        idbox_seq = {}

        cap_2_verb = []
        for verb in v_2_class:
            for sr in v_2_class[verb]:
                for id_box in v_2_class[verb][sr]:
                    if verb not in cap_2_verb:
                        cap_2_verb.append(verb)
                    if id_box not in idbox_seq:
                        idbox_seq[id_box] = loc
                        loc += 1

        control_verb = np.zeros(8)
        for j, verb in enumerate(cap_2_verb):
            if j >= 8:
                continue
            control_verb[j] = self.flickr_verb_idx[verb.split('_')[0]] + 1  # 0代表没有verb
        
        for verb in v_2_class:
            for sr in v_2_class[verb]:
                for id_box in v_2_class[verb][sr]:
                    loc_ = idbox_seq[id_box]
                    loc_2_verb.setdefault(loc_, []).append(verb)
                    loc_2_sr.setdefault(loc_, []).append(sr)

        # append the cls_seq with "verb" in the end
        for verb in cap_2_verb:
            idbox_seq[-1] = loc # -1代表没有或者verb
            loc_2_verb.setdefault(loc, []).append(verb)
            loc_2_sr.setdefault(loc, []).append(25)
            loc += 1

        width, height = self.img_shapes[str(id_image)]
        # visual feature
        pooled_feat = np.mean(det_features, axis=0)
        selected_classes = [self.classes[np.argmax(det_cls_probs[i][1:]) + 1] for i in range(len(det_cls_probs))]
        det_sequences_visual_all = np.zeros((self.fix_length, self.max_detections, det_features.shape[-1]))
        det_sequences_visual = np.zeros((self.fix_length, det_features.shape[-1]))
        det_sequences_word = np.zeros((self.fix_length, 300))
        det_sequences_position = np.zeros((self.fix_length, 4))

        # semantic role feature
        det_sequences_sr = np.zeros((self.fix_length, 8))
        det_sequences_verb = np.zeros((self.fix_length, 8))

        # 不是verb的位置为0，是verb的位置为verb的index
        verb_list = np.zeros((self.fix_length, 1))
        verb_list[verb_list==0] = -1

        for j, idbox in enumerate(idbox_seq):
            if j == 10: break
            if idbox >= 0:
                det_sequences_visual_all[j, 0] = det_features[idbox]  # 当前class所有region的feature
                det_sequences_visual[j] = det_features[idbox]  # 当前class第一个region的feature
                cls_w = selected_classes[idbox].split(',')[0].split(' ')[-1]
                if cls_w in self.vectors:
                    det_sequences_word[j] = self.vectors[cls_w]
                bbox = det_bboxes[idbox]  # 当前class第一个region的框 (x1, y1, x2, y2)
                det_sequences_position[j, 0] = (bbox[2] - bbox[0] / 2) / width  # 中心横坐标
                det_sequences_position[j, 1] = (bbox[3] - bbox[1] / 2) / height  # 中心纵坐标
                det_sequences_position[j, 2] = (bbox[2] - bbox[0]) / width  # 宽
                det_sequences_position[j, 3] = (bbox[3] - bbox[1]) / height  # 长
            else:
                det_sequences_visual_all[j, 0] = pooled_feat
                # 后面这串是verb_idx
                if loc_2_verb[j] != []:
                    if self.gt_verb is False:
                        verb_list[j, :] = self.flickr_verb_idx[loc_2_verb[j][0].split('_')[0]] + 1
                    else:
                        for v_lem, verb_og in vlem_2_verb:
                            if v_lem == loc_2_verb[j][0].split('_')[0]:
                                if verb_og in self.vocab_2_idx:
                                    verb_list[j, :] = self.vocab_2_idx[verb_og]
                                break

            for k, sr in enumerate(loc_2_sr[j]):
                if k >= 8:
                    continue
                det_sequences_sr[j, k] = sr  # 0代表pad
                det_sequences_verb[j, k] = self.flickr_verb_idx[loc_2_verb[j][k].split('_')[0]] + 1 # 0代表pad
        
        return det_sequences_word.astype(np.float32), det_sequences_visual.astype(np.float32), \
            det_sequences_position.astype(np.float32), det_sequences_visual_all.astype(np.float32), \
            det_sequences_verb.astype(np.float32), det_sequences_sr.astype(np.float32), control_verb.astype(np.float32), \
            verb_list

