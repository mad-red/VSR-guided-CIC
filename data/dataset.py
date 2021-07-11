import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import re
import xml.etree.ElementTree
from speaksee.data import field, Example, PairedDataset, COCO
from speaksee.utils import nostdout
from itertools import groupby
import pickle as pkl
from inspect import isgenerator
import h5py


class COCOEntities(PairedDataset):
    def __init__(self, image_field, det_field, text_field, img_root, ann_root, entities_file, id_root=None, 
                data_root='saved_data/coco', use_restval=True, filtering=False, det_filtering=False):
        roots = dict()
        roots['train'] = {
            'img': os.path.join(img_root, 'train2014'),
            'cap': os.path.join(ann_root, 'captions_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }

        if id_root is not None:
            ids = {}
            ids['train'] = np.load(os.path.join(id_root, 'coco_train_ids.npy'))
            ids['val'] = np.load(os.path.join(id_root, 'coco_dev_ids.npy'))
            ids['test'] = np.load(os.path.join(id_root, 'coco_test_ids.npy'))
            ids['trainrestval'] = (
                ids['train'],
                np.load(os.path.join(id_root, 'coco_restval_ids.npy')))

            if use_restval:
                roots['train'] = roots['trainrestval']
                ids['train'] = ids['trainrestval']
        else:
            ids = None

        if det_filtering:
            dataset_path = os.path.join(data_root, 'coco_entities_det_procomp.pkl')
        elif filtering:
            dataset_path = os.path.join(data_root, 'coco_entities_filtered_precomp.pkl')
        else:
            dataset_path = os.path.join(data_root, 'coco_entities_precomp.pkl')

        if not os.path.isfile(dataset_path):
            with nostdout():
                train_examples, val_examples, test_examples = COCO.get_samples(roots, ids)

            self.train_examples, self.val_examples, self.test_examples = \
                self.get_samples([train_examples, val_examples, test_examples], entities_file, filtering, det_filtering)
            pkl.dump((self.train_examples, self.val_examples, self.test_examples), open(dataset_path, 'wb'), -1)
        else:
            self.train_examples, self.val_examples, self.test_examples = pkl.load(open(dataset_path, 'rb'))

        examples = self.train_examples + self.val_examples + self.test_examples
        super(COCOEntities, self).__init__(examples, {'image': image_field, 'detection': det_field, 'text': text_field})

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    @classmethod
    def get_samples(cls, samples, entities_file, filtering=False, det_filtering=False):
        train_examples = []
        val_examples = []
        test_examples = []

        with open(entities_file, 'r') as fp:
            visual_chunks = json.load(fp)
        
        if filtering:
            gt_anno_path = 'saved_data/coco/img_caps_vb.json'
            with open(gt_anno_path) as f:
                img_caps_vb = json.load(f)
        
        if det_filtering:
            det_anno_path = 'saved_data/coco/img_cap_v_2_class_self.json'
            with open(det_anno_path) as f:
                img_cap_v_2_class = json.load(f)

        for id_split, samples_split in enumerate(samples):
            for s in samples_split:
                id_image = str(int(s.image.split('/')[-1].split('_')[-1].split('.')[0]))
                caption = s.text.lower().replace('\t', ' ').replace('\n', '')
                # remove the image-caption pair sample containing no verb
                if filtering and caption not in img_caps_vb[id_image]:
                    continue
                # 将验证的数据集中删除那些verb没有匹配上的image caption pair删掉
                if id_split == 2 and det_filtering:
                    if id_image not in img_cap_v_2_class:
                        continue
                    elif caption not in img_cap_v_2_class[id_image]:
                        continue
                words = caption.strip().split(' ')
                caption_fixed = []
                for w in words:
                    if w not in field.TextField.punctuations and w != '':
                        caption_fixed.append(w)

                det_classes = [None for _ in caption_fixed]
                caption_fixed = ' '.join(caption_fixed)

                for p in field.TextField.punctuations:
                    caption_fixed = caption_fixed.replace(p, '')

                if id_image in visual_chunks:
                    if caption in visual_chunks[id_image]:
                        chunks = visual_chunks[id_image][caption]
                        for chunk in chunks:  # [a couple of words, class]
                            words = chunk[0].split(' ')
                            chunk_fixed = []
                            for w in words:
                                if w not in field.TextField.punctuations and w != '':
                                    chunk_fixed.append(w)
                            chunk_fixed = ' '.join(chunk_fixed)
                            for p in field.TextField.punctuations:
                                chunk_fixed = chunk_fixed.replace(p, '')

                            sub_str = ' '.join(['_' for _ in chunk_fixed.split(' ')])
                            sub_cap = caption_fixed.replace(chunk_fixed, sub_str).split(' ')
                            for i, w in enumerate(sub_cap):
                                if w == '_':
                                    det_classes[i] = chunk[1]

                        example = Example.fromdict({'image': s.image,
                                                    'detection': ((s.image, caption), tuple(det_classes)),
                                                    'text': caption_fixed})

                        det_classes_set = [x[0] for x in groupby(det_classes) if x[0] is not None]
                        chunks_filtered = list(set([c[1] for c in chunks]))
                        if len(det_classes_set) < len(chunks_filtered):
                            pass
                        else:
                            if id_split == 0:
                                train_examples.append(example)
                            elif id_split == 1:
                                if filtering:
                                    if '_' not in example.detection[1]:
                                        val_examples.append(example)
                                else:
                                    val_examples.append(example)
                            elif id_split == 2:
                                if filtering:
                                    if '_' not in example.detection[1]:
                                        test_examples.append(example)
                                else:
                                    test_examples.append(example)

        return train_examples, val_examples, test_examples


class FlickrEntities(PairedDataset):
    def __init__(self, image_field, text_field, det_field, img_root, ann_file, entities_root,
                verb_filter=False, det_filter=False):
        if det_filter:
            precomp_file = 'saved_data/flickr/flickr_entities_precomp_df.pkl'
        elif verb_filter:
            precomp_file = 'saved_data/flickr/flickr_entities_precomp_vf.pkl'
        else:
            precomp_file='saved_data/flickr/flickr_entities_precomp.pkl'

        if os.path.isfile(precomp_file):
            with open(precomp_file, 'rb') as pkl_file:
                self.train_examples, self.val_examples, self.test_examples = pkl.load(pkl_file)
        else:
            self.train_examples, self.val_examples, self.test_examples = \
                self.get_samples(ann_file, img_root, entities_root, verb_filter=verb_filter, det_filter=det_filter)

        examples = self.train_examples + self.val_examples + self.test_examples
        super(FlickrEntities, self).__init__(examples, {'image': image_field, 'detection': det_field, 'text': text_field})

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    def get_samples(self, ann_file, img_root, entities_root, verb_filter=False, det_filter=False):
        def _get_sample(d):
            filename = d['filename']
            split = d['split']
            xml_root = xml.etree.ElementTree.parse(os.path.join(entities_root, 'Annotations',
                                                                filename.replace('.jpg', '.xml'))).getroot()
            det_dict = dict()
            id_counter = 1
            for obj in xml_root.findall('object'):
                obj_names = [o.text for o in obj.findall('name')]
                if obj.find('bndbox'):
                    bbox = tuple(int(o.text) for o in obj.find('bndbox'))
                    for obj_name in obj_names:
                        if obj_name not in det_dict:
                            det_dict[obj_name] = {'id': id_counter, 'bdnbox': [bbox]}
                            id_counter += 1
                        else:
                            det_dict[obj_name]['bdnbox'].append(bbox)

            bdnboxes = [[] for _ in range(id_counter - 1)]
            for it in det_dict.values():
                bdnboxes[it['id'] - 1] = tuple(it['bdnbox'])
            bdnboxes = tuple(bdnboxes)

            captions = [l.strip() for l in open(os.path.join(entities_root, 'Sentences',
                                                             filename.replace('.jpg', '.txt')), encoding="utf-8").readlines()]
            outputs = []
            for c in captions:
                matches = prog.findall(c)
                caption = []
                det_ids = []

                for match in matches:
                    for i, grp in enumerate(match):
                        if i in (0, 2):
                            if grp != '':
                                words = grp.strip().split(' ')
                                for w in words:
                                    if w not in field.TextField.punctuations and w != '':
                                        caption.append(w)
                                        det_ids.append(0)
                        elif i == 1:
                            words = grp[1:-1].strip().split(' ')
                            obj_name = words[0].split('#')[-1].split('/')[0]
                            words = words[1:]
                            for w in words:
                                if w not in field.TextField.punctuations and w != '':
                                    caption.append(w)
                                    if obj_name in det_dict:
                                        det_ids.append(det_dict[obj_name]['id'])
                                    else:
                                        det_ids.append(0)

                caption = ' '.join(caption)
                if caption != '' and np.sum(np.asarray(det_ids)) > 0:
                    example = Example.fromdict({'image': os.path.join(img_root, filename),
                                                'detection': ((os.path.join(img_root, filename), caption), bdnboxes, det_ids),
                                                'text': caption})
                    outputs.append([example, split])

            return outputs

        train_samples = []
        val_samples = []
        test_samples = []

        prog = re.compile(r'([^\[\]]*)(\[[^\[\]]+\])([^\[\]]*)')
        dataset = json.load(open(ann_file, 'r'))['images']

        samples = []
        for d in dataset:
            samples.extend(_get_sample(d))

        if verb_filter:
            gt_anno_path = 'datasets/flickr/cap_2_verb_nv.json'
            with open(gt_anno_path) as f:
                cap_2_verb = json.load(f)
        
        if det_filter:
            det_anno_path = 'saved_data/flickr/img_cap_v_2_idbox_flickr.json'
            with open(det_anno_path) as f:
                img_cap_v_2_class = json.load(f)

        for example, split in samples:
            # delete the image-caption pair containing no-verb
            if det_filter and split == 'test':
                imgid = example.image.split('/')[-1].split('.')[0]
                caption = example.text
                if imgid not in img_cap_v_2_class:
                    continue
                if caption not in img_cap_v_2_class[imgid]:
                    continue
            
            if verb_filter:
                imgid = example.image.split('/')[-1].split('.')[0]
                caption = example.text
                if cap_2_verb[imgid][caption] == []:
                    continue

            if split == 'train':
                train_samples.append(example)
            elif split == 'val':
                val_samples.append(example)
            elif split == 'test':
                test_samples.append(example)
        
        return train_samples, val_samples, test_samples


class NEWDataset(Dataset):
    """NEW dataset."""

    def __init__(self, train_file, is_training, inference=False, inference_verbs=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.inference = inference
        self.inference_verbs = inference_verbs
        self.train_file = train_file
        self.is_training = is_training


        self.image_names = []
        with open(train_file) as f:
            for line in f:
                self.image_names.append(line.split('\n')[0])

        with open('datasets/coco/img_idx_2_sr.json') as f:
            self.img_idx_2_sr = json.load(f)
        
        self.image_to_image_idx = {}
        i = 0
        for image_name in self.image_names:
            self.image_to_image_idx[image_name] = i
            i += 1

    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, idx):

        verb_idx = self.inference_verbs[self.image_names[idx]]
        id_img = int(self.image_names[idx].split('__')[0].split('_')[-1].split('.')[0])

        f = h5py.File('datasets/coco/coco_detections.hdf5', 'r')
        det_cls_probs = f['%s_cls_prob' % id_img][()]
        det_features = f['%s_features' % id_img][()]
        det_boxes = f['%s_boxes' % id_img][()]

        sr = self.img_idx_2_sr[self.image_names[idx]]

        sample = {'img_name': self.image_names[idx], 'verb_idx': verb_idx, 'det_cls_probs': det_cls_probs, 
                    'det_features': det_features, 'det_boxes': det_boxes, 'sr': sr}

        return sample


def collater_new(data):
    batch_size = len(data)
    img_names = [s['img_name'] for s in data]
    
    # 最多只保留前50个det
    det_features = np.zeros((batch_size, 50, 2048))
    det_cls_probs = np.zeros((batch_size, 50, data[0]['det_cls_probs'].shape[-1]))
    det_boxes = np.zeros((batch_size, 50, 4))
    for i, s in enumerate(data):
        max_len = min(50, len(s['det_features']))
        det_features[i, :max_len] = s['det_features'][:50]
        det_cls_probs[i, :max_len] = s['det_cls_probs'][:50]
        det_boxes[i, :max_len] = s['det_boxes'][:50]
    
    det_classes = np.zeros((batch_size, 26))
    det_cls_feat = np.zeros((batch_size, 26, 2048))
    # b_s, sr_len
    selected_classes = [np.argmax(s['det_cls_probs'][i][1:]) + 1 for s in data for i in range(len(s['det_cls_probs']))]

    cls_2_region = []
    k = 0
    for i in range(batch_size):
        cls_2_region.append({})
        for j in range(len(data[i]['det_cls_probs'])):
            cls_ = selected_classes[k]
            k += 1
            if j < 50:
                cls_2_region[i].setdefault(cls_, []).append(j)
    
    for i in range(batch_size):
        for j, cls_ in enumerate(cls_2_region[i]):
            if j < 20:
                det_classes[i, j] = cls_
                det_cls_feat[i, j] = np.mean(det_features[i, cls_2_region[i][cls_], :], axis=0)

    det_features = torch.tensor(det_features).float()
    det_cls_probs = torch.tensor(det_cls_probs).float()
    det_boxes = torch.tensor(det_boxes).float()
    det_classes = torch.tensor(det_classes).float()
    det_cls_feat = torch.tensor(det_cls_feat).float()
    
    sr = [s['sr'] for s in data]
    verb_indices = [s['verb_idx'] for s in data]
    verb_indices = torch.tensor(verb_indices)

    return {'img_name': img_names, 'verb_idx': verb_indices, 'det_cls_probs': det_cls_probs, 'det_features': det_features, 
            'det_classes': det_classes, 'det_cls_feat': det_cls_feat, 'det_boxes': det_boxes, 'sr': sr}


class Dataset_Flickr(Dataset):
    """dataset of flickr"""

    def __init__(self, train_file, is_training, inference=False, inference_verbs=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.inference = inference
        self.inference_verbs = inference_verbs
        self.train_file = train_file
        self.is_training = is_training

        self.image_names = []
        with open(train_file) as f:
            for line in f:
                self.image_names.append(line.split('\n')[0])

        with open('datasets/flickr/img_idx_2_srb_flickr.json') as f:
            self.img_idx_2_sr = json.load(f)
        
        self.image_to_image_idx = {}
        i = 0
        for image_name in self.image_names:
            self.image_to_image_idx[image_name] = i
            i += 1

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        verb_idx = self.inference_verbs[self.image_names[idx]]
        id_img = self.image_names[idx].split('__')[0]

        f = h5py.File('datasets/flickr/flickr30k_detections.hdf5', 'r')
        det_cls_probs = f['%s_cls_prob' % id_img][()]
        det_features = f['%s_features' % id_img][()]
        det_boxes = f['%s_boxes' % id_img][()]

        sr = self.img_idx_2_sr[self.image_names[idx]]

        sample = {'img_name': self.image_names[idx], 'verb_idx': verb_idx, 'det_cls_probs': det_cls_probs, 
                    'det_features': det_features, 'det_boxes': det_boxes, 'sr': sr}

        return sample


def collater_flickr(data):
    batch_size = len(data)
    img_names = [s['img_name'] for s in data]
    
    # 最多只保留前56个det
    det_features = np.zeros((batch_size, 56, 2048))
    det_cls_probs = np.zeros((batch_size, 56, data[0]['det_cls_probs'].shape[-1]))
    det_boxes = np.zeros((batch_size, 56, 4))
    for i, s in enumerate(data):
        max_len = min(56, len(s['det_features']))
        det_features[i, :max_len] = s['det_features'][:56]
        det_cls_probs[i, :max_len] = s['det_cls_probs'][:56]
        det_boxes[i, :max_len] = s['det_boxes'][:56]

    det_features = torch.tensor(det_features).float()
    det_cls_probs = torch.tensor(det_cls_probs).float()
    det_boxes = torch.tensor(det_boxes).float()
    
    sr = [s['sr'] for s in data]
    verb_indices = [s['verb_idx'] for s in data]
    verb_indices = torch.tensor(verb_indices)

    return {'img_name': img_names, 'verb_idx': verb_indices, 'det_cls_probs': det_cls_probs, 
            'det_features': det_features, 'det_boxes': det_boxes, 'sr': sr}

