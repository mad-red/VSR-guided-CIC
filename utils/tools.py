import six
from six.moves import cPickle

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def pickle_load(f):
    """ Load a pickle.
    Parameters
    ----------
    f: file-like object
    """
    if six.PY3:
        return cPickle.load(f, encoding='latin-1')
    else:
        return cPickle.load(f)

def pickle_dump(obj, f):
    """ Dump a pickle.
    Parameters
    ----------
    obj: pickled object
    f: file-like object
    """
    if six.PY3:
        return cPickle.dump(obj, f, protocol=2)
    else:
        return cPickle.dump(obj, f)

def verb_rank_merge(la, lb):
    merge_idx = []
    same_idx = []
    idx_in_b = []  # 相同的值在list b中的顺序
    origin_idx_in_b = []
    for idx_a in la:
        merge_idx.append(idx_a)  # 复制la到merge_idx
        for j, idx_b in enumerate(lb):
            if idx_a == idx_b:
                same_idx.append(idx_a)
                idx_in_b.append(j)
                origin_idx_in_b.append(j)
                break
    # 检测lb里的same_idx是否排列出错
    idx_in_b.sort()
    if origin_idx_in_b != idx_in_b:
        for j, idx in enumerate(idx_in_b):
            lb[idx] = same_idx[j]
    # 遍历lb里不是重合的词，将其插入la中
    right_idx = None
    right_idxs = {}
    for idx in reversed(lb):
        if idx not in same_idx:
            right_idxs[idx] = right_idx
        else:
            right_idx = idx
    for idx in lb:
        if idx not in same_idx:
            r_idx = right_idxs[idx]
            if r_idx is None:
                merge_idx.append(idx)
            else:
                for j, idx_m in enumerate(merge_idx):
                    if idx_m == r_idx:
                        merge_idx.insert(j, idx)
                        break
    return merge_idx

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def get_mapping(word_file):
    dict_ = {}
    word_list = []
    # idx = 0 means no verb
    word_list.append('non-verb')
    with open(word_file) as f:
        verb_2_idx = json.load(f)
    verb_num = len(verb_2_idx) + 1
    for verb, idx in verb_2_idx.items():
        dict_[verb] = idx + 1
        word_list.append(verb)

    return dict_, word_list, verb_num
