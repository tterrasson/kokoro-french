import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
from munch import Munch


def mask_from_lens(attn, in_lens, out_lens):
    """Build a [B, T_text, T_mel] boolean mask from length vectors."""
    B, T_t, T_s = attn.shape
    device = attn.device
    in_mask  = torch.arange(T_t, device=device).unsqueeze(0) < in_lens.unsqueeze(1)   # [B, T_t]
    out_mask = torch.arange(T_s, device=device).unsqueeze(0) < out_lens.unsqueeze(1)  # [B, T_s]
    return (in_mask.unsqueeze(2) & out_mask.unsqueeze(1)).float()                      # [B, T_t, T_s]


def _maximum_path_each(path, value, t_y, t_x):
    """DP monotonic alignment for a single batch element (pure NumPy)."""
    INF = float('inf')
    Q = np.full((t_y, t_x), -INF, dtype=np.float32)

    for x in range(t_x):
        for y in range(max(0, t_y + x - t_x), min(t_y, x + 1)):
            v_cur  = Q[y, x - 1] if x > y else -INF
            v_prev = (0.0 if x == 0 else -INF) if y == 0 else (Q[y - 1, x - 1] if x > 0 else -INF)
            Q[y, x] = max(v_cur, v_prev) + value[y, x]

    idx = t_y - 1
    for x in range(t_x - 1, -1, -1):
        path[idx, x] = 1
        if idx > 0 and (idx == x or Q[idx - 1, x - 1] > Q[idx, x - 1]):
            idx -= 1


def maximum_path(neg_cent, mask):
    """Monotonic alignment search (pure Python/NumPy, replaces Cython monotonic_align).
    neg_cent: [B, T_text, T_mel]
    mask:     [B, T_text, T_mel]
    """
    device = neg_cent.device
    dtype  = neg_cent.dtype
    value  = neg_cent.data.cpu().numpy().astype(np.float32)
    path   = np.zeros(value.shape, dtype=np.int32)

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)

    for b in range(value.shape[0]):
        _maximum_path_each(path[b], value[b], int(t_t_max[b]), int(t_s_max[b]))

    return torch.from_numpy(path).to(device=device, dtype=dtype)

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message)
    