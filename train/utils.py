import torch
import torch.nn as nn
import numpy as np

"""
Squeeze-Excitation (SE) layer: https://arxiv.org/pdf/1709.01507
- integrated into residual blocks of conv net
- performs global average pooling on feature maps, then squeeze, then excitation steps
- out -> set of weights for channel-wise multiplication (importance map)
"""

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        # squeeze: (B, C, H, W) -> (B, C, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(
            nn.Linear(
                channels,               # 1 x 1 x C 
                channels // reduction   # 1 x 1 x C/r
            ),
            nn.ReLU(),                  # 1 x 1 x C/r
            nn.Linear(
                channels // reduction,  # 1 x 1 x C/r
                channels                # 1 x 1 x C
            ),
            nn.Sigmoid()                # 1 x 1 x C
        )

    def forward(self, x):
        # b: batch; c: channels; _: height; _: width
        b, c, _, _ = x.shape            # unpack batch & channel dims
        w = self.pool(x)                # pool to B x C x 1 x 1
        w = w.view(b, c)                # reshape to B x C
        w = self.fc(w)                  # pass through FC layers to get B x C weights
        w = w.view(b, c, 1, 1)          # reshape back to B x C x 1 x 1
        
        return x * w                    # channel-wise mult

"""
https://arxiv.org/abs/1807.09441

split channels, replace the first BN layer in certain residual blocks with a half IN, half BN operation, then concatenate
"""

class IBN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        half = channels // 2    # split
        self.IN = nn.InstanceNorm2d(half, affine=True)  # instancenorm
        self.BN = nn.BatchNorm2d(half)  # batchnorm

    def forward(self, x):
        first, second = torch.chunk(x, 2, dim=1)
        return torch.cat( # concatenate
            [self.IN(first),
             self.BN(second)],
            dim=1)

"""
Relation Preserving Triplet Mining (RPTM), as described in https://arxiv.org/pdf/2110.07933v2
- positives: for each anchor, out of ALL images of its vehicle ID, take the one whose number of GMS matches
  with the anchor is closest to tau (mean of the anchor's non-zero matches). picked once, before training.
- negatives: batch hard mining, the closest embedding in the batch that has a different ID
"""

def rptm_positives(rel_mat, groups) -> np.ndarray:
    """
    rel_mat: (m, m) GMS match counts (a np.memmap is fine, only same-ID blocks are read)
    groups: list of dataset-index lists, one per vehicle ID
    returns: (m,) dataset index of the RPTM positive of each image, -1 if it has no GMS matches
    """
    pos = np.full(rel_mat.shape[0], -1, dtype=np.int64)

    for idx in groups:
        idx = np.asarray(idx)
        block = np.asarray(rel_mat[idx][:, idx], dtype=np.float64)  # match counts within this ID
        np.fill_diagonal(block, 0)                                  # an image is never its own positive

        for row, anchor in enumerate(idx):
            counts = block[row]
            nonzero = np.flatnonzero(counts)
            if len(nonzero) == 0:
                continue

            tau = counts[nonzero].mean()
            best = nonzero[np.argmin(np.abs(counts[nonzero] - tau))]
            pos[anchor] = idx[best]

    return pos


def rptm_triplet_loss(feats: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    """
    feats: (2B, D) features, first B rows are anchors, last B rows are their RPTM positives (same order)
    labels: (2B,) vehicle IDs
    """
    b = feats.shape[0] // 2
    anchors = feats[:b]

    # euclidean distance from each anchor to everything in the batch (clamped so sqrt has a finite gradient)
    sq = anchors.pow(2).sum(1, keepdim=True) + feats.pow(2).sum(1) - 2 * anchors @ feats.t()
    dist = sq.clamp(min=1e-12).sqrt()                               # (B, 2B)

    rows = torch.arange(b, device=feats.device)
    d_ap = dist[rows, rows + b]                                     # anchor -> its RPTM positive

    is_neg = labels[:b, None] != labels[None, :]                    # (B, 2B)
    d_an = dist.masked_fill(~is_neg, float('inf')).min(dim=1).values  # anchor -> hardest negative

    valid = is_neg.any(dim=1)                                       # anchors that have a negative in the batch
    if not valid.any():
        return feats.sum() * 0.0

    return torch.relu(d_ap[valid] - d_an[valid] + margin).mean()


"""
k-reciprocal re-ranking: https://arxiv.org/abs/1701.08398
re-orders the gallery using each image's k-reciprocal neighbours (jaccard distance) on top of the original distance
"""

def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3) -> np.ndarray:
    original_dist = np.concatenate([
        np.concatenate([q_q_dist, q_g_dist], axis=1),
        np.concatenate([q_g_dist.T, g_g_dist], axis=1)
    ], axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist, dtype=np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]
    half_k1 = int(np.around(k1 / 2.))

    for i in range(all_num):
        # k-reciprocal neighbours: j is in i's top k1 and i is in j's top k1
        forward = initial_rank[i, :k1 + 1]
        backward = initial_rank[forward, :k1 + 1]
        k_reciprocal = forward[np.where(backward == i)[0]]
        expansion = k_reciprocal

        for candidate in k_reciprocal:
            cand_forward = initial_rank[candidate, :half_k1 + 1]
            cand_backward = initial_rank[cand_forward, :half_k1 + 1]
            cand_k_reciprocal = cand_forward[np.where(cand_backward == candidate)[0]]

            if len(np.intersect1d(cand_k_reciprocal, k_reciprocal)) > 2. / 3 * len(cand_k_reciprocal):
                expansion = np.append(expansion, cand_k_reciprocal)

        expansion = np.unique(expansion)
        weight = np.exp(-original_dist[i, expansion])
        V[i, expansion] = weight / np.sum(weight)

    original_dist = original_dist[:query_num]

    if k2 != 1:                                     # local query expansion
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i] = np.mean(V[initial_rank[i, :k2]], axis=0)
        V = V_qe
        del V_qe
    del initial_rank

    inv_index = [np.where(V[:, i] != 0)[0] for i in range(all_num)]
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros((1, all_num), dtype=np.float32)
        nonzero = np.where(V[i] != 0)[0]

        for j in nonzero:
            images = inv_index[j]
            temp_min[0, images] += np.minimum(V[i, j], V[images, j])

        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    return final_dist[:query_num, query_num:]
