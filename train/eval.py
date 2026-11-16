"""
evaluate on the VeRi-776 query / gallery split: mAP and CMC rank-1/5/10

    python eval.py --weights results/veri_rptm_r50ibn.pth --rerank

- deterministic transforms (no random augmentation at test time)
- embedding = model(img) + model(horizontally flipped img), L2 normalized
- gallery images of the query's own vehicle from the query's own camera are ignored (standard protocol)
- --rerank also reports the numbers after k-reciprocal re-ranking (what the paper reports)
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

from model import EmbeddingNet
from dataset import VeRiEval
from utils import re_ranking


@torch.no_grad()
def embed(model, loader, device, desc):
    embs, vids, cids = [], [], []

    for imgs, vid, cid in tqdm(loader, desc=desc):
        imgs = imgs.to(device)
        emb = model(imgs)[0] + model(torch.flip(imgs, dims=[3]))[0]     # flip test-time augmentation
        embs.append(F.normalize(emb, p=2, dim=1).cpu())
        vids.extend(vid.tolist())
        cids.extend(cid.tolist())

    return torch.cat(embs), np.array(vids), np.array(cids)


def score(dist, q_vids, q_cids, g_vids, g_cids, topk=(1, 5, 10)):
    """dist: (num_query, num_gallery), smaller = more similar"""
    cmc_counts = np.zeros(max(topk), dtype=int)
    APs = []

    for i in range(dist.shape[0]):
        order = np.argsort(dist[i])     # sort by dist to query (ascending)
        g_vids_ord = g_vids[order]
        g_cids_ord = g_cids[order]

        junk = (g_vids_ord == q_vids[i]) & (g_cids_ord == q_cids[i])
        matches = (g_vids_ord[~junk] == q_vids[i]).astype(int)

        if matches.sum() == 0:
            continue    # no valid match for this query, so skip

        hits = np.where(matches == 1)[0]
        if hits[0] < len(cmc_counts):
            cmc_counts[hits[0]:] += 1

        c = np.cumsum(matches)
        APs.append(np.mean([c[k] / (k + 1) for k in hits]))

    if not APs:
        return 0.0, {k: 0.0 for k in topk}

    cmc = cmc_counts / len(APs)
    return float(np.mean(APs)), {k: float(cmc[k - 1]) for k in topk}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="datasets/VeRi", help="VeRi dataset root")
    parser.add_argument("--size", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rerank", action="store_true", help="also report k-reciprocal re-ranked results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmbeddingNet(num_classes=576, pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    query = VeRiEval(f"{args.data}/image_query", f"{args.data}/name_query.txt", transform)
    gallery = VeRiEval(f"{args.data}/image_test", f"{args.data}/name_test.txt", transform)

    q_embs, q_vids, q_cids = embed(
        model, torch.utils.data.DataLoader(query, batch_size=args.batch_size, num_workers=8), device, "Query Embeddings")
    g_embs, g_vids, g_cids = embed(
        model, torch.utils.data.DataLoader(gallery, batch_size=args.batch_size, num_workers=8), device, "Gallery Embeddings")

    q_g = torch.cdist(q_embs, g_embs).numpy()

    mAP, cmc = score(q_g, q_vids, q_cids, g_vids, g_cids)
    print(f"mAP: {mAP:.4f}")
    print("CMC:", cmc)

    if args.rerank:
        q_q = torch.cdist(q_embs, q_embs).numpy()
        g_g = torch.cdist(g_embs, g_embs).numpy()

        mAP, cmc = score(re_ranking(q_g, q_q, g_g), q_vids, q_cids, g_vids, g_cids)
        print(f"re-ranked mAP: {mAP:.4f}")
        print("re-ranked CMC:", cmc)
