import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model import EmbeddingNet
from dataset import VeRi
from utils import rptm_triplet_loss

# hyperparams follow https://github.com/adhirajghosh/RPTM_reid/blob/main/configs/veri_r101.yml
IMG_SIZE = 240
ANCHORS_PER_GPU = 16        # every anchor comes with its RPTM positive, so 32 images per GPU per step
EPOCHS = 80
BASE_LR = 0.01              # paper: 0.005 at batch 20. scaled up for 4 GPUs x 16 anchors = 64 per step
WARMUP_EPOCHS = 10
WARMUP_FACTOR = 0.01
LR_STEPS = (20, 40, 60)     # decay by 0.1 at each
MARGIN = 1.0
LAMBDA_ENT = 1.0
LAMBDA_TRI = 1.0

RESULTS_DIR = "/mnt/beegfs/home/jpindell2022/ouri_project/mltests/caridentify/results"
WEIGHTS_PATH = f"{RESULTS_DIR}/veri_rptm_r50ibn.pth"


def lr_factor(epoch):
    if epoch < WARMUP_EPOCHS:   # linear warmup
        return WARMUP_FACTOR + (1 - WARMUP_FACTOR) * epoch / WARMUP_EPOCHS

    return 0.1 ** sum(epoch >= step for step in LR_STEPS)


if __name__ == "__main__":
    print(torch.cuda.is_available())

    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    # num_classes = 776
    # # VeRi-776, BUT there are 576 VIDs in the training dataset

    model = EmbeddingNet(num_classes=576).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=BASE_LR,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(10),
        transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5),
    ])

    dataset = VeRi(
        data_dir='datasets/VeRi/image_train',
        file='datasets/VeRi/train_label.xml',
        transform=transform,
        rel_mat='rel_mat.npy'       # RPTM positives are picked from this, once, over each whole vehicle ID
    )

    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=ANCHORS_PER_GPU,
                                             sampler=sampler,
                                             num_workers=6,     # per GPU process
                                             pin_memory=True,
                                             drop_last=True
                                             )

    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = total_ent = total_tri = 0.0

        for anchor, positive, label, idx in dataloader:
            img = torch.cat([anchor, positive]).cuda(local_rank, non_blocking=True)     # (2B, 3, H, W)
            label = torch.cat([label, label]).cuda(local_rank, non_blocking=True)       # (2B,)

            optimizer.zero_grad()
            feats, embeds, logits = model(img)

            e_ent = ce_loss_fn(logits, label)                   # entropy loss, on the post-BN branch
            e_tri = rptm_triplet_loss(feats, label, MARGIN)     # RPTM triplet loss, on the pre-BN feature

            loss = LAMBDA_ENT * e_ent + LAMBDA_TRI * e_tri

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ent += e_ent.item()
            total_tri += e_tri.item()

        scheduler.step()

        if local_rank == 0:
            n = len(dataloader)
            print(
                f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/n:.4f} "
                f"(ent {total_ent/n:.4f}, tri {total_tri/n:.4f}), lr: {optimizer.param_groups[0]['lr']:.6f}",
                flush=True)

            # overwrite every epoch so a job that hits the time limit still leaves usable weights
            torch.save(model.module.state_dict(), WEIGHTS_PATH)

    dist.destroy_process_group()
