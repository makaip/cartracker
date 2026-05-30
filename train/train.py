import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np

from model import EmbeddingNet
from dataset import VeRi
from utils import rtpm


if __name__ == "__main__":
    print(torch.cuda.is_available())

    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    num_classes = 776  # VeRi-776

    model = EmbeddingNet().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    optimizer = torch.optim.SGD(
        model.parameters(),         # from paper
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.1
    )

    ce_loss_fn = nn.CrossEntropyLoss()
    tri_loss_fn = nn.TripletMarginLoss(margin=0.3)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = VeRi(
        data_dir='datasets/VeRi/image_train',
        file='datasets/VeRi/train_label.xml',
        transform=transform
    )

    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=256,
                                             sampler=sampler,
                                             num_workers=8,
                                             pin_memory=True
                                             )

    rel_matrix = np.load('rel_mat.npy')

    epochs = 100
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        for img, label, idx in dataloader:
            loss = 0.0

            img = img.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)

            optimizer.zero_grad()
            embeds, logits = model(img)

            e_ent = ce_loss_fn(logits, label)                   # entropy loss
            triplets = rtpm(embeds, label, idx, rel_matrix)     # RTPM triplets

            if triplets:
                a = embeds[[t[0] for t in triplets]]            # anchor
                p = embeds[[t[1] for t in triplets]]            # positive
                n = embeds[[t[2] for t in triplets]]            # negative

                e_tri = tri_loss_fn(a, p, n)                    # triplet loss

                # im using lambda=1.0 for both losses, might adjust later tho
                loss = e_ent + e_tri
            else:
                # if there's no triplets then just use entropy loss
                loss = e_ent

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if local_rank == 0:
            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    dist.destroy_process_group()

    if local_rank == 0:
        torch.save(model.module.state_dict(
        ), "/mnt/beegfs/home/jpindell2022/ouri_project/mltests/caridentify/results/veri_rtpm_model.pth")
