import numpy as np
import torch
from torch.utils.data import DataLoader
from data_loaders import labelFpsDataLoader
from models import LPRNet
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

use_gpu = False


def train():
    train_dir = ["./data/test"]
    image_size = (100, 100)
    epochs = 100
    batch_size = 32

    transforms = Compose([ToTensor(), Resize(image_size)])

    model = LPRNet()
    optimizer = torch.optim.Adam(model.parameters())
    dst = labelFpsDataLoader(train_dir, transforms)
    trainloader = DataLoader(dst, batch_size=batch_size, shuffle=True, num_workers=8)
    for epoch in range(epochs):
        print(epoch)
        train_one_epoch(model, optimizer, trainloader, batch_size)


def train_one_epoch(model, optimizer, dataloader, batch_size):
    for i, (images, bboxes, labels, image_names) in enumerate(dataloader):
        if not len(images) == batch_size:
            continue
        YI = [[int(ee) for ee in el.split("_")[:7]] for el in labels]
        # y = torch.from_numpy(np.array([el.numpy() for el in bboxes]).T)
        y = bboxes

        if use_gpu:
            x = images.cuda(0)
            # y = y.cuda(0)
            y = {k: v.cuda(0) for k, v in y.items()}
        else:
            x = images
            y = y

        loss = model(x, y, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)


if __name__ == "__main__":
    train()
