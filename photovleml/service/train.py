import os
import cv2
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from .dataloaders import davis_2016 as db
from .dataloaders import custom_transforms as tr
from .networks.vgg_osvos import OSVOS
from .layers.osvos_layers import class_balanced_cross_entropy_loss


def trainer(user_id, timestamp, epoches=100, batch=500):
    db_root_dir = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp)
    save_dir = os.path.join('./models', user_id, timestamp)

    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Network definition
    net = OSVOS(pretrained=0)
    net.load_state_dict(torch.load(os.getenv("PRETRAINED_MODEL_PATH"), map_location=lambda storage, loc: storage))
    net.to(device)
    net.train()

    lr = 1e-9
    wd = 0.005
    optimizer = optim.SGD([
        {'params': [pr[1] for pr in net.stages.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
        {'params': [pr[1] for pr in net.stages.named_parameters() if 'bias' in pr[0]], 'lr': lr * 2},
        {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'weight' in pr[0]], 'weight_decay': wd},
        {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'bias' in pr[0]], 'lr': lr*2},
        {'params': [pr[1] for pr in net.upscale.named_parameters() if 'weight' in pr[0]], 'lr': 0},
        {'params': [pr[1] for pr in net.upscale_.named_parameters() if 'weight' in pr[0]], 'lr': 0},
        {'params': net.fuse.weight, 'lr': lr/100, 'weight_decay': wd},
        {'params': net.fuse.bias, 'lr': 2*lr/100},
        ], lr=lr, momentum=0.9)

    composed_transforms = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-20, 20), scales=(.90, 1.10)),
        tr.ToTensor()
    ])

    db_train = db.DAVIS2016(train=True, db_root_dir=db_root_dir, transform=composed_transforms, seq_name="")
    train_loader = DataLoader(db_train, batch_size=batch, shuffle=True, num_workers=0)

    previous_loss = 1000000000000
    for epoch in range(0, epoches):
        for ii, sample_batched in enumerate(train_loader):

            inputs, gts = sample_batched['image'], sample_batched['gt']

            inputs.requires_grad_()
            inputs, gts = inputs.to(device), gts.to(device)

            outputs = net.forward(inputs)

            loss = class_balanced_cross_entropy_loss(outputs[-1], gts, size_average=False)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            current_loss = loss.item()
            print(current_loss)

        if current_loss < previous_loss:
            torch.save(net.state_dict(), os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "model.pth"))
            previous_loss = current_loss

