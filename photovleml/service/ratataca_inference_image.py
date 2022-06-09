import os

# PyTorch includes
import torch
from torch.utils.data import DataLoader

# Custom includes
from dataloaders import davis_2016 as db
from dataloaders import custom_transforms as tr

import cv2
import networks.vgg_osvos as vo
from dataloaders.helpers import *
from mypath import Path


def predictor(user_id):
    db_root_dir = Path.db_root_dir()
    save_dir = os.path.join('./models', user_id, timestamp)

    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Network definition
    net = vo.OSVOS(pretrained=0)
    net.load_state_dict(torch.load(os.path.join(save_dir, 'model.pth'), map_location=lambda storage, loc: storage))
    net.to(device)
    net.eval()

    # Testing dataset and its iterator
    db_test = db.DAVIS2016(train=False, db_root_dir=db_root_dir, transform=tr.ToTensor(), seq_name="")
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

    save_dir_res = os.path.join(save_dir, 'Results', "")

    if not os.path.exists(save_dir_res):
        os.makedirs(save_dir_res)

    print('Testing Network')
    with torch.no_grad():
        for ii, sample_batched in enumerate(test_loader):

            img, gt, fname = sample_batched['image'], sample_batched['gt'], sample_batched['fname']

            # Forward of the mini-batch
            inputs, gts = img.to(device), gt.to(device)

            outputs = net.forward(inputs)

            for jj in range(int(inputs.size()[0])):
                pred = np.transpose(outputs[-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
                pred = 1 / (1 + np.exp(-pred))
                pred = np.squeeze(pred)

                pred = np.float32((pred > 0.99))

                # Save the result, attention to the index jj
                cv2.imwrite(os.path.join(save_dir_res, os.path.basename(fname[jj]) + '.png'), pred)

