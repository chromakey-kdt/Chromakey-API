import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataloaders import davis_2016 as db
from .dataloaders import custom_transforms as tr
from .networks.vgg_osvos import OSVOS

from .dataloaders.helpers import *


def predictor(user_id, timestamp):
    db_root_dir = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "predict")
    save_dir = os.path.join('./models', user_id, timestamp)

    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = OSVOS(pretrained=0)
    net.load_state_dict(torch.load(os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "model.pth"), map_location=lambda storage, loc: storage))
    net.to(device)
    net.eval()

    db_test = db.DAVIS2016(train=False, db_root_dir=db_root_dir, transform=transforms.Compose([tr.ToTensor()]), seq_name="")
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

    save_dir_res = os.path.join(save_dir, 'Results')
    if not os.path.exists(save_dir_res):
        os.makedirs(save_dir_res)

    with torch.no_grad():
        for ii, sample_batched in enumerate(test_loader):
            img, _, fname = sample_batched['image'], sample_batched['gt'], sample_batched['fname']
            inputs = img.to(device)

            outputs = net.forward(inputs)

            for jj in range(int(inputs.size()[0])):
                pred = np.transpose(outputs[-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
                pred = 1 / (1 + np.exp(-pred))
                pred = np.squeeze(pred)

                pred = np.float32((pred > 0.99)) * 255
                cv2.imwrite(os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, "tmp.png"), pred)

                pred = pred.tolist()

                pixels = []
                for y in range(len(pred)):
                    for x in range(len(pred[y])):
                        if pred[y][x] != 0:
                            pixels.append((x, y))

                return pixels
                # cv2.imwrite(os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, "tmp.png"), pred)
                #
                # return os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, "tmp.png")

def video_predictor(user_id, timestamp):
    db_root_dir = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "video")
    save_dir = os.path.join('./models', user_id, timestamp)

    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = OSVOS(pretrained=0)
    net.load_state_dict(torch.load(os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "model.pth"), map_location=lambda storage, loc: storage))
    net.to(device)
    net.eval()

    db_test = db.DAVIS2016(train=False, db_root_dir=db_root_dir, transform=transforms.Compose([tr.ToTensor()]), seq_name="")
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

    save_dir_res = os.path.join(save_dir, 'Results')
    if not os.path.exists(save_dir_res):
        os.makedirs(save_dir_res)

    video_path = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "video")
    os.makedirs(os.path.join(video_path, "Predict"), exist_ok=True)

    predict_img_list = []
    with torch.no_grad():
        for ii, sample_batched in enumerate(test_loader):
            img, _, fname = sample_batched['image'], sample_batched['gt'], sample_batched['fname']

            inputs = img.to(device)

            outputs = net.forward(inputs)

            for jj in range(int(inputs.size()[0])):
                pred = np.transpose(outputs[-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
                pred = 1 / (1 + np.exp(-pred))
                pred = np.squeeze(pred)

                pred = np.float32((pred > 0.99)) * 255

                # 비디오 이미지 예측 부분
                # cv2.imwrite(os.path.join(video_path, "JPEGImages", f"video-frame-{idx}.png"), frame)
                cv2.imwrite(os.path.join(video_path, "Predict", f"predict-frame-{ii + 1}.png"), pred)
                # 원본 ) cv2.imwrite(os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, f"result-{ii + 1}.png", pred)

                H, W = pred.shape

                mask = np.zeros([H, W, 3])
                # mask[:, :, 0] = pred
                # mask[:, :, 1] = pred
                mask[:, :, 2] = pred

                origin_image = cv2.imread(os.path.join(db_root_dir, "JPEGImages", f"video-frame-{ii + 1}.png"))

                # new_image = np.where((origin_image + np.asarray(mask, dtype = int)) > 255, 255, origin_image + np.asarray(mask, dtype=int))
                new_image = cv2.addWeighted(np.asarray(origin_image, dtype=int), 0.7, np.asarray(mask, dtype=int), 0.3, 0)

                predict_img_list.append(new_image)

        fcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "output.avi"), fcc, 20, (W, H))

        for result in predict_img_list:
            out.write(np.asarray(result, dtype=np.uint8))

        out.release()


def new_video_predictor(user_id):
    pass