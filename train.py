import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
import tqdm
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from utils.general import increment_path, init_seeds
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import MyModel
from loss import MyLoss
from dataset import MyDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.logutil import LoggerConfig, create_logger
from collections import Counter

def output_to_labels(o):
    pre_1 = [1 if prob >= 0.5 else 0 for prob in
                            o.detach().cpu().numpy()]
    return pre_1


def train(opt):
    save_dir, epochs, batch_size = \
        Path(
            opt.save_dir), opt.epochs, opt.batch_size

    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    init_seeds(2)

    ### model
    model = MyModel(in_channels=768, out_channels=768, opt=opt).cuda()#to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    ### Trainloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图片大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        # 标准化
    ])
    # train dataset
    cap_file_name = os.path.join(opt.cap_data_dir, 'caption_train.json')
    twt_file_name = os.path.join(opt.twt_data_dir, 'train.txt')
    attr_file_name = os.path.join(opt.twt_data_dir, 'img_to_five_words.json')
    train_dataset = MyDataset(cap_file_name=cap_file_name, twt_file_name=twt_file_name, attr_file_name=attr_file_name,img_data_dir=opt.img_data_dir,
                        transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

    # test dataset
    test_cap_file_name = os.path.join(opt.cap_data_dir, 'caption_test.json')
    test_twt_file_name = os.path.join(opt.twt_data_dir, 'test.txt')
    test_dataset = MyDataset(cap_file_name=test_cap_file_name,
                              twt_file_name=test_twt_file_name,
                             attr_file_name=attr_file_name,
                              img_data_dir=opt.img_data_dir,
                              transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size,
                              shuffle=True)


    criterion = MyLoss(batch_size=opt.batch_size, temperature=opt.temp, cl_weight=opt.cl_weight, bce_weight=opt.bce_weight) # nn.BCELoss()  # 二分类交叉熵损失


    for epoch in range(0, epochs):
        model.train()
        losses = 0
        preds = np.array([])
        gts = np.array([])
        for images, twts, caps, attrs, labels in tqdm.tqdm(train_loader, total=len(train_loader)):
            optimizer.zero_grad()  # 梯度清零
            labels = labels.cuda() # to(device)
            images = images.cuda() # to(device)
            if sum([x in opt.modality for x in 'itca']) == 4: # all modality
                v, o = model(images, twts, caps, attrs)

            elif sum([x in opt.modality for x in 'itc']) == 3: # img twt cap
                v, o = model(images, twts, caps, None)
            else:
                v, o = None, None

            loss = criterion(v, o.float(), labels.float())  # 计算损失
            # loss /= len(outs)
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数
            losses += loss

            predicted_labels = output_to_labels(o)

            gts = np.hstack((gts, labels.cpu().numpy()))
            preds = np.hstack((preds, predicted_labels))

        train_acc = accuracy_score(gts, preds)
        test_Acc, test_P, test_R, test_F1 = test(model, test_loader, )

        logger.info(f'**train** epoch {epoch}, train_acc: {train_acc}, train_loss: {losses.item()}, '
                    f'test_acc: {test_Acc}, test_P: {test_P}, test_R: {test_R}, test_F1: {test_F1}')


def test(model, test_loader):
    model.eval()
    preds = np.array([])
    gts = np.array([])
    for images, twts, caps, attrs, labels in tqdm.tqdm(test_loader,
                                                total=len(test_loader)):
        images = images.cuda() # to(device)
        labels = labels.cuda() # to(device)
        # v, o = model(images, twts, caps, attrs)  # 前向传播
        v, o = model(images, twts, caps, None)
        predicted_labels = output_to_labels(o)

        gts = np.hstack((gts, labels.cpu().numpy()))
        preds = np.hstack((preds, predicted_labels))
    test_Acc = accuracy_score(gts, preds)
    test_P = precision_score(gts, preds)
    test_R = recall_score(gts, preds)
    test_F1 = f1_score(gts, preds)
    return test_Acc, test_P, test_R, test_F1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--twt_data_dir', type=str, default='/home/p/Documents/Codes/SarcasmDetection/text_data/',
                        help='twt_data_dir')
    parser.add_argument('--img_data_dir', type=str, default='/home/p/Documents/Codes/SarcasmDetection/dataset_image/',
                        help='img_data_dir')
    parser.add_argument('--cap_data_dir', type=str, default='/home/p/Documents/Codes/SarcasmDetection/text_data/',
                        help='cap_data_dir')

    parser.add_argument('--train', default='train', help='train/test')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--epochs', type=int, default=50)
    # parser.add_argument('--n_block', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--temp', type=float, default=0.2)
    parser.add_argument('--cl-weight', type=float, default=0.2,
                        help='contrastive loss weight')
    parser.add_argument('--bce-weight', type=float, default=0.8,
                        help='binary cross entropy loss weight')
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--modality', default='itca', help='i:img, t:twt, c:cap, a:attr')
    parser.add_argument('--memo', default='',
                        help='notes')
    parser.add_argument('--fus', default='weighted',
                        help='fusion method: weighted/')


    opt = parser.parse_args()

    opt.name = f'temp:{opt.temp}_clw:{opt.cl_weight}_bcew:{opt.bce_weight}_lr:{opt.lr}.exp'

    WORKDIR = f'/home/p/Documents/Codes/workdir/{opt.modality}/{opt.fus}/{opt.memo}'
    opt.save_dir = increment_path(Path(WORKDIR) / opt.name,
                                  exist_ok=opt.exist_ok,
                                  sep='')  # increment run
    log_dir = opt.save_dir # os.path.join(os.path.dirname(__file__), opt.save_dir)
    print(f'log_dir: {log_dir}')
    logger_config = LoggerConfig(log_dir, log_file_name='train.log')
    logger = create_logger(__file__, logger_config)

    logger.info(
        f'-------------------------------hyperparameters---------------------------\n'
         f'{opt.name} \n'
         '-------------------------------------------------------------------------')
    train(opt)