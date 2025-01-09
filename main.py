import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('/home/p/Documents/Codes/sarcasm_homework/')
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
from util import AverageMeter

def parse_option():
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
    parser.add_argument('--modality', 
                        default='itc', 
                        # default='itca', 
                        help='i:img, t:twt, c:cap, a:attr')
    parser.add_argument('--memo', default='',
                        help='notes')
    parser.add_argument('--fus', default='weighted',
                        help='fusion method: weighted/')


    opt = parser.parse_args()
    opt.name = f'temp:{opt.temp}_clw:{opt.cl_weight}_bcew:{opt.bce_weight}_lr:{opt.lr}.exp'

    WORKDIR = f'/home/p/Documents/Codes/workdir2/{opt.modality}/{opt.fus}/{opt.memo}'
    opt.save_dir = increment_path(Path(WORKDIR) / opt.name,
                                  exist_ok=opt.exist_ok,
                                  sep='')  # increment run
    print(opt)
    
    return opt

def output_to_labels(o):
    pre_1 = [1 if prob >= 0.5 else 0 for prob in
                            o.detach().cpu().numpy()]

    # 使用列表推导和Counter统计每个位置上0和1的出现次数
    # predicted_labels = [
    #     0 if
    #     Counter(pred[i] for pred in [pre_1, pre_2, pre_3]).most_common(1)[0][
    #         0] == 0 else 1 for i in range(len(pre_1))]
    return pre_1

def set_loader(opt):
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
    return train_loader, test_loader

def set_model(opt):
    ### model
    model = MyModel(in_channels=768, out_channels=768, opt=opt).cuda()#to(device)
    criterion = MyLoss(batch_size=opt.batch_size, temperature=opt.temp, cl_weight=opt.cl_weight, bce_weight=opt.bce_weight).cuda() # nn.BCELoss()  # 二分类交叉熵损失
    return model, criterion


def set_optimizer(opt, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    return optimizer

def train(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()

    # save_dir, epochs, batch_size = \
    #     Path(
    #         opt.save_dir), opt.epochs, opt.batch_size

    # with open(save_dir / 'opt.yaml', 'w') as f:
    #     yaml.dump(vars(opt), f, sort_keys=False)



    losses = AverageMeter()
    top1 = AverageMeter()
    preds = np.array([])
    gts = np.array([])
    for idx, batch in enumerate(tqdm(train_loader, total=len(train_loader))):
        images, twts, caps, attrs, labels = batch
        bsz = len(labels)
        labels = labels.cuda() # to(device)
        images = images.cuda() # to(device)
        if opt.modality == 'itca': # all modality
            v, o = model(images, twts, caps, attrs)
        # elif sum([x in opt.modality for x in 'ct']) == 2: # cap_twt
        #     v, o = model(None, twts, caps, None)
        # elif sum([x in opt.modality for x in 'ic']) == 2:  # img cap
        #     v, o = model(images, None, caps, None)
        # elif sum([x in opt.modality for x in 'it']) == 2: # img twt
        #     v, o = model(images, twts, None, None)
        elif opt.modality == 'itc': # img twt cap
            v, o = model(images, twts, caps, None)
        else:
            v, o = None, None

        loss = criterion(v, o.float(), labels.float())  # 计算损失
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

        predicted_labels = output_to_labels(o)

        gts = np.hstack((gts, labels.cpu().numpy()))
        preds = np.hstack((preds, predicted_labels))
    train_acc = accuracy_score(gts, preds)

    return losses.avg, train_acc


def eval(test_loader, model, criterion, opt):
    model.eval()

    losses = AverageMeter()
    preds = np.array([])
    gts = np.array([])
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, total=len(test_loader))):
            images, twts, caps, attrs, labels = batch
            bsz = len(labels)
            labels = labels.cuda() # to(device)
            images = images.cuda() # to(device)
            # v, o = model(images, twts, caps, attrs)  # 前向传播
            v, o = model(images, twts, caps, None)
            predicted_labels = output_to_labels(o)

            gts = np.hstack((gts, labels.cpu().numpy()))
            preds = np.hstack((preds, predicted_labels))
            loss = criterion(v, o.float(), labels.float())  # 计算损失
            losses.update(loss.item(), bsz)
        test_Acc = accuracy_score(gts, preds)
        test_P = precision_score(gts, preds)
        test_R = recall_score(gts, preds)
        test_F1 = f1_score(gts, preds)
    return losses.avg, test_Acc, test_P, test_R, test_F1

    


def main():
    init_seeds(2)
    best_test_acc = 0
    opt = parse_option()
    log_dir = opt.save_dir # os.path.join(os.path.dirname(__file__), opt.save_dir)
    print(f'log_dir: {log_dir}')
    logger_config = LoggerConfig(log_dir, log_file_name='train.log')
    logger = create_logger(__file__, logger_config)
    writer = SummaryWriter(log_dir)
    logger.info(
        f'-------------------------------hyperparameters---------------------------\n'
         f'{opt.name} \n'
         '-------------------------------------------------------------------------')
    
    train_loader, test_loader = set_loader(opt)
    model, criterion = set_model(opt)
    # build optimizer
    optimizer = set_optimizer(opt, model)
    for epoch in range(1, opt.epochs + 1):
        time1 = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        writer.add_scalar('train_loss', train_loss, global_step=epoch)
        writer.add_scalar('train_acc', train_acc, global_step=epoch)
        print(f'Train epoch {epoch}, total time {time2 - time1}, train_loss:{train_loss}, train_accuracy:{train_acc}')
        
        test_loss, test_Acc, test_P, test_R, test_F1 = eval(test_loader, model, criterion, opt)
        print(f'Train epoch {epoch}, valid_loss:{test_loss}, valid_accuracy:{test_Acc} \nprecision:{test_P}, recall:{test_R},F_score:{test_F1}')
        if test_Acc > best_test_acc:
            best_test_acc = test_Acc
            # torch.save(model.state_dict(), os.path.join(opt.save_folder, "pytorch_model_aver_valid0"  + ".bin"))
            print("better model")
        writer.add_scalar('test_loss', test_loss, global_step=epoch)
        writer.add_scalar('test_Acc', test_Acc, global_step=epoch)
        writer.add_scalar('test_P', test_P, global_step=epoch)
        writer.add_scalar('test_R', test_R, global_step=epoch)
        writer.add_scalar('test_F1', test_F1, global_step=epoch)
        info = f'**train** epoch {epoch}, train_acc: {train_acc}, train_loss: {train_loss}, '\
               f'test_Acc: {test_Acc}, test_P: {test_P}, test_R: {test_R}, test_F1: {test_F1}'
        logger.info(info)
        print(info)
    print('best accuracy: {:.2f}'.format(best_test_acc))
    

if __name__ == '__main__':
    main()