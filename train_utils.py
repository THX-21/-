import math
from functools import partial
import os
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelBinarizer,label_binarize
import matplotlib.pyplot as plt

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 获得优化器学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# 学习单个epoch
def fit_one_epoch(model, optimizer, epoch, train_dataloader, val_dataloader, device, save_dir,log_dir,
                  Epoch_num, eval_period=5, save_period=5):
    loss = 0
    val_loss = 0
    write_loss=SummaryWriter(os.path.join(log_dir,'loss'))
    write_acc = SummaryWriter(os.path.join(log_dir, 'acc'))
    # 训练
    model.train()
    acc_num = 0
    num = 0
    pbar = tqdm(total=len(train_dataloader), desc=f'TRAIN Epoch {epoch + 1}/{Epoch_num}', postfix=dict, mininterval=0.3)
    for iteration, batch in enumerate(train_dataloader):
        images, labels = batch[0], batch[1]
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
        optimizer.zero_grad()
        probability = model(images)
        loss_value=F.nll_loss(torch.log(probability),labels)
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-1)
        optimizer.step()
        acc_num += (torch.argmax(probability, dim=1) == labels).int().sum()
        num += len(labels)
        loss = loss_value.item()
        write_loss.add_scalar('loss',loss,iteration+epoch*len(train_dataloader))
        write_acc.add_scalar('acc',(acc_num / num).item(),iteration+epoch*len(train_dataloader))
        pbar.set_postfix(**{'loss': loss ,
                            'lr': get_lr(optimizer),
                            'acc': (acc_num / num).item()})
        pbar.update(1)
    pbar.close()
    write_loss.close()
    write_acc.close()

    # 验证
    if (epoch+1) % eval_period == 0:
        write_loss = SummaryWriter(os.path.join(log_dir, 'val_loss'))
        write_acc = SummaryWriter(os.path.join(log_dir, 'val_acc'))
        model.eval()
        acc_num=0
        num=0
        pbar = tqdm(total=len(val_dataloader), desc=f'VALIDATION Epoch {epoch + 1}/{Epoch_num}', postfix=dict, mininterval=0.3)
        for iteration, batch in enumerate(val_dataloader):
            images, labels = batch[0], batch[1]
            with torch.no_grad():
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                probability = model(images)
                loss_value = F.nll_loss(torch.log(probability), labels)
                acc_num+=(torch.argmax(probability,dim=1)==labels).int().sum()
                num+=len(labels)


            val_loss = loss_value.item()
            write_loss.add_scalar('val_loss', val_loss, iteration + epoch * len(val_dataloader)//eval_period)
            write_acc.add_scalar('val_acc', (acc_num / num).item(), iteration + epoch * len(val_dataloader)//eval_period)
            pbar.set_postfix(**{'val_loss': val_loss,
                                'lr': get_lr(optimizer),
                                'val_acc': (acc_num/num).item() })
            pbar.update(1)
        pbar.close()
        write_loss.close()
        write_acc.close()
    else:
        val_loss = None

    # 保存模型
    if (epoch+1) % save_period == 0:
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
        print("save latest model")
    return loss, val_loss


def test_model(model,test_dataloader,device,graph=False):
    model.eval()
    acc_num = 0
    num = 0
    test_loss = 0
    y_true=torch.tensor([]).to(device)
    y_pred=torch.tensor([]).to(device)
    y_scores=torch.tensor([]).to(device)
    pbar = tqdm(total=len(test_dataloader),desc="TEST", postfix=dict, mininterval=0.3)
    for iteration, batch in enumerate(test_dataloader):
        images, labels = batch[0], batch[1]
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            probability = model(images)
            loss_value = F.nll_loss(torch.log(probability), labels)
            acc_num += (torch.argmax(probability, dim=1) == labels).int().sum()
            num += len(labels)

        y_true=torch.cat((y_true,labels),dim=0)
        y_pred=torch.cat((y_pred, torch.argmax(probability, dim=1)), dim=0)
        y_scores=torch.cat((y_scores, probability), dim=0)
        test_loss = loss_value.item()
        pbar.set_postfix(**{'loss': test_loss,
                            'acc': (acc_num / num).item()})

        pbar.update(1)
    # 计算各项指标
    y_true=y_true.cpu()
    y_pred=y_pred.cpu()
    y_scores=y_scores.cpu()
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # 计算 auROC
    # 需要将标签二值化
    y_true_binarized = label_binarize(y_true, classes=np.arange(6))
    auroc = roc_auc_score(y_true_binarized, y_scores, average='macro', multi_class='ovr')

    # 打印结果
    print(f'Accuracy: {accuracy}')
    print(f'Macro Precision: {macro_precision}')
    print(f'Macro Recall: {macro_recall}')
    print(f'Macro F1-score: {macro_f1}')
    print(f'auROC: {auroc}')

    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)
    if graph:
        # 从数据集中随机选择6张图片及其标签
        indices = torch.randint(0, len(test_dataloader.dataset), (6,))
        images, labels = zip(*[test_dataloader.dataset[i] for i in indices])

        classes = ('Buildings', 'Forest', 'Mountains', 'Glacier', 'Sea', 'Street')

        # 显示图片及其标签
        fig, axes = plt.subplots(2, 3, figsize=(15, 3))
        for i, (image, label) in enumerate(zip(images, labels)):
            predict = torch.argmax(model(image.unsqueeze(0).to(device))).cpu()
            ax = axes[i // 3, i % 3]
            ax.imshow(image.permute(1, 2, 0))
            ax.set_title(f'Label: {classes[label]} \n Predict: {classes[predict]}')
            ax.axis('off')

        plt.show()

        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.arange(6))
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()

    return accuracy,macro_precision,macro_recall,macro_f1,auroc,conf_matrix