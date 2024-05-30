import torch
import torch.nn as nn
from torch import optim
from tensorboardX import SummaryWriter
import datetime
from data_utils import *
from model import *
from utils import *
from train_utils import *
import old
import os

if __name__=='__main__':
    # 参数设置
    seed = 11  # 随机种子
    model_path = ""  # 模型权重路径，空字符串表示使用初始化模型
    pretrained = False  # 是否加载主干网络预训练模型
    pretrain_path="pretrain/resnet18-5c106cde.pth"
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')

    # 训练参数
    Freeze=False # 是否冻结训练
    Epoch=30
    freeze_epoch=3
    batch_size=64
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    momentum = 0.9
    lr_decay_type = 'cos'
    weight_decay = 0
    save_period = 5
    save_dir = 'models'
    eval_flag = True
    eval_period = 5
    log_dir=os.path.join('logs', time_str)
    seed_everything(seed)

    # os.mkdir(os.path.join('tensorboard',time_str))


    # 加载模型
    net=resnet18(6)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if pretrained:
        print('Load weights {}.'.format(pretrain_path))
        model_dict = net.state_dict()
        pretrained_dict = torch.load(pretrain_path)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        net.load_state_dict(model_dict)

        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        model_dict = net.state_dict()
        pretrained_dict = torch.load(model_path)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        net.load_state_dict(model_dict)

        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    writer = SummaryWriter(log_dir)
    writer.add_graph(net, torch.rand(1, 3, 224, 224))
    writer.close()
    net.to(device)


    nbs = 16
    lr_limit_max = 1e-3
    lr_limit_min = 1e-5
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    optimizer = optim.Adam(net.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay)
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

    # 数据导入
    dir='scene_classification'
    data_train = ImgDataset(os.path.join(dir,'train_data.csv'),True,[1,2,3],[1,1,1])
    print(data_train.count_label())
    data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    data_val = ImgDataset(os.path.join(dir,'val_data.csv'),False,[1,2,3],[1,1,1])
    print(data_val.count_label())
    data_val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=True)
    data_test = ImgDataset(os.path.join(dir,'test_data.csv'),False,[1,2,3],[1,1,1])
    print(data_test.count_label())
    data_test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)

    if Freeze:
        for k, v in net.named_parameters():
            if not k.startswith('fc'):
                v.requires_grad=False

    # 训练
    loss = 0
    val_loss = 0
    Unfreeze_flag = False
    for epoch in range(Epoch):
        if not Unfreeze_flag and epoch>=freeze_epoch:
            for k, v in net.named_parameters():
                if not k.startswith('fc'):
                    v.requires_grad = True
            Unfreeze_flag=True
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        loss, val_loss_temp = fit_one_epoch(net, optimizer=optimizer, epoch=epoch,
                                            train_dataloader=data_train_loader, val_dataloader=data_val_loader,
                                            device=device, save_dir=save_dir,log_dir=log_dir,
                                            Epoch_num=Epoch, eval_period=eval_period, save_period=save_period)
        if val_loss_temp:
            val_loss = val_loss_temp
            writer.add_scalar('val_loss',val_loss,epoch)

    # 测试
    test_model(net,data_test_loader,device,graph=True)












