import torch
import random
import numpy as np
import visdom
import json
import csv


# 设置随机种子
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 保存Visdom数据
def saveVisdomData(win, env, fileName, mode='w'):
    assert mode=='w' or mode=='a'
    viz = visdom.Visdom()
    win_data = viz.get_window_data(win, env)
    pre_data = json.loads(win_data)
    x = pre_data["content"]["data"][0]["x"]  # x坐标的值
    y1 = pre_data["content"]["data"][0]["y"]  # 曲线１
    y2 = pre_data["content"]["data"][1]["y"]  # 曲线２
    assert len(x)==len(y1)==len(y2)
    with open(fileName, mode) as f:
        writer = csv.writer(f)
        for i in range(len(x)):
            writer.writerow([x[i], y1[i], y2[i]])