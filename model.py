import torch
import torch.nn as nn
import torch.nn.functional as F

class resblk(nn.Module):
    def __init__(self,in_dim,out_dim,stride=1):
        super(resblk,self).__init__()
        self.conv1=nn.Conv2d(in_dim,out_dim,kernel_size=(3,3),stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_dim)
        self.conv2=nn.Conv2d(out_dim,out_dim,kernel_size=(3,3),stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_dim)
        if in_dim!=out_dim or stride!=1:
            self.downsample=nn.Sequential(nn.Conv2d(in_dim,out_dim ,kernel_size=(1,1),stride=stride,padding=0,bias=False),
                                          nn.BatchNorm2d(out_dim))
            self.changeInput=True
        else:
            self.changeInput=False


    def forward(self,x):
        y=self.conv1(x)
        y=self.bn1(y)
        y=F.relu(y)
        y=self.conv2(y)
        y=self.bn2(y)
        if self.changeInput:
            x=self.downsample(x)
        y=y+x
        y=F.relu(y)
        return y



class resnet18(nn.Module):
    def __init__(self,num_class):
        super(resnet18,self).__init__()
        self.conv1=nn.Conv2d(3,64,(7,7),2,3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.max_pool=nn.MaxPool2d((3,3),2,1)
        self.layer1=nn.Sequential(
            resblk(64, 64),
            resblk(64, 64),
        )
        self.layer2=nn.Sequential(
            resblk(64, 128,2),
            resblk(128, 128),
        )
        self.layer3=nn.Sequential(
            resblk(128, 256,2),
            resblk(256, 256),
        )
        self.layer4 = nn.Sequential(
            resblk(256, 512, 2),
            resblk(512, 512),
        )
        self.avg_pool=nn.AvgPool2d((7,7))
        self.fc=nn.Linear(512,num_class)
        self.softmax=nn.Softmax(1)


    def forward(self,x):
        x=self.conv1(x)
        x=self.max_pool(x)
        x=self.bn1(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.avg_pool(x)
        x=x.reshape(-1,512)
        x=self.fc(x)
        x=self.softmax(x)
        return x

class resnet34(nn.Module):
    def __init__(self,num_class):
        super(resnet34,self).__init__()
        self.conv1=nn.Conv2d(3,64,(7,7),2,3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.max_pool=nn.MaxPool2d((3,3),2,1)
        self.layer1=nn.Sequential(
            resblk(64, 64),
            resblk(64, 64),
            resblk(64, 64),
        )
        self.layer2=nn.Sequential(
            resblk(64, 128,2),
            resblk(128, 128),
            resblk(128, 128),
            resblk(128, 128),
        )
        self.layer3=nn.Sequential(
            resblk(128, 256,2),
            resblk(256, 256),
            resblk(256, 256),
            resblk(256, 256),
            resblk(256, 256),
            resblk(256, 256),
        )
        self.layer4 = nn.Sequential(
            resblk(256, 512, 2),
            resblk(512, 512),
            resblk(512, 512),
        )
        self.avg_pool=nn.AvgPool2d((7,7))
        self.fc=nn.Linear(512,num_class)
        self.softmax=nn.Softmax(1)


    def forward(self,x):
        x=self.conv1(x)
        x=self.max_pool(x)
        x=self.bn1(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.avg_pool(x)
        x=x.reshape(-1,512)
        x=self.fc(x)
        x=self.softmax(x)
        return x
