import torch
from PIL import Image
import os
import os.path
import numpy as np
import pickle

from torch.fx.experimental.unification import reify
from torch.utils.data import Dataset
from torchvision import transforms
from util.utils import *
from configration import CAN_DATA_DIR,LabelEncoder_PATH

class CICIDS(Dataset):
    def __init__(self,root=CAN_DATA_DIR,train=True,base_sess=True,
                 transform=None,index_path=None,index=None):
        """

        Args:
            root: 根路径
            train: 是否时训练模式
            base_sess: 是否是基类
            transform: 是否有transform
            index_path: session_x.txt的位置，主要是SelectBaseClass和SelectNewClass
            index: 需要的类，主要是SelectTestClass调用
        """
        self.root = os.path.expanduser(root)
        self.init_index_path = index_path if not index_path else os.path.join(BASE_DIR,r"data\index_list\cicids2017\session_1.txt")
        if train:
            self.transform = transforms.Compose(
                transforms.Resize([6*6])
            )
            self.lebalencoder = self.LoadLabelEncoder(LabelEncoder_PATH)
            if base_sess:
                self.data,self.labels = self.SelectBaseClass(index_path)
            else:
                self.data,self.labels = self.SelectNewClass(index_path,LabelEncoder_PATH)
        else:
            self.data,self.labels = self.SelectTestClass(index_path)
        pass
    def __getitem__(self, index):
        """

        Args:
            index:

        Returns: (image, target) where target is index of the target class.

        """
        img, target = self.data[index], self.labels[index]
        # reshape 为 [1,8,8] pytorch中要求的[C,H,W] c是通道数
        img_reshape = img.reshape(1, 8, 8)
        # 转为tensor
        img_tensor = torch.from_numpy(img_reshape)
        # 归一化
        img_tensor = img_tensor.float().div(255)

        return img_tensor, target
        pass
    def __len__(self):
        return len(self.data)
        pass

    def SelectBaseClass(self,session_path):
        """
        选取基本类，在初始化时调用，由路径控制
        Returns:

        """
        data,labels = CEC_read_img(session_path)
        return data,self.lebalencoder.transform(labels)

    def SelectNewClass(self,session_path,label_path):
        """
        选取增类，在初始化时调用，由路径控制
        Returns:

        """
        data,labels = CEC_read_img(session_path)
        le = self.AddLabelEncoder(label_path,labels)
        return data,self.lebalencoder.transform(labels)
        pass
    def SelectTestClass(self,session_path):
        """
        选取特定的类，初始化时调用，由传入的类别控制，暂定为list,后面整理一个test.txt
        用以测试
        Returns:

        """
        data, labels = CEC_read_img(session_path)
        return data, self.lebalencoder.transform(labels)

        pass
    def LoadLabelEncoder(self,file_path):
        """
        没有则创建
        Args:
            file_path:

        Returns: 一个Label Encoder类

        """
        if not os.path.exists(file_path):
            self.InitLabelEncoder(file_path)
        label_encoder = read_pickle(file_path)
        if label_encoder is not None:
            return label_encoder
        else:
            print("请确认LabelEncoder的存在:\t{}".format(file_path))
            return None
    def InitLabelEncoder(self,file_path):
        """
        读取session_1的值并存入路径
        Args:
            file_path:

        Returns:

        """
        index_path = os.path.join(BASE_DIR,r"data\index_list\cicids2017\session_1.txt")
        _,labels = CEC_read_img(index_path)
        le = LabelEncoder()
        le.fit(labels)
        creat_pickle(file_path,le)
    def AddLabelEncoder(self,file_path,add_list):
        """
        增加类，并编码
        Args:
            file_path:
            add_list:

        Returns:

        """
        le = self.LoadLabelEncoder(file_path)
        all_list = le.classes_.tolist() + add_list
        le.fit(all_list)
        #写入并返回
        creat_pickle(file_path,le)
        self.lebalencoder = le
        return le




class MyDateset(Dataset):
    def __init__(self,data_dir='./data/image',train=True):
        #替换路径为绝对路径，做路径兼容
        self.data_path = os.path.expanduser(data_dir)
        self.data,self.labels = try_read_img(self.data_path)
        #对Label进行编码
        le = LabelEncoder()
        self.labels = le.fit_transform(self.labels)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """

        :param index: int
        :return: (image, target) where target is index of the target class.
        """
        img,target = self.data[index],self.labels[index]
        #reshape 为 [1,8,8] pytorch中要求的[C,H,W] c是通道数
        img_reshape = img.reshape(1,8,8)
        #转为tensor
        img_tensor = torch.from_numpy(img_reshape)
        #归一化
        img_tensor = img_tensor.float().div(255)

        return img_tensor,target