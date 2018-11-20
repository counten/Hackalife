# -*- coding:utf-8 -*-

# 图片压缩与转换，图片格式化（特征与标签的表示方式）
# 因使用的是CNN识别cifar10的数据集，需要将图片转化为32*32 size大小的图片，并全部转化为RGB模式的图片：
# read file then return a (1,3072) array
import os
import pickle
import numpy
from PIL import Image
import matplotlib.image as plot_img

from back.configs import Configs


class PreProcess:
    def __init__(self, dir_pic, dir_save):
        # 图片根目录
        self.dir_pic = dir_pic
        # 输出文件目录
        self.path_save = dir_save
        # 图片路径
        self.path_pic = []

        self.label = []
        self.label_name = []
        self.all_arr = []
        self.get_pic_path()

    # 其中label是在读取文件夹的时候以文件夹名称划分：
    def get_pic_path(self):
        for i, dirs in enumerate(os.listdir(self.dir_pic)):
            if dirs != ".placeholder":
                self.label_name.append(dirs)
                for f in os.listdir((os.path.join(self.dir_pic, dirs))):
                    self.label.append(i)
                    img_path = os.path.join(os.path.join(self.dir_pic, dirs), f)
                    self.path_pic.append(img_path)

    # 读取图片
    def read_file(self):
        for file in self.path_pic:
            img = Image.open(file)
            img.convert('RGB')  # 转化为RGB
            img = img.resize((32, 32))  # 压缩为32*32
            try:
                # 将此图像拆分为单独的波段
                red, green, blue = img.split()
                red_arr = plot_img.pil_to_array(red)
                green_arr = plot_img.pil_to_array(green)
                blue_arr = plot_img.pil_to_array(blue)

                r_arr = red_arr.reshape(1024)
                g_arr = green_arr.reshape(1024)
                b_arr = blue_arr.reshape(1024)

                res = numpy.concatenate((r_arr, g_arr, b_arr))
                if not self.all_arr:
                    self.all_arr = res
                else:
                    self.all_arr = numpy.concatenate((self.all_arr, res))
            except ValueError:
                print(file)
                # img.show()

    # 压缩与转化完成后，将其与label合并后以字典的形式存储并写入文件：
    def save_pickle(self):
        print("=====saving picture, please wait=====")

        dic = {'label': self.label, 'data': self.all_arr, 'label_name': self.label_name}
        file_path = self.path_save + "/data_batch_test"
        with open(file_path, 'wb') as f:
            pickle.dump(dic, f)
        print("=====save mode end=====")


if __name__ == '__main__':
    conf = Configs()
    pp = PreProcess(conf.dir_pic, conf.dir_train_file)
    pp.read_file()
    pp.save_pickle()
