# -*- coding:utf-8 -*-
import os

# 项目基本路径
import threading

project_dir = os.path.dirname(os.path.abspath(__file__))


class Configs(object):
    # 单例模式
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(Configs, "_instance"):
            with Configs._instance_lock:
                if not hasattr(Configs, "_instance"):
                    Configs._instance = object.__new__(cls)
        return Configs._instance

    def __init__(self):
        # 文件路径
        self.dir_log = project_dir + "/data/logs"
        self.dir_model = project_dir + "/data/model"
        self.dir_pic = project_dir + "/data/pic"
        self.dir_train_file = project_dir + "/data/train_file"

        # 爬虫
        self.spider_source_url = 'https://image.baidu.com/search/acjson'
