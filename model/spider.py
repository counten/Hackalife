# -*- coding:utf-8 -*-

# 百度图片是动态js加载的action，所以使用python的request库添加参数动态获取页面。
# 将搜索关键词作为keyword参数传递，抓取想要数目的页面图片
import os
import re
import requests
from back.configs import Configs


class Spider:
    def __init__(self, source_url, path_save):
        self.url = source_url
        self.path_save = path_save
        self.urls = list()

    def get_int_pages(self, keyword, pages):
        params = []
        for i in range(30, 30 * pages + 30, 30):
            params.append({
                'tn': 'resultjson_com',
                'ipn': 'rj',
                'ct': '201326592',
                'is': '',
                'fp': 'result',
                'queryWord': keyword,
                'cl': '2',
                'lm': '-1',
                'ie': 'utf-8',
                'oe': 'utf-8',
                'st': '-1',
                'ic': '0',
                'word': keyword,
                'face': '0',
                'istype': '2',
                'nc': '1',
                'pn': i,
                'rn': '30'
            })
        for i in params:
            content = requests.get(self.url, params=i).text
            # 正则获取图片链接
            img_urls = re.findall(r'"thumbURL":"(.*?)"', content)
            self.urls.append(img_urls)

    # 将获取的图片保存在本地文件夹中
    def fetch_img(self):
        if not os.path.exists(self.path_save):
            os.mkdir(self.path_save)
        x = 0
        for list in self.urls:
            for i in list:
                print("=====downloading %d/1200=====" % (x + 1))
                ir = requests.get(i)
                open(self.path_save + '%d.jpg' % x, 'wb').write(ir.content)
                x += 1


if __name__ == '__main__':
    conf = Configs()
    spider = Spider(conf.spider_source_url, conf.dir_pic + "/tomato/")
    # 获取40页西红柿的图片
    spider.get_int_pages('西红柿', 1)
    spider.fetch_img()
