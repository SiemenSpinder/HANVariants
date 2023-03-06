# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import os
from urllib.parse import urlsplit

save_path = r'C:\Users\sieme\Documents\CBS\NonSustain\FullHTML\htmloutput'

if not os.path.exists(save_path):
    os.makedirs(save_path)

class fullhtmlPipeline(object):
    def process_item(self, item, spider):
        page = urlsplit(item['url'])
        replaced = str.replace(page.path, '/', '-')
        filename = page.hostname + replaced + '.html'
        with open(os.path.join(save_path, filename), 'wb') as f:
            f.write(item['html'])
