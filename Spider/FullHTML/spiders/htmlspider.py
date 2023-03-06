# -*- coding: utf-8 -*-
import scrapy
import pandas
import xlrd
import os
from scrapy.contrib.linkextractors import LinkExtractor
from scrapy.contrib.spiders import CrawlSpider, Rule
from urllib.parse import urlsplit
from FullHTML.items import MyItem
import collections
import tldextract

class HtmlSpider(CrawlSpider):

        path = r'C:\Users\sieme\Documents\CBS\NonSustain\FullHTML\notsustain.xlsx' 
        #read list of urls from excel, takes 'URL' column as input
        df = pandas.read_excel(path)
        df = df[pandas.notnull(df['URL'])]
        urls = df['URL'].values.tolist()

        scraped_count = collections.defaultdict(int)
        limit = 15

        #convert urls to domains
        domains = []
        for url in urls:
            tlresult = tldextract.extract(url)
            domains.append(tlresult.domain + '.' + tlresult.suffix)

        #list of allowed domains and start urls to crawl
        name = 'htmlspider'
        allowed_domains = domains
        start_urls = ['http://' + url for url in urls]

        #rule to crawl
        rules = (Rule(LinkExtractor(), callback='parse_item', process_request = 'process_request'), )

        def process_request(self, request):
                url = urlsplit(request.url)[1]
                if self.scraped_count[url] < self.limit:
                    self.scraped_count[url] += 1
                    return request
                else:
                    print('Limit reached for {}'.format(url))
                
        def parse_item(self, response):
                item = MyItem()
                item['url'] = response.url
                item['html'] = response.body
                return item


