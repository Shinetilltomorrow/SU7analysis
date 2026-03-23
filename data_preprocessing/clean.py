# 弹幕清洗
# data_preprocessing/clean.py
# 弹幕清洗与过滤

import re
import pandas as pd
import config


class DataCleaner:
    """数据清洗类"""

    def __init__(self, raw_path, output_path):
        self.raw_path = raw_path
        self.output_path = output_path
        self.df = None

    def load_data(self):
        """加载原始数据"""
        self.df = pd.read_csv(self.raw_path, encoding='utf-8-sig')
        print(f"原始数据量: {len(self.df)} 条")

    def remove_duplicates(self):
        """去除重复弹幕"""
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=['text', 'bv_id'], keep='first')
        after = len(self.df)
        print(f"去除重复弹幕: {before} -> {after}")

    def filter_content(self):
        """过滤无关内容"""
        # 定义过滤规则
        patterns = [
            r'^[\d\W]+$',  # 纯数字或特殊符号
            r'^[a-zA-Z]+$',  # 纯英文
            r'^.{0,2}$',  # 过短文本（少于2个字符）
            r'广告',  # 广告关键词
            r'加微信',  # 广告关键词
        ]

        for pattern in patterns:
            before = len(self.df)
            self.df = self.df[~self.df['text'].str.contains(pattern, na=False, regex=True)]
            after = len(self.df)
            print(f"过滤 {pattern}: {before} -> {after}")

    def clean_text(self):
        """清洗文本"""

        def clean_single(text):
            if pd.isna(text):
                return ""
            # 去除HTML标签
            text = re.sub(r'<[^>]+>', '', text)
            # 去除URL
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', '', text)
            # 去除特殊符号（保留中文、英文、数字、常用标点）
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''、]', '', text)
            # 去除多余空格
            text = re.sub(r'\s+', '', text)
            return text.strip()

        self.df['cleaned_text'] = self.df['text'].apply(clean_single)
        # 删除清洗后为空的行
        self.df = self.df[self.df['cleaned_text'] != '']

    def run(self):
        """执行清洗流程"""
        self.load_data()
        self.remove_duplicates()
        self.filter_content()
        self.clean_text()
        self.save()

    def save(self):
        """保存清洗后的数据"""
        self.df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        print(f"清洗后数据保存到 {self.output_path}，共 {len(self.df)} 条")


