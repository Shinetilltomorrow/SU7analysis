# data_preprocessing/clean.py
import re
import pandas as pd
import os
import config


class BaseCleaner:
    def __init__(self, raw_path, output_path=None):
        self.raw_path = raw_path
        self.keyword = self._extract_keyword()
        if output_path is None:
            self.output_path = self._default_output_path()
        else:
            self.output_path = output_path
        self.df = None

    def _extract_keyword(self):
        """从原始文件路径中提取关键词"""
        norm_path = os.path.normpath(self.raw_path)
        parts = norm_path.split(os.sep)
        for i, part in enumerate(parts):
            if part == 'raw' and i+2 < len(parts):
                candidate = parts[i+2]
                if candidate in config.KEYWORDS:
                    return candidate
        filename = os.path.basename(self.raw_path)
        for kw in config.KEYWORDS:
            if kw in filename:
                return kw
        return "未知"

    def _default_output_path(self):
        subdir = getattr(self, 'subdir', 'processed')
        output_dir = os.path.join(config.PROCESSED_DATA_DIR, subdir, self.keyword)
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(self.raw_path)
        return os.path.join(output_dir, filename)

    def load_data(self):
        self.df = pd.read_csv(self.raw_path, encoding='utf-8-sig')
        config.logger.info(f"[{self.keyword}] 原始数据量: {len(self.df)} 条")

    def remove_duplicates(self, subset):
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        after = len(self.df)
        config.logger.info(f"[{self.keyword}] 去除重复: {before} -> {after}")

    def filter_content(self, column, patterns):
        for pattern in patterns:
            before = len(self.df)
            self.df = self.df[~self.df[column].str.contains(pattern, na=False, regex=True)]
            after = len(self.df)
            config.logger.info(f"[{self.keyword}] 过滤 '{pattern}': {before} -> {after}")

    def clean_text(self, column, new_column):
        def clean_single(text):
            if pd.isna(text):
                return ""
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', '', text)
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''、]', '', text)
            text = re.sub(r'\s+', '', text)
            text = re.sub(r'(.)\1{2,}', r'\1', text)
            return text.strip()

        self.df[new_column] = self.df[column].apply(clean_single)
        self.df = self.df[self.df[new_column] != '']

    def save(self):
        self.df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        config.logger.info(f"[{self.keyword}] 清洗后数据保存到 {self.output_path}，共 {len(self.df)} 条")


class VideoCleaner(BaseCleaner):
    subdir = 'videos'
    def run(self):
        print(f"\n{'='*60}")
        print(f"▶ 开始清洗视频数据: 关键词 [{self.keyword}]")
        print(f"{'='*60}")
        config.logger.info(f"[{self.keyword}] 检测到视频数据，开始清洗...")
        self.load_data()
        self.remove_duplicates(subset=['bv_id'])
        self.filter_content(column='title', patterns=[
            r'^[\d\W]+$', r'^[a-zA-Z]+$', r'^.{0,2}$',
            r'广告', r'加微信', r'宣传', r'推广'
        ])
        self.clean_text(column='title', new_column='cleaned_title')
        self.save()
        print(f"\n{'='*60}")
        print(f"✔ 视频数据清洗完成: 关键词 [{self.keyword}]")
        print(f"{'='*60}\n")


class DanmakuCleaner(BaseCleaner):
    subdir = 'danmaku'
    def run(self):
        print(f"\n{'='*60}")
        print(f"▶ 开始清洗弹幕数据: 关键词 [{self.keyword}]")
        print(f"{'='*60}")
        config.logger.info(f"[{self.keyword}] 检测到弹幕数据，开始清洗...")
        self.load_data()
        self.remove_duplicates(subset=['text', 'bv_id'])
        self.filter_content(column='text', patterns=[
            r'^[\d\W]+$', r'^[a-zA-Z]+$', r'^.{0,2}$',
            r'广告', r'加微信', r'求赞', r'关注', r'三连'
        ])
        self.clean_text(column='text', new_column='cleaned_text')
        self.save()
        print(f"\n{'='*60}")
        print(f"✔ 弹幕数据清洗完成: 关键词 [{self.keyword}]")
        print(f"{'='*60}\n")


def detect_data_type(file_path):
    df_sample = pd.read_csv(file_path, encoding='utf-8-sig', nrows=1)
    return 'danmaku' if 'text' in df_sample.columns else 'video'


def clean_data(file_path=None, output_path=None):
    if file_path is None:
        return None
    else:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        data_type = detect_data_type(file_path)
        cleaner = VideoCleaner(file_path, output_path) if data_type == 'video' else DanmakuCleaner(file_path, output_path)
        cleaner.run()
        return cleaner.output_path


def get_latest_csv_by_type(folder_path):
    pass


def auto_clean_latest(video_root=None, danmaku_root=None):
    pass