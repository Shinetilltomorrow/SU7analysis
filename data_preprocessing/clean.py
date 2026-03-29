# data_preprocessing/clean.py
import re
import pandas as pd
import os
import config


class BaseCleaner:
    def __init__(self, raw_path, output_path=None):
        self.raw_path = raw_path
        if output_path is None:
            self.output_path = self._default_output_path()
        else:
            self.output_path = output_path
        self.df = None

    def _default_output_path(self):
        subdir = getattr(self, 'subdir', 'processed')
        output_dir = os.path.join(config.BASE_DIR, 'data', 'processed', subdir)
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(self.raw_path)
        return os.path.join(output_dir, filename)

    def load_data(self):
        self.df = pd.read_csv(self.raw_path, encoding='utf-8-sig')
        config.logger.info(f"原始数据量: {len(self.df)} 条")

    def remove_duplicates(self, subset):
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        after = len(self.df)
        config.logger.info(f"去除重复: {before} -> {after}")

    def filter_content(self, column, patterns):
        for pattern in patterns:
            before = len(self.df)
            self.df = self.df[~self.df[column].str.contains(pattern, na=False, regex=True)]
            after = len(self.df)
            config.logger.info(f"过滤 {pattern}: {before} -> {after}")

    def clean_text(self, column, new_column):
        def clean_single(text):
            if pd.isna(text):
                return ""
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', '', text)
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''、]', '', text)
            text = re.sub(r'\s+', '', text)
            text = re.sub(r'(.)\1{2,}', r'\1', text)  # 简化重复字符
            return text.strip()

        self.df[new_column] = self.df[column].apply(clean_single)
        self.df = self.df[self.df[new_column] != '']

    def save(self):
        self.df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        config.logger.info(f"清洗后数据保存到 {self.output_path}，共 {len(self.df)} 条")


class VideoCleaner(BaseCleaner):
    subdir = 'videos'
    def run(self):
        config.logger.info("检测到视频数据，开始清洗...")
        self.load_data()
        self.remove_duplicates(subset=['bv_id'])
        self.filter_content(column='title', patterns=[
            r'^[\d\W]+$', r'^[a-zA-Z]+$', r'^.{0,2}$',
            r'广告', r'加微信', r'宣传', r'推广'
        ])
        self.clean_text(column='title', new_column='cleaned_title')
        self.save()


class DanmakuCleaner(BaseCleaner):
    subdir = 'danmaku'
    def run(self):
        config.logger.info("检测到弹幕数据，开始清洗...")
        self.load_data()
        self.remove_duplicates(subset=['text', 'bv_id'])
        self.filter_content(column='text', patterns=[
            r'^[\d\W]+$', r'^[a-zA-Z]+$', r'^.{0,2}$',
            r'广告', r'加微信', r'求赞', r'关注', r'三连'
        ])
        self.clean_text(column='text', new_column='cleaned_text')
        self.save()


def detect_data_type(file_path):
    df_sample = pd.read_csv(file_path, encoding='utf-8-sig', nrows=1)
    return 'danmaku' if 'text' in df_sample.columns else 'video'


def clean_data(file_path=None, output_path=None):
    if file_path is None:
        video_root = getattr(config, 'RAW_VIDEOS_PATH', '.')
        danmaku_root = getattr(config, 'RAW_DANMAKU_PATH', '.')
        auto_clean_latest(video_root=video_root, danmaku_root=danmaku_root)
        return None
    else:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        data_type = detect_data_type(file_path)
        cleaner = VideoCleaner(file_path, output_path) if data_type == 'video' else DanmakuCleaner(file_path, output_path)
        cleaner.run()
        return cleaner.output_path


def get_latest_csv_by_type(folder_path):
    csv_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    latest = {'video': None, 'danmaku': None}
    latest_mtime = {'video': 0, 'danmaku': 0}
    for file in csv_files:
        try:
            data_type = detect_data_type(file)
            mtime = os.path.getmtime(file)
            if mtime > latest_mtime[data_type]:
                latest_mtime[data_type] = mtime
                latest[data_type] = file
        except:
            continue
    return latest


def auto_clean_latest(video_root=None, danmaku_root=None):
    if video_root is None:
        video_root = config.RAW_DATA_DIR
    if danmaku_root is None:
        danmaku_root = config.RAW_DATA_DIR
    video_latest = get_latest_csv_by_type(video_root)
    danmaku_latest = get_latest_csv_by_type(danmaku_root)
    if video_latest['video']:
        config.logger.info(f"找到最新视频文件: {video_latest['video']}")
        clean_data(video_latest['video'])
    if danmaku_latest['danmaku']:
        config.logger.info(f"找到最新弹幕文件: {danmaku_latest['danmaku']}")
        clean_data(danmaku_latest['danmaku'])