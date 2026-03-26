# data_preprocessing/clean.py
# 自动识别文件夹中最新的视频/弹幕文件并清洗（支持关键词搜索子文件夹，读取config配置）

import re
import pandas as pd
import os
import glob
import config


class BaseCleaner:
    """基础清洗类，提供通用方法"""
    def __init__(self, raw_path, output_path=None):
        self.raw_path = raw_path
        # 如果没有指定输出路径，则使用默认路径
        if output_path is None:
            self.output_path = self._default_output_path()
        else:
            self.output_path = output_path
        self.df = None

    def _default_output_path(self):
        """生成默认输出路径：data/processed/{subdir}/原文件名.csv"""
        # 获取当前清洗器对应的子目录（由子类定义）
        subdir = getattr(self, 'subdir', 'processed')
        # 构建输出目录路径
        output_dir = os.path.join(config.BASE_DIR, 'data', 'processed', subdir)
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        # 获取原文件名（不含目录）
        filename = os.path.basename(self.raw_path)
        return os.path.join(output_dir, filename)

    def load_data(self):
        self.df = pd.read_csv(self.raw_path, encoding='utf-8-sig')
        print(f"原始数据量: {len(self.df)} 条")

    def remove_duplicates(self, subset):
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        after = len(self.df)
        print(f"去除重复: {before} -> {after}")

    def filter_content(self, column, patterns):
        for pattern in patterns:
            before = len(self.df)
            self.df = self.df[~self.df[column].str.contains(pattern, na=False, regex=True)]
            after = len(self.df)
            print(f"过滤 {pattern}: {before} -> {after}")

    def clean_text(self, column, new_column):
        def clean_single(text):
            if pd.isna(text):
                return ""
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', '', text)
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''、]', '', text)
            text = re.sub(r'\s+', '', text)
            return text.strip()

        self.df[new_column] = self.df[column].apply(clean_single)
        self.df = self.df[self.df[new_column] != '']

    def save(self):
        self.df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        print(f"清洗后数据保存到 {self.output_path}，共 {len(self.df)} 条")


class VideoCleaner(BaseCleaner):
    """视频数据清洗"""
    subdir = 'videos'  # 指定子目录

    def run(self):
        print("检测到视频数据，开始清洗...")
        self.load_data()
        self.remove_duplicates(subset=['bv_id'])
        self.filter_content(column='title', patterns=[
            r'^[\d\W]+$',
            r'^[a-zA-Z]+$',
            r'^.{0,2}$',
            r'广告',
            r'加微信',
        ])
        self.clean_text(column='title', new_column='cleaned_title')
        self.save()


class DanmakuCleaner(BaseCleaner):
    """弹幕数据清洗"""
    subdir = 'danmaku'  # 指定子目录

    def run(self):
        print("检测到弹幕数据，开始清洗...")
        self.load_data()
        self.remove_duplicates(subset=['text', 'bv_id'])
        self.filter_content(column='text', patterns=[
            r'^[\d\W]+$',
            r'^[a-zA-Z]+$',
            r'^.{0,2}$',
            r'广告',
            r'加微信',
        ])
        self.clean_text(column='text', new_column='cleaned_text')
        self.save()


def detect_data_type(file_path):
    """通过读取文件头判断数据类型"""
    df_sample = pd.read_csv(file_path, encoding='utf-8-sig', nrows=1)
    if 'text' in df_sample.columns:
        return 'danmaku'
    else:
        return 'video'


def clean_data(file_path=None, output_path=None):
    """统一入口：自动识别数据类型并清洗

    参数:
        file_path: 可选，要清洗的单个文件路径。若不指定，则从 config 中读取 RAW 路径，并清洗最新文件。
        output_path: 可选，输出文件路径（仅当清洗单个文件时有效）。
    """
    if file_path is None:
        # 未指定文件时，自动清洗 config 中定义的两个 RAW 目录下的最新文件
        video_root = getattr(config, 'RAW_VIDEOS_PATH', '.') if config else '.'
        danmaku_root = getattr(config, 'RAW_DANMAKU_PATH', '.') if config else '.'
        auto_clean_latest(video_root=video_root, danmaku_root=danmaku_root, keyword=None)
        return None
    else:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        data_type = detect_data_type(file_path)
        if data_type == 'video':
            cleaner = VideoCleaner(file_path, output_path)
        else:
            cleaner = DanmakuCleaner(file_path, output_path)
        cleaner.run()
        return cleaner.output_path


def find_folders_by_keyword(root_path, keyword):
    """递归查找所有包含关键词的文件夹路径（不区分大小写）"""
    matches = []
    if not os.path.exists(root_path):
        return matches
    for dirpath, dirnames, filenames in os.walk(root_path):
        if keyword.lower() in os.path.basename(dirpath).lower():
            matches.append(dirpath)
    return matches


def get_latest_csv_by_type(folder_path, keyword=None):
    """
    递归扫描文件夹（包括所有子文件夹），返回最新视频文件和最新弹幕文件的路径（按修改时间）
    """
    csv_files = []
    # 递归遍历 folder_path 下所有 .csv 文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                csv_files.append(full_path)

    latest = {'video': None, 'danmaku': None}
    latest_mtime = {'video': 0, 'danmaku': 0}

    for file in csv_files:
        try:
            data_type = detect_data_type(file)
            mtime = os.path.getmtime(file)
            if mtime > latest_mtime[data_type]:
                latest_mtime[data_type] = mtime
                latest[data_type] = file
        except Exception as e:
            print(f"跳过文件 {file}，读取失败: {e}")
            continue

    return latest


def auto_clean_latest(video_root=None, danmaku_root=None, keyword=None):
    """
    自动清洗最新的视频和弹幕文件，分别从指定根目录（默认读取config）中查找
    """
    if video_root is None:
        video_root = getattr(config, 'RAW_VIDEOS_PATH', '.') if config else '.'
    if danmaku_root is None:
        danmaku_root = getattr(config, 'RAW_DANMAKU_PATH', '.') if config else '.'

    video_latest = get_latest_csv_by_type(video_root, keyword)
    danmaku_latest = get_latest_csv_by_type(danmaku_root, keyword)

    if video_latest['video']:
        print(f"\n找到最新视频文件: {video_latest['video']}")
        clean_data(video_latest['video'])
    else:
        print("未找到视频文件")

    if danmaku_latest['danmaku']:
        print(f"\n找到最新弹幕文件: {danmaku_latest['danmaku']}")
        clean_data(danmaku_latest['danmaku'])
    else:
        print("未找到弹幕文件")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="清洗视频或弹幕数据")
    parser.add_argument("file", nargs="?", help="指定要清洗的文件路径（可选）")
    parser.add_argument("--keyword", "-k", help="搜索包含该关键词的文件夹，并清洗其中的最新文件")
    parser.add_argument("--video_path", "-v", help="视频文件根目录（覆盖config设置）")
    parser.add_argument("--danmaku_path", "-d", help="弹幕文件根目录（覆盖config设置）")

    args = parser.parse_args()

    if args.file:
        clean_data(args.file)
    else:
        auto_clean_latest(video_root=args.video_path, danmaku_root=args.danmaku_path, keyword=args.keyword)