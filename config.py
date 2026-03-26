# config.py
# 配置文件，集中管理所有参数

import os
import pandas as pd
from datetime import datetime

# B站采集配置
KEYWORDS = ["SU7", "小米SU7", "小米汽车SU7"]  # 搜索关键词
START_DATE = "2024-04-01"  # 采集开始时间
END_DATE = "2026-01-31"    # 采集结束时间

# 情感分析配置
POSITIVE_THRESHOLD = 0.6   # 积极情感阈值
NEGATIVE_THRESHOLD = 0.4   # 消极情感阈值

# LDA主题模型配置
N_TOPICS = 5               # 主题数量
N_TOP_WORDS = 10           # 每个主题展示的关键词数量

# --- 路径配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 原始数据目录
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

# 分词输出路径（固定文件名）
SEGMENTED_VIDEOS_PATH = os.path.join(BASE_DIR, "data", "processed", "segmented_videos.csv")
SEGMENTED_COMMENTS_PATH = os.path.join(BASE_DIR, "data", "processed", "segmented_comments.csv")

# 销量数据
SALES_DATA_PATH = os.path.join(BASE_DIR, "data", "sales", "xiaomi_su7_sales.csv")

# 结果目录
RESULTS_PATH = os.path.join(BASE_DIR, "results")

# 以下旧路径变量保留，以防其他代码引用（但新流程已不再使用）
CLEANED_COMMENTS_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_comments.csv")
PROCESSED_VIDEOS_PATH = os.path.join(BASE_DIR, "data", "processed", "videos", "cleaned_videos.csv")


class SaveData:
    """保存数据的工具类"""

    def __init__(self, data, result_type, add_some=None, filename=None, add_timestamp=True, keyword=None):
        self.data = data
        self.result_type = result_type
        self.add_some = add_some
        self.filename = filename          # 仅 result_type == "result" 时使用
        self.add_timestamp = add_timestamp
        self.keyword = keyword

    def _add_some_(self, filepath: str):
        """在文件名中插入细分标识"""
        add = self.add_some
        dirname, filename = os.path.split(filepath)
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_{add}{ext}"
        return os.path.join(dirname, new_filename)

    def _add_timestamp_to_filename(self, filepath):
        """在文件名中插入时间戳"""
        dirname, filename = os.path.split(filepath)
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{name}_{timestamp}{ext}"
        return os.path.join(dirname, new_filename)

    def _get_full_path(self):
        result_type = self.result_type
        keyword = self.keyword
        add_some = self.add_some
        add_timestamp = self.add_timestamp

        # 构造文件名
        if result_type == "videos":
            base_name = f"videos_{keyword}" if keyword else "videos"
            parts = [base_name]
            if add_timestamp:
                parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
            if add_some is not None:
                parts.append(add_some)
            filename = "_".join(parts) + ".csv"
            if keyword:
                base_dir = os.path.join(RAW_DATA_DIR, keyword, "videos")
            else:
                base_dir = RAW_DATA_DIR
            full_path = os.path.join(base_dir, filename)

        elif result_type == "danmaku":
            base_name = f"danmaku_{keyword}" if keyword else "danmaku"
            parts = [base_name]
            if add_timestamp:
                parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
            if add_some is not None:
                parts.append(add_some)
            filename = "_".join(parts) + ".csv"
            if keyword:
                base_dir = os.path.join(RAW_DATA_DIR, keyword, "danmaku")
            else:
                base_dir = RAW_DATA_DIR
            full_path = os.path.join(base_dir, filename)

        elif result_type == "processed":
            full_path = CLEANED_COMMENTS_PATH   # 旧逻辑，保留

        elif result_type == "sales":
            full_path = SALES_DATA_PATH

        elif result_type == "result":
            if not self.filename:
                raise ValueError("result_type 为 'result' 时必须提供 filename 参数")
            full_path = os.path.join(RESULTS_PATH, self.filename)

        else:
            raise ValueError(f"不支持的结果类型: {result_type}")

        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        return full_path

    def save(self, **kwargs):
        data = self.data
        if isinstance(data, list):
            if not data:
                print("警告：数据为空，跳过保存")
                return
            data = pd.DataFrame(data)

        full_path = self._get_full_path()

        if isinstance(data, pd.DataFrame):
            data.to_csv(full_path, **kwargs)
        elif isinstance(data, str):
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(data)
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")

        relative_path = os.path.relpath(full_path, BASE_DIR)
        print(f"{self.result_type}数据已保存至：{relative_path}")