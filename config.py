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
USE_BERT = False           # 是否使用BERT情感分析（需先训练或下载模型）

# LDA主题模型配置
N_TOPICS = 5               # 主题数量（可设为None自动选择）
N_TOP_WORDS = 10           # 每个主题展示的关键词数量
AUTO_SELECT_TOPICS = True  # 是否自动选择最优主题数

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

# 以下旧路径变量保留，以防其他代码引用
CLEANED_COMMENTS_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_comments.csv")
PROCESSED_VIDEOS_PATH = os.path.join(BASE_DIR, "data", "processed", "videos", "cleaned_videos.csv")

# 新增路径配置
STOPWORDS_PATH = os.path.join(BASE_DIR, "data", "stopwords.txt")           # 停用词文件
USER_DICT_PATH = os.path.join(BASE_DIR, "data", "user_dict.txt")           # 用户自定义词典
POS_DICT_PATH = os.path.join(BASE_DIR, "data", "sentiment", "positive_words.txt")   # 积极词词典
NEG_DICT_PATH = os.path.join(BASE_DIR, "data", "sentiment", "negative_words.txt")   # 消极词词典
DEGREE_DICT_PATH = os.path.join(BASE_DIR, "data", "sentiment", "degree_words.txt")  # 程度副词词典
NEGATION_DICT_PATH = os.path.join(BASE_DIR, "data", "sentiment", "negation_words.txt") # 否定词词典
BERT_MODEL_PATH = os.path.join(BASE_DIR, "models", "bert-base-chinese")    # BERT模型路径


class SaveData:
    """保存数据的工具类（保持不变）"""

    def __init__(self, data, result_type, add_some=None, filename=None, add_timestamp=True, keyword=None):
        self.data = data
        self.result_type = result_type
        self.add_some = add_some
        self.filename = filename          # 仅 result_type == "result" 时使用
        self.add_timestamp = add_timestamp
        self.keyword = keyword

    def _add_some_(self, filepath: str):
        add = self.add_some
        dirname, filename = os.path.split(filepath)
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_{add}{ext}"
        return os.path.join(dirname, new_filename)

    def _add_timestamp_to_filename(self, filepath):
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
            full_path = CLEANED_COMMENTS_PATH

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

# 新增：打印步骤辅助函数
def print_step(step_name, is_start=True):
    if is_start:
        print(f"\n{'='*60}\n【{step_name}】开始\n{'='*60}")
    else:
        print(f"\n{'='*60}\n【{step_name}】完成\n{'='*60}\n")

def print_table(df, title=None):
    if title:
        print(f"\n{title}")
    print(df.to_string(index=False))

def ensure_directories():
    """确保所有需要的目录存在"""
    dirs = [
        os.path.dirname(SEGMENTED_VIDEOS_PATH),
        os.path.dirname(SEGMENTED_COMMENTS_PATH),
        RESULTS_PATH,
        os.path.dirname(SALES_DATA_PATH),
        RAW_DATA_DIR,
        os.path.join(BASE_DIR, "data", "processed", "videos"),
        os.path.join(BASE_DIR, "data", "processed", "danmaku"),
    ]
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)