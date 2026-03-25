# config.py
# 配置文件，集中管理所有参数
import os
import pandas as pd
from datetime import datetime


# B站采集配置
KEYWORDS = ["小米SU7", "小米汽车SU7", "SU7"]  # 搜索关键词
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

RAW_VIDEOS_PATH = os.path.join(BASE_DIR, "data", "raw", "videos.csv")
RAW_DANMAKU_PATH = os.path.join(BASE_DIR, "data", "raw", "danmaku.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_comments.csv")
SALES_DATA_PATH = os.path.join(BASE_DIR, "data", "sales", "xiaomi_su7_sales.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "results")


def _add_some_(add: str, filepath: str):
    dirname, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_{add}{ext}"
    return os.path.join(dirname, new_filename)


def _add_timestamp_to_filename(filepath):
    """在文件名中插入时间戳（保留扩展名）"""
    dirname, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{name}_{timestamp}{ext}"
    return os.path.join(dirname, new_filename)


def save_data(data,
              result_type: str,
              add_some: str,
              filename: str = None,
              add_timestamp: bool = True,
              keyword = None,
              **kwargs) -> None:
    """
    根据结果类型保存数据到对应路径，自动添加时间戳。

    参数:
        data: 要保存的数据 (DataFrame, str, 或列表)
        result_type: 结果类型，支持 'videos', 'danmaku', 'processed', 'sales', 'result'
        add_some: 文件名添加细分
        filename: 当 result_type 为 'result' 时，必须提供文件名；其他类型忽略此参数
        add_timestamp: 是否在文件名中添加时间戳 (默认 True)
        keyword: 关键词，用于在 data/raw 下创建子文件夹（仅对 'videos' 和 'danmaku' 有效）
        **kwargs: 传递给具体保存方法的额外参数 (如 index=False 等)
    """
    # 如果 data 是列表，先转为 DataFrame
    if isinstance(data, list):
        if not data:
            print("警告：数据为空，跳过保存")
            return
        data = pd.DataFrame(data)

    # 确定最终保存路径
    if result_type == "videos":
        if keyword is not None:
            # 使用关键词子目录：data/raw/{keyword}/videos.csv
            base_dir = os.path.join(BASE_DIR, "data", "raw", keyword)
            # 先判断 keyword 文件夹是否存在，不存在则创建
            os.makedirs(base_dir, exist_ok=True)
            full_path = os.path.join(base_dir, "videos.csv")
        else:
            full_path = RAW_VIDEOS_PATH
    elif result_type == "danmaku":
        if keyword is not None:
            base_dir = os.path.join(BASE_DIR, "data", "raw", keyword)
            os.makedirs(base_dir, exist_ok=True)
            full_path = os.path.join(base_dir, "danmaku.csv")
        else:
            full_path = RAW_DANMAKU_PATH
    elif result_type == "processed":
        full_path = PROCESSED_DATA_PATH
    elif result_type == "sales":
        full_path = SALES_DATA_PATH
    elif result_type == "result":
        if not filename:
            raise ValueError("result_type 为 'result' 时必须提供 filename 参数")
        full_path = os.path.join(RESULTS_PATH, filename)
    else:
        raise ValueError(f"不支持的结果类型: {result_type}")

    # 确保父目录存在（对其他类型也做保障）
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    # 如果需要添加细分标识，则修改文件名
    if add_some is not None:
        full_path = _add_some_(add_some, full_path)

    # 如果需要添加时间戳，则修改文件名
    if add_timestamp:
        full_path = _add_timestamp_to_filename(full_path)

    # 保存数据
    if isinstance(data, pd.DataFrame):
        data.to_csv(full_path, **kwargs)
    elif isinstance(data, str):
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(data)
    else:
        raise TypeError(f"不支持的数据类型: {type(data)}")

    # 打印保存信息（包含类型和相对路径）
    relative_path = os.path.relpath(full_path, BASE_DIR)
    print(f"{result_type}数据已保存至：{relative_path}")