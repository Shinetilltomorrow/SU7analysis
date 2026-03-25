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

RAW_VIDEO_PATH = os.path.join(BASE_DIR, "data", "raw", "videos.csv")
RAW_DANMAKU_PATH = os.path.join(BASE_DIR, "data", "raw", "danmaku.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_comments.csv")
SALES_DATA_PATH = os.path.join(BASE_DIR, "data", "sales", "xiaomi_su7_sales.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "results")



def _add_timestamp_to_filename(filepath):
    """在文件名中插入时间戳（保留扩展名）"""
    dirname, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{name}_{timestamp}{ext}"
    return os.path.join(dirname, new_filename)

def save_data(data, result_type, filename=None, add_timestamp=True, **kwargs):
    """
    根据结果类型保存数据到对应路径，自动添加时间戳。

    参数:
        data: 要保存的数据 (DataFrame, str 等)
        result_type: 结果类型，支持 'video', 'danmaku', 'processed', 'sales', 'result'
        filename: 当 result_type 为 'result' 时，必须提供文件名；其他类型忽略此参数
        add_timestamp: 是否在文件名中添加时间戳 (默认 True)
        **kwargs: 传递给具体保存方法的额外参数 (如 index=False 等)
    """
    path_map = {
        "video": RAW_VIDEO_PATH,
        "danmaku": RAW_DANMAKU_PATH,
        "processed": PROCESSED_DATA_PATH,
        "sales": SALES_DATA_PATH,
        "result": RESULTS_PATH,
    }

    if result_type not in path_map:
        raise ValueError(f"不支持的结果类型: {result_type}")

    base_path = path_map[result_type]

    # 确定最终保存路径
    if result_type == "result":
        if not filename:
            raise ValueError("result_type 为 'result' 时必须提供 filename 参数")
        # 确保结果目录存在
        os.makedirs(base_path, exist_ok=True)
        full_path = os.path.join(base_path, filename)
    else:
        full_path = base_path
        # 确保文件的父目录存在
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

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