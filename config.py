# config.py
# 配置文件，集中管理所有参数

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

# 文件路径配置
RAW_DATA_PATH = "data/raw/bullet_comments.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_comments.csv"
SALES_DATA_PATH = "data/sales/xiaomi_su7_sales.csv"
RESULTS_PATH = "results/"