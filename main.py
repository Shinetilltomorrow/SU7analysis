# main.py
# 主程序入口 - 实现先采集视频数据，再处理视频数据，再基于视频ID采集弹幕，再处理弹幕，最后进行综合分析

import os
import config
import pandas as pd
from data_collection.bilibili_crawler import BilibiliCrawler
import asyncio

# 创建必要的目录（新增视频数据目录）
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/sales', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('data/raw/videos', exist_ok=True)          # 存放原始视频数据
os.makedirs('data/processed/videos', exist_ok=True)    # 存放处理后的视频数据

print("=" * 50)
print("小米SU7 B站弹幕情感分析系统")
print("=" * 50)

# ============================================================================
# 步骤1: 采集视频数据（异步）
# ============================================================================

print("\n[步骤1] 采集视频数据...")
crawler = BilibiliCrawler(config.KEYWORDS, config.START_DATE, config.END_DATE)
video_raw_path = "data/raw/videos/videos.csv"
videos = asyncio.run(crawler.crawl_videos(save_path=video_raw_path))
# 如果 videos 为空，则退出
if not videos:
    print("未采集到视频数据，程序退出")
    exit(1)



# ============================================================================
# 步骤2: 处理视频数据（清洗、分词、提取视频ID）
# ============================================================================
print("\n[步骤2] 处理视频数据...")
from data_preprocessing.clean import DataCleaner
from data_preprocessing.segment import TextSegmenter

# 清洗视频数据（视频标题、简介等文本）
video_cleaner = DataCleaner(
    raw_path=config.RAW_VIDEOS_PATH,   # 使用配置中的原始视频路径
    output_path=config.PROCESSED_VIDEOS_PATH
)
video_cleaner.run()  # 清洗后的数据保存到 config.PROCESSED_VIDEOS_PATH

# 对视频文本进行分词（可仅对标题和简介分词）
video_segmenter = TextSegmenter(
    config.PROCESSED_VIDEOS_PATH,
    config.SEGMENTED_VIDEOS_PATH
)
video_segmenter.load_data()
video_segmenter.segment()
video_segmenter.save()

# 从处理后的视频数据中提取视频ID列表（用于后续弹幕采集）
video_df = pd.read_csv(config.SEGMENTED_VIDEOS_PATH)
# 假设视频数据中有 'video_id' 列，如果没有请根据实际字段名修改
video_ids = video_df['video_id'].tolist()
print(f"共获取 {len(video_ids)} 个视频ID")


# ============================================================================
# 步骤3: 基于视频ID采集弹幕数据（异步）
# ============================================================================
print("\n[步骤3] 采集弹幕数据...")
video_ids = video_df['bv_id'].tolist()  # 假设使用 'bv_id' 列
danmaku_data = asyncio.run(crawler.crawl_danmaku(video_ids, save_path=config.RAW_DANMAKU_PATH))
if not danmaku_data:
    print("未采集到弹幕数据，程序退出")
    exit(1)

# ============================================================================
# 步骤4: 处理弹幕数据（清洗、分词）
# ============================================================================
print("\n[步骤4] 处理弹幕数据...")
# 清洗弹幕
comment_cleaner = DataCleaner(
    raw_path=config.RAW_DANMAKU_PATH,
    output_path=config.CLEANED_COMMENTS_PATH
)
comment_cleaner.run()

# 弹幕分词
comment_segmenter = TextSegmenter(
    config.CLEANED_COMMENTS_PATH,
    config.SEGMENTED_COMMENTS_PATH
)
comment_segmenter.load_data()
comment_segmenter.segment()
comment_segmenter.save()
print("弹幕数据处理完成")

# ============================================================================
# 步骤5: 情感分析（基于处理后的弹幕）
# ============================================================================
print("\n[步骤5] 情感分析...")
from sentiment_analysis.lexicon_model import LexiconSentimentAnalyzer

analyzer = LexiconSentimentAnalyzer(config.SEGMENTED_COMMENTS_PATH)  # 使用分词后的弹幕
sentiment_df = analyzer.analyze()
analyzer.save("results/sentiment_lexicon.csv")

# ============================================================================
# 步骤6: 主题建模
# ============================================================================
print("\n[步骤6] 主题建模...")
from topic_modeling.lda_model import LDATopicModeler

modeler = LDATopicModeler(config.SEGMENTED_COMMENTS_PATH, n_topics=config.N_TOPICS)
topic_df, topics = modeler.run()
modeler.save("results/topic_modeling.csv")

# ============================================================================
# 步骤7: 情感-销量关联分析
# ============================================================================
print("\n[步骤7] 情感-销量关联分析...")
from correlation.sales_correlation import SalesCorrelationAnalyzer

corr_analyzer = SalesCorrelationAnalyzer(
    sentiment_path="results/sentiment_lexicon.csv",
    sales_path=config.SALES_DATA_PATH
)
corr_df, corr_results, lag_results = corr_analyzer.run()
corr_analyzer.save("results/sales_correlation.csv")

# ============================================================================
# 步骤8: 可视化
# ============================================================================
print("\n[步骤8] 生成图表...")
from visualization import plots

# 绘制情感时序图
plots.plot_sentiment_timeline(sentiment_df, "results/sentiment_timeline.png")

# 绘制情感分布图
plots.plot_sentiment_distribution(sentiment_df, "results/sentiment_distribution.png")

# 绘制情感与销量对比图
plots.plot_sentiment_vs_sales(corr_df, "results/sentiment_vs_sales.png")

# 绘制词云图（基于弹幕文本）
plots.plot_wordcloud(sentiment_df['cleaned_text'], "results/wordcloud.png")

print("\n分析完成！所有结果已保存至 results/ 目录")