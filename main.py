# main.py
# 小米SU7 B站弹幕情感分析系统主程序

import config
from steps import (
    step_collect_videos,
    step_process_videos,
    step_collect_danmaku,
    step_process_danmaku,
    step_sentiment_analysis,
    step_topic_modeling,
    step_correlation_analysis,
    step_visualization
)
from data_collection.bilibili_crawler import BilibiliCrawler


def main():
    print("="*50)
    print("小米SU7 B站弹幕情感分析系统")
    print("="*50)
    config.ensure_directories()

    # 1. 采集视频
    crawler = BilibiliCrawler(keywords=config.KEYWORDS, start_date=config.START_DATE, end_date=config.END_DATE)
    step_collect_videos(crawler)

    # 2. 处理视频（清洗+分词）
    keyword_video_ids, all_video_ids = step_process_videos()

    # 3. 采集弹幕
    step_collect_danmaku(crawler, keyword_video_ids)

    # 4. 处理弹幕（清洗+合并+分词）
    step_process_danmaku()

    # 5. 情感分析
    sentiment_df = step_sentiment_analysis()

    # 6. 主题建模
    step_topic_modeling()

    # 7. 关联分析
    sentiment_path = "results/sentiment_bert.csv" if config.USE_BERT else "results/sentiment_lexicon.csv"
    corr_df = step_correlation_analysis(sentiment_path)

    # 8. 可视化
    step_visualization(sentiment_df, corr_df)

    print("\n分析完成！所有结果已保存至 results/ 目录")


if __name__ == "__main__":
    main()