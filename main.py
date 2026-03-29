# main.py
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

    crawler = BilibiliCrawler(keywords=config.KEYWORDS, start_date=config.START_DATE, end_date=config.END_DATE)
    step_collect_videos(crawler)

    video_ids = step_process_videos()
    step_collect_danmaku(crawler, video_ids)
    step_process_danmaku()

    sentiment_df = step_sentiment_analysis()
    step_topic_modeling()

    sentiment_path = "results/sentiment_lexicon.csv"
    corr_df = step_correlation_analysis(sentiment_path)

    step_visualization(sentiment_df, corr_df)

    print("\n分析完成！所有结果已保存至 results/ 目录")


if __name__ == "__main__":
    main()