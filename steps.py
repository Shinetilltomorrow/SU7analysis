# steps.py
import os
import asyncio
import pandas as pd
import config
from data_collection.bilibili_crawler import BilibiliCrawler
from data_preprocessing.clean import VideoCleaner, DanmakuCleaner
from data_preprocessing.segment import TextSegmenter
from sentiment_analysis.lexicon_model import LexiconSentimentAnalyzer
from sentiment_analysis.bert_model import BERTSentimentAnalyzer   # 新增
from topic_modeling.lda_model import LDATopicModeler
from correlation.sales_correlation import SalesCorrelationAnalyzer
from visualization import plots


def step_collect_videos(crawler):
    config.print_step("步骤1: 采集视频数据")
    videos = asyncio.run(crawler.crawl_videos())
    if not videos:
        config.logger.error("未采集到视频数据，程序退出")
        exit(1)
    config.print_step("步骤1: 采集视频数据", is_start=False)
    return videos


def step_process_videos():
    config.print_step("步骤2: 处理视频数据")
    all_cleaned_paths = []
    for kw in config.KEYWORDS:
        kw_video_dir = os.path.join(config.RAW_DATA_DIR, kw, "videos")
        if not os.path.exists(kw_video_dir):
            continue
        video_files = []
        for root, _, files in os.walk(kw_video_dir):
            for f in files:
                if f.endswith('.csv'):
                    video_files.append(os.path.join(root, f))
        if not video_files:
            continue
        latest_video = max(video_files, key=os.path.getmtime)
        config.logger.info(f"处理关键词 '{kw}'，使用视频文件: {latest_video}")
        cleaner = VideoCleaner(raw_path=latest_video)
        cleaner.run()
        all_cleaned_paths.append(cleaner.output_path)

    if all_cleaned_paths:
        combined_df = pd.concat([pd.read_csv(p, encoding='utf-8-sig') for p in all_cleaned_paths], ignore_index=True)
        combined_path = os.path.join(config.BASE_DIR, "data", "processed", "combined_cleaned_videos.csv")
        combined_df.to_csv(combined_path, index=False, encoding='utf-8-sig')
        config.logger.info(f"合并视频保存至 {combined_path}，共 {len(combined_df)} 条")
        segmenter = TextSegmenter(combined_path, config.SEGMENTED_VIDEOS_PATH, text_column='cleaned_title', use_pos_filter=False)
        segmenter.load_data()
        segmenter.segment()
        segmenter.save()
    else:
        config.logger.warning("没有找到任何视频文件")

    if os.path.exists(config.SEGMENTED_VIDEOS_PATH):
        video_df = pd.read_csv(config.SEGMENTED_VIDEOS_PATH)
        all_video_ids = video_df['bv_id'].tolist()
    else:
        all_video_ids = []
    config.logger.info(f"共获取 {len(all_video_ids)} 个视频ID")
    config.print_step("步骤2: 处理视频数据", is_start=False)
    return all_video_ids


def step_collect_danmaku(crawler, video_ids):
    config.print_step("步骤3: 采集弹幕数据")
    if not video_ids:
        config.logger.warning("没有视频ID，跳过弹幕采集")
        return
    asyncio.run(crawler.crawl_danmaku(video_ids))
    config.print_step("步骤3: 采集弹幕数据", is_start=False)


def step_process_danmaku():
    config.print_step("步骤4: 处理弹幕数据")
    all_cleaned_paths = []
    for kw in config.KEYWORDS:
        kw_danmaku_dir = os.path.join(config.RAW_DATA_DIR, kw, "danmaku")
        if not os.path.exists(kw_danmaku_dir):
            continue
        danmaku_files = []
        for root, _, files in os.walk(kw_danmaku_dir):
            for f in files:
                if f.endswith('.csv'):
                    danmaku_files.append(os.path.join(root, f))
        if not danmaku_files:
            continue
        latest_danmaku = max(danmaku_files, key=os.path.getmtime)
        config.logger.info(f"处理关键词 '{kw}'，使用弹幕文件: {latest_danmaku}")
        cleaner = DanmakuCleaner(raw_path=latest_danmaku)
        cleaner.run()
        all_cleaned_paths.append(cleaner.output_path)

    if not all_cleaned_paths:
        config.logger.error("未找到任何弹幕文件")
        return

    combined_df = pd.concat([pd.read_csv(p, encoding='utf-8-sig') for p in all_cleaned_paths], ignore_index=True)
    combined_path = os.path.join(config.BASE_DIR, "data", "processed", "combined_cleaned_comments.csv")
    combined_df.to_csv(combined_path, index=False, encoding='utf-8-sig')
    config.logger.info(f"合并弹幕保存至 {combined_path}，共 {len(combined_df)} 条")

    segmenter = TextSegmenter(combined_path, config.SEGMENTED_COMMENTS_PATH, text_column='cleaned_text', use_pos_filter=config.USE_POS_FILTER)
    segmenter.load_data()
    segmenter.segment()
    segmenter.save()
    config.logger.info("弹幕分词完成")
    config.print_step("步骤4: 处理弹幕数据", is_start=False)


def step_sentiment_analysis():
    config.print_step("步骤5: 情感分析")
    try:
        if config.USE_BERT:
            analyzer = BERTSentimentAnalyzer(config.SEGMENTED_COMMENTS_PATH, batch_size=32)
            output_file = "sentiment_bert.csv"
        else:
            analyzer = LexiconSentimentAnalyzer(config.SEGMENTED_COMMENTS_PATH)
            output_file = "sentiment_lexicon.csv"

        sentiment_df = analyzer.analyze()
        config.SaveData(sentiment_df, result_type="result", filename=output_file).save()
        config.print_table(sentiment_df['sentiment_label'].value_counts().reset_index(name='数量'), title="情感分布")
        config.logger.info("情感分析完成")
        config.print_step("步骤5: 情感分析", is_start=False)
        return sentiment_df
    except Exception as e:
        config.logger.error(f"情感分析失败: {e}", exc_info=True)
        raise

def step_topic_modeling():
    config.print_step("步骤6: 主题建模")
    try:
        modeler = LDATopicModeler(
            config.SEGMENTED_COMMENTS_PATH,
            n_topics=config.N_TOPICS,
            use_tfidf=True,
            use_pos_filter=config.USE_POS_FILTER
        )
        topic_df, topics = modeler.run()
        config.SaveData(topic_df, result_type="result", filename="topic_modeling.csv").save()
        topic_table = pd.DataFrame([{'主题ID': t['topic_id'], '关键词': t['keywords_str']} for t in topics])
        config.SaveData(topic_table, result_type="result", filename="topic_keywords.csv").save()
        config.print_table(topic_table, title="各主题关键词")
        config.logger.info("主题建模完成")
        config.print_step("步骤6: 主题建模", is_start=False)
        return topic_df, topics
    except Exception as e:
        config.logger.error(f"主题建模失败: {e}", exc_info=True)
        raise


def step_correlation_analysis(sentiment_path):
    config.print_step("步骤7: 情感-销量关联分析")
    try:
        if not os.path.exists(sentiment_path):
            config.logger.error(f"情感结果文件不存在: {sentiment_path}")
            return None
        if not os.path.exists(config.SALES_DATA_PATH):
            config.logger.error(f"销量数据文件不存在: {config.SALES_DATA_PATH}")
            return None
        analyzer = SalesCorrelationAnalyzer(sentiment_path, config.SALES_DATA_PATH)
        corr_df, corr_results, lag_results = analyzer.run()
        config.SaveData(corr_df, result_type="result", filename="sales_correlation.csv").save()
        corr_table = pd.DataFrame(corr_results).T
        corr_table.columns = ['Pearson_r', 'Pearson_p', 'Spearman_r', 'Spearman_p']
        config.print_table(corr_table.round(4), title="相关性分析结果")
        lag_list = []
        for col, lags in lag_results.items():
            for lag in lags:
                lag_list.append({'指标': col, '滞后月数': lag['lag'], '相关系数': lag['correlation'], 'p值': lag['p_value']})
        lag_table = pd.DataFrame(lag_list)
        config.print_table(lag_table.round(4), title="滞后效应分析结果")
        config.logger.info("关联分析完成")
        config.print_step("步骤7: 情感-销量关联分析", is_start=False)
        return corr_df
    except Exception as e:
        config.logger.error(f"关联分析失败: {e}", exc_info=True)
        raise


def step_visualization(sentiment_df, corr_df):
    config.print_step("步骤8: 生成图表")
    try:
        plots.plot_sentiment_timeline(sentiment_df, "results/sentiment_timeline.png")
        plots.plot_sentiment_distribution(sentiment_df, "results/sentiment_distribution.png")
        if corr_df is not None:
            plots.plot_sentiment_vs_sales(corr_df, "results/sentiment_vs_sales.png")
            plots.plot_sentiment_sales_scatter(corr_df, "results/sentiment_sales_scatter.png")
        plots.plot_wordcloud(sentiment_df['cleaned_text'], "results/wordcloud.png")
        config.logger.info("图表生成完成")
        config.print_step("步骤8: 生成图表", is_start=False)
    except Exception as e:
        config.logger.error(f"图表生成失败: {e}", exc_info=True)