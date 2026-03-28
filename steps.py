# steps.py
# 各个分析步骤的实现

import os
import asyncio
import pandas as pd
import config
from data_collection.bilibili_crawler import BilibiliCrawler
from data_preprocessing.clean import VideoCleaner, DanmakuCleaner
from data_preprocessing.segment import TextSegmenter
from sentiment_analysis.lexicon_model import LexiconSentimentAnalyzer
from sentiment_analysis.bert_model import BERTSentimentAnalyzer
from topic_modeling.lda_model import LDATopicModeler
from correlation.sales_correlation import SalesCorrelationAnalyzer
from visualization import plots


# 步骤1：采集视频
def step_collect_videos(crawler):
    config.print_step("步骤1: 采集视频数据")
    videos = asyncio.run(crawler.crawl_videos())
    if not videos:
        print("未采集到视频数据，程序退出")
        exit(1)
    config.print_step("步骤1: 采集视频数据", is_start=False)
    return videos


# 步骤2：处理视频（清洗+分词）
def step_process_videos():
    config.print_step("步骤2: 处理视频数据")
    keyword_video_ids = {}
    all_video_ids = []
    for kw in config.KEYWORDS:
        kw_video_dir = os.path.join(config.RAW_DATA_DIR, kw, "videos")
        if not os.path.exists(kw_video_dir):
            print(f"未找到关键词 '{kw}' 的视频目录，跳过")
            continue
        video_files = []
        for root, dirs, files in os.walk(kw_video_dir):
            for f in files:
                if f.endswith('.csv'):
                    video_files.append(os.path.join(root, f))
        if not video_files:
            print(f"未找到关键词 '{kw}' 的视频文件，跳过")
            continue
        latest_video = max(video_files, key=os.path.getmtime)
        print(f"处理关键词 '{kw}'，使用视频文件: {latest_video}")

        video_cleaner = VideoCleaner(raw_path=latest_video)
        video_cleaner.run()
        cleaned_video_path = video_cleaner.output_path

        video_segmenter = TextSegmenter(
            cleaned_video_path,
            config.SEGMENTED_VIDEOS_PATH,
            text_column='cleaned_title'
        )
        video_segmenter.load_data()
        video_segmenter.segment()
        video_segmenter.save()

        video_df = pd.read_csv(config.SEGMENTED_VIDEOS_PATH)
        ids = video_df['bv_id'].tolist()
        keyword_video_ids[kw] = ids
        all_video_ids.extend(ids)

    print(f"共获取 {len(all_video_ids)} 个视频ID")
    config.print_step("步骤2: 处理视频数据", is_start=False)
    return keyword_video_ids, all_video_ids


# 步骤3：采集弹幕
def step_collect_danmaku(crawler, keyword_video_ids):
    config.print_step("步骤3: 采集弹幕数据")
    for kw, ids in keyword_video_ids.items():
        print(f"\n正在采集关键词 '{kw}' 的弹幕，共 {len(ids)} 个视频...")
        asyncio.run(crawler.crawl_danmaku(ids, keyword=kw))
        print(f"关键词 '{kw}' 采集完成")
    config.print_step("步骤3: 采集弹幕数据", is_start=False)


# 步骤4：处理弹幕（清洗+合并+分词）
def step_process_danmaku():
    config.print_step("步骤4: 处理弹幕数据")
    all_cleaned_paths = []
    for kw in config.KEYWORDS:
        kw_danmaku_dir = os.path.join(config.RAW_DATA_DIR, kw, "danmaku")
        if not os.path.exists(kw_danmaku_dir):
            continue
        danmaku_files = []
        for root, dirs, files in os.walk(kw_danmaku_dir):
            for f in files:
                if f.endswith('.csv'):
                    danmaku_files.append(os.path.join(root, f))
        if not danmaku_files:
            continue
        latest_danmaku = max(danmaku_files, key=os.path.getmtime)
        print(f"处理关键词 '{kw}'，使用弹幕文件: {latest_danmaku}")

        comment_cleaner = DanmakuCleaner(raw_path=latest_danmaku)
        comment_cleaner.run()
        all_cleaned_paths.append(comment_cleaner.output_path)

    # 合并清洗后的弹幕
    combined_df = pd.concat([pd.read_csv(p, encoding='utf-8-sig') for p in all_cleaned_paths], ignore_index=True)
    combined_path = os.path.join(config.BASE_DIR, "data", "processed", "combined_cleaned_comments.csv")
    combined_df.to_csv(combined_path, index=False, encoding='utf-8-sig')
    print(f"合并后的清洗弹幕保存至 {combined_path}，共 {len(combined_df)} 条")

    # 分词
    comment_segmenter = TextSegmenter(combined_path, config.SEGMENTED_COMMENTS_PATH)
    comment_segmenter.load_data()
    comment_segmenter.segment()
    comment_segmenter.save()
    print("弹幕分词完成")
    config.print_step("步骤4: 处理弹幕数据", is_start=False)


# 步骤5：情感分析
def step_sentiment_analysis():
    config.print_step("步骤5: 情感分析")
    if config.USE_BERT:
        analyzer = BERTSentimentAnalyzer(config.SEGMENTED_COMMENTS_PATH)
        output_file = "sentiment_bert.csv"
    else:
        analyzer = LexiconSentimentAnalyzer(config.SEGMENTED_COMMENTS_PATH)
        output_file = "sentiment_lexicon.csv"

    sentiment_df = analyzer.analyze()
    # 使用 SaveData 保存结果
    config.SaveData(sentiment_df, result_type="result", filename=output_file).save()
    config.print_table(sentiment_df['sentiment_label'].value_counts().reset_index(name='数量'), title="情感分布")
    config.print_step("步骤5: 情感分析", is_start=False)
    return sentiment_df


# 步骤6：主题建模
def step_topic_modeling():
    config.print_step("步骤6: 主题建模")
    modeler = LDATopicModeler(config.SEGMENTED_COMMENTS_PATH, n_topics=config.N_TOPICS)
    topic_df, topics = modeler.run()
    # 保存主题模型结果
    config.SaveData(topic_df, result_type="result", filename="topic_modeling.csv").save()
    # 保存主题关键词表格（可选）
    topic_table = pd.DataFrame([{'主题ID': t['topic_id'], '关键词': t['keywords_str']} for t in topics])
    config.SaveData(topic_table, result_type="result", filename="topic_keywords.csv").save()
    config.print_table(topic_table, title="各主题关键词")
    config.print_step("步骤6: 主题建模", is_start=False)
    return topic_df, topics


# 步骤7：情感-销量关联分析
def step_correlation_analysis(sentiment_path):
    config.print_step("步骤7: 情感-销量关联分析")
    analyzer = SalesCorrelationAnalyzer(sentiment_path, config.SALES_DATA_PATH)
    corr_df, corr_results, lag_results = analyzer.run()
    # 保存关联分析结果
    config.SaveData(corr_df, result_type="result", filename="sales_correlation.csv").save()

    # 打印结果表格
    corr_table = pd.DataFrame(corr_results).T
    corr_table.columns = ['Pearson_r', 'Pearson_p', 'Spearman_r', 'Spearman_p']
    config.print_table(corr_table.round(4), title="相关性分析结果")

    lag_list = []
    for col, lags in lag_results.items():
        for lag in lags:
            lag_list.append({
                '指标': col,
                '滞后月数': lag['lag'],
                '相关系数': lag['correlation'],
                'p值': lag['p_value']
            })
    lag_table = pd.DataFrame(lag_list)
    config.print_table(lag_table.round(4), title="滞后效应分析结果")
    config.print_step("步骤7: 情感-销量关联分析", is_start=False)
    return corr_df


# 步骤8：可视化
def step_visualization(sentiment_df, corr_df):
    config.print_step("步骤8: 生成图表")
    plots.plot_sentiment_timeline(sentiment_df, "results/sentiment_timeline.png")
    plots.plot_sentiment_distribution(sentiment_df, "results/sentiment_distribution.png")
    plots.plot_sentiment_vs_sales(corr_df, "results/sentiment_vs_sales.png")
    plots.plot_sentiment_sales_scatter(corr_df, "results/sentiment_sales_scatter.png")
    plots.plot_wordcloud(sentiment_df['cleaned_text'], "results/wordcloud.png")
    config.print_step("步骤8: 生成图表", is_start=False)