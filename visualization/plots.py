# visualization/plots.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import config

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_sentiment_timeline(df, save_path=None):
    if 'date' not in df.columns:
        config.logger.warning("缺少 date 列")
        return
    df['date'] = pd.to_datetime(df['date'])
    daily = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
    daily.columns = ['date', 'sentiment_score']
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(daily['date'], daily['sentiment_score'], color='steelblue', linewidth=1.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax.fill_between(daily['date'], 0.5, daily['sentiment_score'], where=(daily['sentiment_score']>=0.5), color='green', alpha=0.3)
    ax.fill_between(daily['date'], 0.5, daily['sentiment_score'], where=(daily['sentiment_score']<=0.5), color='red', alpha=0.3)
    ax.set_xlabel('日期')
    ax.set_ylabel('情感得分')
    ax.set_title('小米SU7 B站弹幕情感得分时序图')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sentiment_distribution(df, save_path=None):
    if 'sentiment_label' not in df.columns:
        config.logger.warning("数据中缺少 sentiment_label 列，无法绘制情感分布图")
        return
    counts = df['sentiment_label'].value_counts()
    # 动态生成标签和颜色
    label_map = {'positive': '积极', 'neutral': '中性', 'negative': '消极'}
    labels = [label_map.get(k, k) for k in counts.index]
    colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    color_list = [colors.get(k, 'blue') for k in counts.index]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        counts.values,
        labels=labels,
        colors=color_list,
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.05 if i == 0 else 0 for i in range(len(counts))]   # 仅突出第一个
    )
    ax.set_title('小米SU7 B站弹幕情感分布', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_topic_trend(topic_trend_df, save_path=None):
    if topic_trend_df is None or topic_trend_df.empty:
        return
    if not pd.api.types.is_datetime64_any_dtype(topic_trend_df.index):
        topic_trend_df.index = pd.to_datetime(topic_trend_df.index)
    fig, ax = plt.subplots(figsize=(12,6))
    topic_trend_df.plot.area(ax=ax, alpha=0.7)
    ax.set_xlabel('月份')
    ax.set_ylabel('主题占比 (%)')
    ax.set_title('小米SU7 B站弹幕主题演化趋势')
    ax.legend(title='主题', bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sentiment_vs_sales(merged_df, save_path=None):
    if merged_df is None or merged_df.empty:
        return
    merged_df['month'] = pd.to_datetime(merged_df['month'])
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(merged_df['month'], merged_df['sales'], color='steelblue', marker='o', label='销量')
    ax1.set_xlabel('月份')
    ax1.set_ylabel('销量 (辆)', color='steelblue')
    ax2 = ax1.twinx()
    ax2.plot(merged_df['month'], merged_df['avg_sentiment'], color='coral', marker='s', label='情感得分')
    ax2.set_ylabel('平均情感得分', color='coral')
    ax2.axhline(y=0.5, color='gray', linestyle='--')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='upper left')
    ax1.set_title('小米SU7 B站弹幕情感与销量对比图')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sentiment_sales_scatter(merged_df, save_path=None):
    if merged_df is None or merged_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10,8))
    sizes = merged_df['danmaku_count'] / merged_df['danmaku_count'].max() * 500
    sc = ax.scatter(merged_df['avg_sentiment'], merged_df['sales'], s=sizes, alpha=0.7, c=merged_df['danmaku_count'], cmap='viridis', edgecolors='black')
    from scipy import stats
    slope, intercept, r_value, _, _ = stats.linregress(merged_df['avg_sentiment'], merged_df['sales'])
    x_fit = np.linspace(merged_df['avg_sentiment'].min(), merged_df['avg_sentiment'].max(), 100)
    ax.plot(x_fit, slope*x_fit+intercept, color='red', linestyle='--', label=f'R²={r_value**2:.3f}')
    ax.set_xlabel('平均情感得分')
    ax.set_ylabel('月销量 (辆)')
    ax.set_title('情感得分 vs 销量（气泡大小=弹幕数量）')
    plt.colorbar(sc, label='弹幕数量')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_wordcloud(text_series, save_path=None):
    try:
        from wordcloud import WordCloud
        import os
        stopwords = set()
        if os.path.exists(config.STOPWORDS_PATH):
            with open(config.STOPWORDS_PATH, 'r', encoding='utf-8') as f:
                stopwords = set(line.strip() for line in f if line.strip())
        all_words = []
        for txt in text_series:
            words = txt.split()
            words = [w for w in words if w not in stopwords and len(w) > 1]
            all_words.extend(words)
        text = ' '.join(all_words)
        wc = WordCloud(width=800, height=400, background_color='white', font_path='simhei.ttf', max_words=200).generate(text)
        fig, ax = plt.subplots(figsize=(10,6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('小米SU7 B站弹幕词云图')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    except ImportError:
        config.logger.warning("未安装wordcloud库，跳过词云图")