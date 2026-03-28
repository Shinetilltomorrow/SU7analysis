# visualization/plots.py
# 图表绘制

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import config

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_sentiment_timeline(df, save_path=None):
    if 'date' not in df.columns:
        print("警告：数据中缺少 date 列，无法绘制情感时序图")
        return
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    fig, ax = plt.subplots(figsize=(12, 6))
    daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['date', 'sentiment_score']

    ax.plot(daily_sentiment['date'], daily_sentiment['sentiment_score'],
            color='steelblue', linewidth=1.5, label='平均情感得分')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='中性线')
    ax.fill_between(daily_sentiment['date'], 0.5, daily_sentiment['sentiment_score'],
                    where=(daily_sentiment['sentiment_score'] >= 0.5),
                    color='green', alpha=0.3, interpolate=True)
    ax.fill_between(daily_sentiment['date'], 0.5, daily_sentiment['sentiment_score'],
                    where=(daily_sentiment['sentiment_score'] <= 0.5),
                    color='red', alpha=0.3, interpolate=True)

    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('情感得分', fontsize=12)
    ax.set_title('小米SU7 B站弹幕情感得分时序图', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sentiment_distribution(df, save_path=None):
    if 'sentiment_label' not in df.columns:
        print("警告：数据中缺少 sentiment_label 列，无法绘制情感分布图")
        return
    sentiment_counts = df['sentiment_label'].value_counts()

    colors = ['green', 'gray', 'red']
    labels = ['积极', '中性', '消极']

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sentiment_counts.values,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0, 0)
    )
    ax.set_title('小米SU7 B站弹幕情感分布', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_topic_trend(topic_trend_df, save_path=None):
    if topic_trend_df is None or topic_trend_df.empty:
        print("警告：主题趋势数据为空，无法绘图")
        return
    if not pd.api.types.is_datetime64_any_dtype(topic_trend_df.index):
        try:
            topic_trend_df.index = pd.to_datetime(topic_trend_df.index)
        except Exception as e:
            print(f"无法将索引转换为日期: {e}")
            return

    fig, ax = plt.subplots(figsize=(12, 6))
    topic_trend_df.plot.area(ax=ax, alpha=0.7)
    ax.set_xlabel('月份', fontsize=12)
    ax.set_ylabel('主题占比 (%)', fontsize=12)
    ax.set_title('小米SU7 B站弹幕主题演化趋势', fontsize=14)
    ax.legend(title='主题', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sentiment_vs_sales(merged_df, save_path=None):
    if merged_df is None or merged_df.empty:
        print("警告：合并数据为空，无法绘制对比图")
        return
    if 'month' not in merged_df.columns:
        print("警告：数据中缺少 month 列，无法绘制对比图")
        return
    if not pd.api.types.is_datetime64_any_dtype(merged_df['month']):
        merged_df['month'] = pd.to_datetime(merged_df['month'])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    color = 'steelblue'
    ax1.set_xlabel('月份', fontsize=12)
    ax1.set_ylabel('销量 (辆)', color=color, fontsize=12)
    ax1.plot(merged_df['month'], merged_df['sales'], color=color,
             marker='o', linewidth=1.5, label='销量')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'coral'
    ax2.set_ylabel('平均情感得分', color=color, fontsize=12)
    ax2.plot(merged_df['month'], merged_df['avg_sentiment'],
             color=color, marker='s', linewidth=1.5, label='情感得分')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.set_title('小米SU7 B站弹幕情感与销量对比图', fontsize=14)
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_wordcloud(text_series, save_path=None):
    try:
        from wordcloud import WordCloud
        text_series = text_series.fillna('').astype(str)
        text = ' '.join(text_series)
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            font_path='simhei.ttf',
            max_words=200,
            collocations=False
        ).generate(text)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('小米SU7 B站弹幕词云图', fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    except ImportError:
        print("请安装wordcloud库: pip install wordcloud")


def plot_sentiment_sales_scatter(merged_df, save_path=None):
    """情感-销量散点图，并添加回归线"""
    if merged_df is None or merged_df.empty:
        print("警告：合并数据为空，无法绘制散点图")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(merged_df['avg_sentiment'], merged_df['sales'], alpha=0.7)
    # 添加线性回归线
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df['avg_sentiment'], merged_df['sales'])
    x_fit = np.linspace(merged_df['avg_sentiment'].min(), merged_df['avg_sentiment'].max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color='red', linestyle='--', label=f'R²={r_value**2:.3f}')
    ax.set_xlabel('平均情感得分')
    ax.set_ylabel('月销量 (辆)')
    ax.set_title('情感得分 vs 销量散点图')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()