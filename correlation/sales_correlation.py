# 情感-销量关联分析
# correlation/sales_correlation.py
# 情感与销量的关联分析

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import config


class SalesCorrelationAnalyzer:
    """情感-销量关联分析类"""

    def __init__(self, sentiment_path, sales_path):
        self.sentiment_path = sentiment_path
        self.sales_path = sales_path
        self.sentiment_df = None
        self.sales_df = None
        self.merged_df = None

    def load_data(self):
        """加载情感数据和销量数据"""
        # 加载情感分析结果
        self.sentiment_df = pd.read_csv(self.sentiment_path, encoding='utf-8-sig')
        print(f"情感数据列: {self.sentiment_df.columns.tolist()}")  # 调试

        # 加载销量数据，根据实际列名调整
        self.sales_df = pd.read_csv(self.sales_path, encoding='utf-8-sig')
        # 将 '时间' 列转换为 datetime，并重命名为 'month'
        self.sales_df['month'] = pd.to_datetime(self.sales_df['时间'], format='%Y-%m')
        # 重命名销量列
        self.sales_df.rename(columns={'月销量(辆)': 'sales'}, inplace=True)
        # 只保留需要的列，避免后续冲突
        self.sales_df = self.sales_df[['month', 'sales']]

        print(f"情感数据: {len(self.sentiment_df)} 条")
        print(f"销量数据: {len(self.sales_df)} 个月")

    def aggregate_by_month(self):
        """按月聚合情感指标"""
        # 将弹幕按月份聚合
        self.sentiment_df['date'] = pd.to_datetime(self.sentiment_df['date'])
        self.sentiment_df['month'] = self.sentiment_df['date'].dt.to_period('M')

        # 计算月度指标
        monthly_metrics = self.sentiment_df.groupby('month').agg({
            'sentiment_score': ['mean', 'std'],
            'sentiment_label': lambda x: (x == 'positive').sum() / len(x),  # 积极占比
            'bv_id': 'count'  # 弹幕数量
        }).reset_index()

        # 重命名列
        monthly_metrics.columns = ['month', 'avg_sentiment', 'sentiment_std',
                                   'positive_ratio', 'danmaku_count']

        # 添加消极占比
        negative_ratio = self.sentiment_df.groupby('month').apply(
            lambda x: (x['sentiment_label'] == 'negative').sum() / len(x)
        ).reset_index()
        negative_ratio.columns = ['month', 'negative_ratio']

        monthly_metrics = monthly_metrics.merge(negative_ratio, on='month')

        # 转换period为timestamp便于合并
        monthly_metrics['month'] = monthly_metrics['month'].dt.to_timestamp()

        return monthly_metrics

    def merge_with_sales(self, monthly_metrics):
        """合并销量数据"""
        self.merged_df = monthly_metrics.merge(
            self.sales_df,
            left_on='month',
            right_on='month'
        )
        print(f"合并后数据: {len(self.merged_df)} 个月")
        return self.merged_df

    def correlation_analysis(self):
        """相关性分析"""
        corr_results = {}

        # 当月情感与当月销量
        for col in ['avg_sentiment', 'positive_ratio', 'negative_ratio', 'danmaku_count']:
            pearson_corr, pearson_p = pearsonr(self.merged_df[col], self.merged_df['sales'])
            spearman_corr, spearman_p = spearmanr(self.merged_df[col], self.merged_df['sales'])

            corr_results[col] = {
                'pearson_r': pearson_corr,
                'pearson_p': pearson_p,
                'spearman_r': spearman_corr,
                'spearman_p': spearman_p
            }

        # 打印结果
        print("\n相关性分析结果:")
        for col, stats in corr_results.items():
            print(f"\n{col}:")
            print(f"  Pearson r = {stats['pearson_r']:.4f}, p = {stats['pearson_p']:.4f}")
            print(f"  Spearman r = {stats['spearman_r']:.4f}, p = {stats['spearman_p']:.4f}")

        return corr_results

    def lag_analysis(self, max_lag=3):
        """滞后效应分析"""
        lag_results = {}

        # 对每个情感指标计算不同滞后的相关性
        for col in ['avg_sentiment', 'positive_ratio', 'negative_ratio']:
            lag_corrs = []
            for lag in range(1, max_lag + 1):
                # 滞后情感与当月销量
                sentiment_lagged = self.merged_df[col].shift(lag)
                valid_idx = ~(sentiment_lagged.isna() | self.merged_df['sales'].isna())
                if valid_idx.sum() > 0:
                    corr, p = pearsonr(sentiment_lagged[valid_idx], self.merged_df['sales'][valid_idx])
                    lag_corrs.append({'lag': lag, 'correlation': corr, 'p_value': p})

            lag_results[col] = lag_corrs

        # 打印结果
        print("\n滞后效应分析结果:")
        for col, lags in lag_results.items():
            print(f"\n{col}:")
            for lag_info in lags:
                print(f"  滞后{lag_info['lag']}个月: r = {lag_info['correlation']:.4f}, p = {lag_info['p_value']:.4f}")

        return lag_results

    def granger_test(self, max_lag=3):
        """格兰杰因果检验"""
        # 准备数据
        data = pd.DataFrame({
            'sales': self.merged_df['sales'],
            'avg_sentiment': self.merged_df['avg_sentiment']
        })

        print("\n格兰杰因果检验 (情感 -> 销量):")
        try:
            # 需要足够的样本量
            if len(data) > max_lag * 2:
                result = grangercausalitytests(data[['sales', 'avg_sentiment']],
                                               maxlag=max_lag, verbose=False)
                for lag in range(1, max_lag + 1):
                    p_value = result[lag][0]['ssr_ftest'][1]
                    print(f"  滞后{lag}个月: F检验 p值 = {p_value:.4f}")
            else:
                print("  样本量不足，无法进行格兰杰因果检验")
        except Exception as e:
            print(f"  检验失败: {e}")

    def run(self):
        """执行完整关联分析"""
        self.load_data()
        monthly_metrics = self.aggregate_by_month()
        self.merge_with_sales(monthly_metrics)

        corr_results = self.correlation_analysis()
        lag_results = self.lag_analysis()
        self.granger_test()

        return self.merged_df, corr_results, lag_results

    def save(self, output_path):
        """保存结果"""
        self.merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"关联分析结果保存到 {output_path}")


# 使用示例
if __name__ == "__main__":
    analyzer = SalesCorrelationAnalyzer(
        sentiment_path="results/sentiment_lexicon.csv",
        sales_path=config.SALES_DATA_PATH
    )
    result_df, corr, lag = analyzer.run()
    analyzer.save("results/sales_correlation.csv")