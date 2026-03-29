# correlation/sales_correlation.py
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
import config


class SalesCorrelationAnalyzer:
    def __init__(self, sentiment_path, sales_path):
        self.sentiment_path = sentiment_path
        self.sales_path = sales_path
        self.sentiment_df = None
        self.sales_df = None
        self.merged_df = None

    def load_data(self):
        self.sentiment_df = pd.read_csv(self.sentiment_path, encoding='utf-8-sig')
        self.sales_df = pd.read_csv(self.sales_path, encoding='utf-8-sig')
        self.sales_df['month'] = pd.to_datetime(self.sales_df['时间'], format='%Y-%m')
        self.sales_df.rename(columns={'月销量(辆)': 'sales'}, inplace=True)
        self.sales_df = self.sales_df[['month', 'sales']]
        config.logger.info(f"情感数据: {len(self.sentiment_df)} 条，销量数据: {len(self.sales_df)} 个月")

    def aggregate_by_month(self):
        self.sentiment_df['date'] = pd.to_datetime(self.sentiment_df['date'])
        self.sentiment_df['month'] = self.sentiment_df['date'].dt.to_period('M')
        monthly_metrics = self.sentiment_df.groupby('month').agg({
            'sentiment_score': ['mean', 'std'],
            'sentiment_label': lambda x: (x == 'positive').sum() / len(x),
            'bv_id': 'count'
        }).reset_index()
        monthly_metrics.columns = ['month', 'avg_sentiment', 'sentiment_std', 'positive_ratio', 'danmaku_count']
        negative_ratio = self.sentiment_df.groupby('month').apply(lambda x: (x['sentiment_label'] == 'negative').sum() / len(x)).reset_index()
        negative_ratio.columns = ['month', 'negative_ratio']
        monthly_metrics = monthly_metrics.merge(negative_ratio, on='month')
        monthly_metrics['month'] = monthly_metrics['month'].dt.to_timestamp()
        return monthly_metrics

    def merge_with_sales(self, monthly_metrics):
        self.merged_df = monthly_metrics.merge(self.sales_df, left_on='month', right_on='month')
        config.logger.info(f"合并后数据: {len(self.merged_df)} 个月")
        return self.merged_df

    def correlation_analysis(self):
        corr_results = {}
        for col in ['avg_sentiment', 'positive_ratio', 'negative_ratio', 'danmaku_count']:
            pearson_corr, pearson_p = pearsonr(self.merged_df[col], self.merged_df['sales'])
            spearman_corr, spearman_p = spearmanr(self.merged_df[col], self.merged_df['sales'])
            corr_results[col] = {'pearson_r': pearson_corr, 'pearson_p': pearson_p, 'spearman_r': spearman_corr, 'spearman_p': spearman_p}
        config.logger.info("\n相关性分析结果:")
        for col, stats in corr_results.items():
            config.logger.info(f"\n{col}:\n  Pearson r = {stats['pearson_r']:.4f}, p = {stats['pearson_p']:.4f}\n  Spearman r = {stats['spearman_r']:.4f}, p = {stats['spearman_p']:.4f}")
        return corr_results

    def lag_analysis(self, max_lag=3):
        lag_results = {}
        for col in ['avg_sentiment', 'positive_ratio', 'negative_ratio']:
            lag_corrs = []
            for lag in range(1, max_lag + 1):
                sentiment_lagged = self.merged_df[col].shift(lag)
                valid_idx = ~(sentiment_lagged.isna() | self.merged_df['sales'].isna())
                if valid_idx.sum() > 0:
                    corr, p = pearsonr(sentiment_lagged[valid_idx], self.merged_df['sales'][valid_idx])
                    lag_corrs.append({'lag': lag, 'correlation': corr, 'p_value': p})
            lag_results[col] = lag_corrs
        config.logger.info("\n滞后效应分析结果:")
        for col, lags in lag_results.items():
            config.logger.info(f"\n{col}:")
            for lag_info in lags:
                config.logger.info(f"  滞后{lag_info['lag']}个月: r = {lag_info['correlation']:.4f}, p = {lag_info['p_value']:.4f}")
        return lag_results

    def granger_test(self, max_lag=3):
        data = pd.DataFrame({'sales': self.merged_df['sales'], 'avg_sentiment': self.merged_df['avg_sentiment']})
        config.logger.info("\n格兰杰因果检验 (情感 -> 销量):")
        try:
            if len(data) > max_lag * 2:
                result = grangercausalitytests(data[['sales', 'avg_sentiment']], maxlag=max_lag, verbose=False)
                for lag in range(1, max_lag + 1):
                    p_value = result[lag][0]['ssr_ftest'][1]
                    config.logger.info(f"  滞后{lag}个月: F检验 p值 = {p_value:.4f}")
            else:
                config.logger.info("  样本量不足，无法进行格兰杰因果检验")
        except Exception as e:
            config.logger.error(f"  检验失败: {e}")

    def run(self):
        self.load_data()
        monthly_metrics = self.aggregate_by_month()
        self.merge_with_sales(monthly_metrics)
        corr_results = self.correlation_analysis()
        lag_results = self.lag_analysis()
        self.granger_test()
        return self.merged_df, corr_results, lag_results