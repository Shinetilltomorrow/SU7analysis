# sentiment_analysis/paddle_model.py
# 使用 PaddleNLP 进行情感分析（无需手动下载模型，自动从国内镜像拉取）

import pandas as pd
import config
from paddlenlp import Taskflow


class PaddleSentimentAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        # 初始化情感分析任务（首次运行会自动下载模型）
        self.sentiment = Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch")

    def analyze(self):
        self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        # 优先使用清洗后的文本，否则使用分词后的文本
        if 'cleaned_text' in self.df.columns:
            text_col = 'cleaned_text'
        elif 'segmented' in self.df.columns:
            text_col = 'segmented'
        else:
            raise ValueError("数据中缺少文本列（cleaned_text或segmented）")
        texts = self.df[text_col].fillna('').astype(str).tolist()

        config.logger.info(f"正在进行 PaddleNLP 情感分析，共 {len(texts)} 条...")
        # 批量处理（Taskflow 自动分批）
        results = self.sentiment(texts)
        # 将结果转换为情感得分（积极概率）
        scores = [r['score'] if r['label'] == 'positive' else 1 - r['score'] for r in results]
        self.df['sentiment_score'] = scores
        self.df['sentiment_label'] = self.df['sentiment_score'].apply(
            lambda x: 'positive' if x >= config.POSITIVE_THRESHOLD
            else ('negative' if x <= config.NEGATIVE_THRESHOLD else 'neutral')
        )
        sentiment_counts = self.df['sentiment_label'].value_counts()
        config.logger.info(f"情感分布:\n{sentiment_counts}")
        return self.df