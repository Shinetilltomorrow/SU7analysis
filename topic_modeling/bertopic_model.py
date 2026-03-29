# topic_modeling/bertopic_model.py
# 使用 BERTopic 进行主题建模（基于语义聚类）

import pandas as pd
import config
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import numpy as np


class BERTopicModeler:
    def __init__(self, data_path, n_topics=None, min_topic_size=10):
        self.data_path = data_path
        self.df = None
        self.n_topics = n_topics
        self.min_topic_size = min_topic_size
        self.model = None
        self.topics = None
        self.probs = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        # 优先使用清洗后的文本，否则使用分词后的文本
        if 'cleaned_text' in self.df.columns:
            self.df['text'] = self.df['cleaned_text'].fillna('').astype(str)
        elif 'segmented' in self.df.columns:
            self.df['text'] = self.df['segmented'].fillna('').astype(str)
        else:
            raise ValueError("数据中缺少文本列（cleaned_text或segmented）")
        config.logger.info(f"加载数据: {len(self.df)} 条")

    def fit(self):
        # 使用多语言模型（支持中文）
        embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.model = BERTopic(
            embedding_model=embedding_model,
            nr_topics=self.n_topics,
            min_topic_size=self.min_topic_size,
            calculate_probabilities=True,
            verbose=True
        )
        texts = self.df['text'].tolist()
        self.topics, self.probs = self.model.fit_transform(texts)
        self.df['topic'] = self.topics
        # 保存主题概率（如有）
        if self.probs is not None:
            self.df['topic_prob'] = self.probs.max(axis=1)
        else:
            self.df['topic_prob'] = np.nan

        unique_topics = set(self.topics)
        n_topics = len(unique_topics) - (1 if -1 in unique_topics else 0)
        config.logger.info(f"发现主题数: {n_topics}（噪声主题不计）")

    def get_topics(self):
        """返回主题信息 DataFrame"""
        return self.model.get_topic_info()

    def save(self, output_path):
        self.df.to_csv(output_path, index=False, encoding='utf-8-sig')
        config.logger.info(f"主题建模结果保存到 {output_path}")

    def run(self):
        self.load_data()
        self.fit()
        return self.df, self.get_topics()