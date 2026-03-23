# LDA主题模型
# topic_modeling/lda_model.py
# LDA主题模型

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import config


class LDATopicModeler:
    """LDA主题建模类"""

    def __init__(self, data_path, n_topics=5):
        self.data_path = data_path
        self.df = None
        self.n_topics = n_topics
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            learning_method='online'
        )

    def load_data(self):
        """加载数据"""
        self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        print(f"加载数据: {len(self.df)} 条")

    def prepare_corpus(self):
        """准备语料库"""
        texts = self.df['segmented'].tolist()
        # 转换为文档-词频矩阵
        self.doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        print(f"文档-词频矩阵形状: {self.doc_term_matrix.shape}")

    def fit_model(self):
        """训练LDA模型"""
        print("正在训练LDA模型...")
        self.lda.fit(self.doc_term_matrix)
        print("模型训练完成")

    def get_topics(self, n_words=10):
        """提取每个主题的关键词"""
        topics = []
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words_idx = topic.argsort()[:-n_words - 1:-1]
            top_words = [self.feature_names[i] for i in top_words_idx]
            topics.append({
                'topic_id': topic_idx,
                'keywords': top_words,
                'keywords_str': ' '.join(top_words)
            })
        return topics

    def assign_topics(self):
        """为每条弹幕分配主题"""
        topic_distribution = self.lda.transform(self.doc_term_matrix)
        self.df['dominant_topic'] = topic_distribution.argmax(axis=1)
        self.df['topic_confidence'] = topic_distribution.max(axis=1)
        return self.df

    def get_topic_trend(self):
        """获取主题随时间的变化趋势"""
        # 需要按时间分组
        if 'date' in self.df.columns:
            self.df['date_month'] = pd.to_datetime(self.df['date']).dt.to_period('M')
            topic_trend = self.df.groupby(['date_month', 'dominant_topic']).size().unstack(fill_value=0)
            # 归一化为百分比
            topic_trend_pct = topic_trend.div(topic_trend.sum(axis=1), axis=0) * 100
            return topic_trend_pct
        return None

    def run(self):
        """执行完整流程"""
        self.load_data()
        self.prepare_corpus()
        self.fit_model()
        topics = self.get_topics(config.N_TOP_WORDS)

        print("\n各主题关键词:")
        for topic in topics:
            print(f"主题 {topic['topic_id']}: {topic['keywords_str']}")

        self.assign_topics()
        return self.df, topics

    def save(self, output_path):
        """保存结果"""
        self.df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"主题建模结果保存到 {output_path}")


# 使用示例
if __name__ == "__main__":
    modeler = LDATopicModeler(config.PROCESSED_DATA_PATH, n_topics=config.N_TOPICS)
    result_df, topics = modeler.run()
    modeler.save("results/topic_modeling.csv")

    # 获取主题趋势
    trend = modeler.get_topic_trend()
    if trend is not None:
        trend.to_csv("results/topic_trend.csv")