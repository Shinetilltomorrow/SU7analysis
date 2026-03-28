# topic_modeling/lda_model.py
# LDA主题模型（增强版，支持自动选择主题数）

import pandas as pd
import numpy as np
import config
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class LDATopicModeler:
    """LDA主题建模类，支持自动选择主题数"""

    def __init__(self, data_path, n_topics=None):
        self.data_path = data_path
        self.df = None
        self.n_topics = n_topics if n_topics is not None else config.N_TOPICS
        self.auto_select = config.AUTO_SELECT_TOPICS and n_topics is None
        self.vectorizer = CountVectorizer(
            max_df=0.95, min_df=2, stop_words=None,  # 停用词已由分词阶段处理
            token_pattern=r'(?u)\b\w+\b'
        )
        self.lda = None
        self.doc_term_matrix = None
        self.feature_names = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        if 'segmented' in self.df.columns:
            self.df['segmented'] = self.df['segmented'].fillna('').astype(str)
        print(f"加载数据: {len(self.df)} 条")

    def prepare_corpus(self):
        texts = self.df['segmented'].tolist()
        self.doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        print(f"文档-词频矩阵形状: {self.doc_term_matrix.shape}")

    def select_best_topics(self, max_topics=15, min_topics=2, eval_every=1):
        """通过计算困惑度选择最佳主题数"""
        from sklearn.decomposition import LatentDirichletAllocation
        perplexities = []
        topic_range = range(min_topics, max_topics + 1, eval_every)
        for n in topic_range:
            lda = LatentDirichletAllocation(
                n_components=n,
                random_state=42,
                learning_method='online',
                max_iter=50
            )
            lda.fit(self.doc_term_matrix)
            perp = lda.perplexity(self.doc_term_matrix)
            perplexities.append(perp)
            print(f"主题数 {n} 困惑度: {perp:.2f}")

        # 选择困惑度最小的主题数
        best_idx = np.argmin(perplexities)
        best_n = topic_range[best_idx]
        print(f"最佳主题数: {best_n}，困惑度: {perplexities[best_idx]:.2f}")
        return best_n

    def fit_model(self):
        if self.auto_select:
            best_n = self.select_best_topics()
            self.n_topics = best_n
        print(f"训练LDA模型，主题数: {self.n_topics}")
        self.lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            learning_method='online'
        )
        self.lda.fit(self.doc_term_matrix)
        print("模型训练完成")

    def get_topics(self, n_words=config.N_TOP_WORDS):
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
        topic_distribution = self.lda.transform(self.doc_term_matrix)
        self.df['dominant_topic'] = topic_distribution.argmax(axis=1)
        self.df['topic_confidence'] = topic_distribution.max(axis=1)
        return self.df

    def get_topic_trend(self):
        if 'date' in self.df.columns:
            self.df['date_month'] = pd.to_datetime(self.df['date']).dt.to_period('M')
            topic_trend = self.df.groupby(['date_month', 'dominant_topic']).size().unstack(fill_value=0)
            topic_trend_pct = topic_trend.div(topic_trend.sum(axis=1), axis=0) * 100
            return topic_trend_pct
        return None

    def visualize_topics(self, output_html=None):
        """使用 pyLDAvis 可视化主题（需要安装 pyLDAvis）"""
        try:
            import pyLDAvis
            import pyLDAvis.sklearn
            vis_data = pyLDAvis.sklearn.prepare(self.lda, self.doc_term_matrix, self.vectorizer)
            if output_html:
                pyLDAvis.save_html(vis_data, output_html)
                print(f"主题可视化保存到 {output_html}")
            else:
                pyLDAvis.show(vis_data)
        except ImportError:
            print("未安装 pyLDAvis，跳过可视化")

    def run(self):
        self.load_data()
        self.prepare_corpus()
        self.fit_model()
        topics = self.get_topics()
        print("\n各主题关键词:")
        for topic in topics:
            print(f"主题 {topic['topic_id']}: {topic['keywords_str']}")
        self.assign_topics()
        return self.df, topics

    def save(self, output_path):
        self.df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"主题建模结果保存到 {output_path}")


if __name__ == "__main__":
    modeler = LDATopicModeler(config.SEGMENTED_COMMENTS_PATH)
    result_df, topics = modeler.run()
    modeler.save("results/topic_modeling.csv")

    trend = modeler.get_topic_trend()
    if trend is not None:
        trend.to_csv("results/topic_trend.csv")