# topic_modeling/lda_model.py
import pandas as pd
import numpy as np
import config
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class LDATopicModeler:
    def __init__(self, data_path, n_topics=None, use_tfidf=True, use_pos_filter=True):
        self.data_path = data_path
        self.df = None
        self.n_topics = n_topics if n_topics is not None else config.N_TOPICS
        self.auto_select = config.AUTO_SELECT_TOPICS and n_topics is None
        self.use_tfidf = use_tfidf
        self.use_pos_filter = use_pos_filter
        self.vectorizer = None
        self.lda = None
        self.doc_term_matrix = None
        self.feature_names = None
        self.stopwords = set()
        self.keep_pos = {'n', 'v', 'a', 'an', 'vn', 'ad', 'i', 'l', 'j'}
        self.valid_indices = None

        import os
        if os.path.exists(config.STOPWORDS_PATH):
            with open(config.STOPWORDS_PATH, 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f if line.strip())
            config.logger.info(f"LDA 加载停用词表，共 {len(self.stopwords)} 个词")

    def load_data(self):
        self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        if 'segmented' in self.df.columns:
            self.df['segmented'] = self.df['segmented'].fillna('').astype(str)
        config.logger.info(f"加载数据: {len(self.df)} 条")

    def prepare_corpus(self):
        texts = self.df['segmented'].tolist()
        filtered_texts = []
        self.valid_indices = []

        for idx, text in enumerate(texts):
            if self.use_pos_filter:
                words = []
                for word, flag in pseg.cut(text):
                    if word in self.stopwords or len(word) < 2:
                        continue
                    if flag in self.keep_pos:
                        words.append(word)
            else:
                words = [w for w in text.split() if w not in self.stopwords and len(w) > 1]
            if words:
                filtered_texts.append(' '.join(words))
                self.valid_indices.append(idx)

        if not filtered_texts:
            config.logger.warning("过滤后无有效文本，使用原始分词结果")
            filtered_texts = texts
            self.valid_indices = list(range(len(texts)))

        if self.use_tfidf:
            self.vectorizer = TfidfVectorizer(max_df=config.LDA_MAX_DF, min_df=config.LDA_MIN_DF, token_pattern=r'(?u)\b\w+\b')
        else:
            self.vectorizer = CountVectorizer(max_df=config.LDA_MAX_DF, min_df=config.LDA_MIN_DF, token_pattern=r'(?u)\b\w+\b')
        self.doc_term_matrix = self.vectorizer.fit_transform(filtered_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        config.logger.info(f"文档-词频矩阵形状: {self.doc_term_matrix.shape}")

    def select_best_topics(self, max_topics=15, min_topics=2):
        perplexities = []
        topic_range = range(min_topics, max_topics + 1, 1)
        for n in topic_range:
            lda = LatentDirichletAllocation(n_components=n, random_state=42, learning_method='online', max_iter=50)
            lda.fit(self.doc_term_matrix)
            perp = lda.perplexity(self.doc_term_matrix)
            perplexities.append(perp)
            config.logger.info(f"主题数 {n} 困惑度: {perp:.2f}")
        best_idx = np.argmin(perplexities)
        best_n = topic_range[best_idx]
        config.logger.info(f"最佳主题数: {best_n}，困惑度: {perplexities[best_idx]:.2f}")
        return best_n

    def fit_model(self):
        if self.auto_select:
            best_n = self.select_best_topics()
            self.n_topics = best_n
        config.logger.info(f"训练LDA模型，主题数: {self.n_topics}")
        self.lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            learning_method='online',
            learning_decay=0.7,
            learning_offset=10.0,
            max_iter=100
        )
        self.lda.fit(self.doc_term_matrix)
        config.logger.info("模型训练完成")

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
        valid_topics = topic_distribution.argmax(axis=1)
        valid_confidences = topic_distribution.max(axis=1)

        self.df['dominant_topic'] = np.nan
        self.df['topic_confidence'] = np.nan
        self.df.loc[self.valid_indices, 'dominant_topic'] = valid_topics
        self.df.loc[self.valid_indices, 'topic_confidence'] = valid_confidences
        return self.df

    def get_topic_trend(self):
        if 'date' in self.df.columns:
            self.df['date_month'] = pd.to_datetime(self.df['date']).dt.to_period('M')
            topic_trend = self.df.groupby(['date_month', 'dominant_topic']).size().unstack(fill_value=0)
            topic_trend_pct = topic_trend.div(topic_trend.sum(axis=1), axis=0) * 100
            return topic_trend_pct
        return None

    def run(self):
        self.load_data()
        self.prepare_corpus()
        self.fit_model()
        topics = self.get_topics()
        config.logger.info("\n各主题关键词:")
        for topic in topics:
            config.logger.info(f"主题 {topic['topic_id']}: {topic['keywords_str']}")
        self.assign_topics()
        return self.df, topics