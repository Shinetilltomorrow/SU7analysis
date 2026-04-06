# topic_modeling/lda_model.py
import pandas as pd
import numpy as np
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
import logging
import os

# 尝试导入 config，如果没有则使用默认配置
try:
    import config
except ImportError:
    config = None

# 尝试导入 joblib 和 gensim（用于模型保存和 coherence）
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    warnings.warn("joblib 未安装，模型保存/加载功能不可用")

try:
    from gensim.corpora.dictionary import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    warnings.warn("gensim 未安装，主题一致性评估功能不可用")


class LDATopicModeler:
    def __init__(self, data_path, n_topics=None, use_tfidf=False, use_pos_filter=True,
                 stopwords_path=None, max_df=0.8, min_df=2, n_top_words=10,
                 auto_select_topics=False, select_topics_fast_iter=30):
        """
        参数:
            data_path: CSV 文件路径，需包含 'segmented' 列（分词结果，空格分隔）
            n_topics: 主题数，若为 None 且 auto_select_topics=True 则自动选择
            use_tfidf: 是否使用 TF-IDF 向量化（LDA 理论上应使用词频，默认 False）
            use_pos_filter: 是否使用词性过滤
            stopwords_path: 停用词文件路径
            max_df: 向量化时 max_df 参数
            min_df: 向量化时 min_df 参数
            n_top_words: 每个主题输出的关键词数量
            auto_select_topics: 是否自动选择主题数（覆盖 n_topics 参数）
            select_topics_fast_iter: 自动选择时的快速迭代次数（降低计算开销）
        """
        self.data_path = data_path
        self.df = None
        self.n_topics = n_topics if n_topics is not None else 10
        self.auto_select = auto_select_topics
        self.use_tfidf = use_tfidf
        self.use_pos_filter = use_pos_filter
        self.max_df = max_df
        self.min_df = min_df
        self.n_top_words = n_top_words
        self.select_topics_fast_iter = select_topics_fast_iter

        self.vectorizer = None
        self.lda = None
        self.doc_term_matrix = None
        self.feature_names = None
        self.stopwords = set()
        # 保留的词性：名词、动词、形容词、成语、简略语等
        self.keep_pos = {'n', 'v', 'a', 'an', 'vn', 'ad', 'i', 'l', 'j'}
        self.valid_indices = None

        # 设置日志
        self.logger = self._get_logger()

        # 加载停用词
        if stopwords_path is None:
            if config and hasattr(config, 'STOPWORDS_PATH'):
                stopwords_path = config.STOPWORDS_PATH
            else:
                stopwords_path = 'stopwords.txt'
        if os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f if line.strip())
            self.logger.info(f"加载停用词表，共 {len(self.stopwords)} 个词")
        else:
            self.logger.warning(f"停用词文件不存在: {stopwords_path}")

    def _get_logger(self):
        """获取 logger，优先使用 config 中的，否则创建基础 logger"""
        if config and hasattr(config, 'logger') and config.logger:
            return config.logger
        logger = logging.getLogger('LDATopicModeler')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def load_data(self):
        """加载 CSV 数据"""
        self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        if 'segmented' in self.df.columns:
            self.df['segmented'] = self.df['segmented'].fillna('').astype(str)
        self.logger.info(f"加载数据: {len(self.df)} 条")

    def _add_pos_annotations(self):
        """对 segmented 列进行词性标注，结果保存为 'segmented_with_pos' 列（格式：词/词性 词/词性 ...）"""
        self.logger.info("开始词性标注（这可能需要一些时间）...")
        def annotate(text):
            words_flags = []
            for word, flag in pseg.cut(text):
                words_flags.append(f"{word}/{flag}")
            return ' '.join(words_flags)
        self.df['segmented_with_pos'] = self.df['segmented'].astype(str).apply(annotate)
        self.logger.info("词性标注完成，结果已保存到 'segmented_with_pos' 列")

    def prepare_corpus(self):
        """预处理语料：过滤停用词、词性筛选，生成文档-词频矩阵"""
        # 如果使用词性过滤且没有预标注列，则进行标注
        if self.use_pos_filter and 'segmented_with_pos' not in self.df.columns:
            self._add_pos_annotations()

        # 选择源文本列
        if self.use_pos_filter:
            raw_texts = self.df['segmented_with_pos'].tolist()
        else:
            raw_texts = self.df['segmented'].tolist()

        filtered_texts = []
        self.valid_indices = []

        for idx, text in enumerate(raw_texts):
            if self.use_pos_filter:
                words = []
                # 解析 "词/词性" 格式
                for token in text.split():
                    if '/' not in token:
                        continue
                    word, flag = token.rsplit('/', 1)
                    if word in self.stopwords or len(word) < 2:
                        continue
                    if flag in self.keep_pos:
                        words.append(word)
            else:
                words = [w for w in text.split() if w not in self.stopwords and len(w) > 1]

            if words:
                filtered_texts.append(' '.join(words))
                self.valid_indices.append(idx)

        # 如果过滤后为空，回退到原始分词（不过滤）
        if not filtered_texts:
            self.logger.warning("过滤后无有效文本，使用原始分词结果（不过滤停用词和词性）")
            filtered_texts = raw_texts
            self.valid_indices = list(range(len(raw_texts)))

        # 向量化
        if self.use_tfidf:
            warnings.warn("使用 TF-IDF 向量化可能影响 LDA 主题质量，建议使用词频矩阵（use_tfidf=False）", UserWarning)
            self.vectorizer = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df,
                                              token_pattern=r'(?u)\b\w+\b')
        else:
            self.vectorizer = CountVectorizer(max_df=self.max_df, min_df=self.min_df,
                                              token_pattern=r'(?u)\b\w+\b')

        self.doc_term_matrix = self.vectorizer.fit_transform(filtered_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.logger.info(f"文档-词频矩阵形状: {self.doc_term_matrix.shape}")

    def select_best_topics(self, max_topics=15, min_topics=2, use_coherence=False, coherence_limit=0.4):
        """
        自动选择最佳主题数
        参数:
            max_topics: 最大主题数
            min_topics: 最小主题数
            use_coherence: 是否使用 coherence 指标（需要 gensim），否则使用困惑度
            coherence_limit: coherence 最低接受阈值，低于此值继续增加主题数
        返回:
            best_n: 最佳主题数
        """
        if use_coherence and not GENSIM_AVAILABLE:
            self.logger.warning("gensim 未安装，无法计算 coherence，回退到困惑度方法")
            use_coherence = False

        topic_range = range(min_topics, max_topics + 1)
        scores = []  # 存储困惑度或 coherence

        if use_coherence:
            # 准备 gensim 语料（需要原始分词列表）
            # 从 filtered_texts 重建分词列表（注意 self.doc_term_matrix 已经是向量化后，需要原始文本）
            # 这里需要重新获取过滤后的文本列表，可以暂时存储
            # 简单起见，我们重新生成一次过滤后的文本（但会重复 prepare_corpus 的工作）
            # 为了效率，在调用 select_best_topics 前最好保留过滤后的文本
            if not hasattr(self, '_filtered_texts_for_coherence'):
                self.logger.warning("未找到过滤后文本，重新生成...")
                # 重新生成一次（避免重复代码，直接调用 prepare_corpus 的过滤部分）
                self._prepare_filtered_texts_for_coherence()
            texts = [text.split() for text in self._filtered_texts_for_coherence]
            dictionary = Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]

            for n in topic_range:
                self.logger.info(f"测试主题数 {n}，使用 coherence 评估...")
                lda_temp = LatentDirichletAllocation(n_components=n, random_state=42,
                                                     learning_method='online',
                                                     max_iter=self.select_topics_fast_iter)
                lda_temp.fit(self.doc_term_matrix)
                # 获取主题词
                topics_words = []
                for topic_idx, topic in enumerate(lda_temp.components_):
                    top_words_idx = topic.argsort()[:-self.n_top_words - 1:-1]
                    top_words = [self.feature_names[i] for i in top_words_idx]
                    topics_words.append(top_words)
                cm = CoherenceModel(topics=topics_words, texts=texts,
                                    dictionary=dictionary, coherence='c_v')
                coherence = cm.get_coherence()
                scores.append(coherence)
                self.logger.info(f"主题数 {n} coherence: {coherence:.4f}")
                # 如果 coherence 低于阈值且已经大于某个值，可以提前终止（简单策略）
                if n > min_topics and coherence < coherence_limit and scores[-1] < scores[-2]:
                    self.logger.info(f"coherence 下降且低于 {coherence_limit}，提前终止")
                    break
            best_idx = np.argmax(scores) if scores else 0
            best_n = topic_range[best_idx]
            self.logger.info(f"最佳主题数: {best_n}，coherence: {scores[best_idx]:.4f}")
        else:
            # 使用困惑度，但用较少的迭代快速扫描
            perplexities = []
            for n in topic_range:
                self.logger.info(f"测试主题数 {n}，快速迭代 {self.select_topics_fast_iter} 次...")
                lda_temp = LatentDirichletAllocation(n_components=n, random_state=42,
                                                     learning_method='online',
                                                     max_iter=self.select_topics_fast_iter)
                lda_temp.fit(self.doc_term_matrix)
                perp = lda_temp.perplexity(self.doc_term_matrix)
                perplexities.append(perp)
                self.logger.info(f"主题数 {n} 困惑度: {perp:.2f}")
            best_idx = np.argmin(perplexities)
            best_n = topic_range[best_idx]
            self.logger.info(f"最佳主题数: {best_n}，困惑度: {perplexities[best_idx]:.2f}")

        return best_n

    def _prepare_filtered_texts_for_coherence(self):
        """仅用于 coherence 评估时获取过滤后的文本列表（分词列表）"""
        if self.use_pos_filter and 'segmented_with_pos' not in self.df.columns:
            self._add_pos_annotations()
        raw_texts = self.df['segmented_with_pos'].tolist() if self.use_pos_filter else self.df['segmented'].tolist()
        filtered_texts = []
        for idx, text in enumerate(raw_texts):
            if self.use_pos_filter:
                words = []
                for token in text.split():
                    if '/' not in token:
                        continue
                    word, flag = token.rsplit('/', 1)
                    if word in self.stopwords or len(word) < 2:
                        continue
                    if flag in self.keep_pos:
                        words.append(word)
            else:
                words = [w for w in text.split() if w not in self.stopwords and len(w) > 1]
            if words:
                filtered_texts.append(' '.join(words))
        self._filtered_texts_for_coherence = filtered_texts
        return filtered_texts

    def fit_model(self):
        """训练 LDA 模型"""
        if self.auto_select:
            best_n = self.select_best_topics()
            self.n_topics = best_n
        self.logger.info(f"训练 LDA 模型，主题数: {self.n_topics}")
        self.lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            learning_method='online',
            learning_decay=0.7,
            learning_offset=10.0,
            max_iter=100
        )
        self.lda.fit(self.doc_term_matrix)
        self.logger.info("模型训练完成")

    def get_topics(self, n_words=None):
        """获取每个主题的关键词列表"""
        if n_words is None:
            n_words = self.n_top_words
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
        """为每个文档分配主导主题"""
        topic_distribution = self.lda.transform(self.doc_term_matrix)
        valid_topics = topic_distribution.argmax(axis=1)
        valid_confidences = topic_distribution.max(axis=1)

        self.df['dominant_topic'] = np.nan
        self.df['topic_confidence'] = np.nan
        self.df.loc[self.valid_indices, 'dominant_topic'] = valid_topics
        self.df.loc[self.valid_indices, 'topic_confidence'] = valid_confidences
        return self.df

    def get_topic_trend(self):
        """计算主题随时间的变化趋势（需要 'date' 列）"""
        if 'date' not in self.df.columns:
            return None
        try:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            self.df = self.df.dropna(subset=['date'])
            self.df['date_month'] = self.df['date'].dt.to_period('M')
            topic_trend = self.df.groupby(['date_month', 'dominant_topic']).size().unstack(fill_value=0)
            topic_trend_pct = topic_trend.div(topic_trend.sum(axis=1), axis=0) * 100
            return topic_trend_pct
        except Exception as e:
            self.logger.error(f"趋势分析失败: {e}")
            return None

    def save_model(self, model_path, vectorizer_path):
        """保存训练好的 LDA 模型和向量化器"""
        if not JOBLIB_AVAILABLE:
            self.logger.error("joblib 未安装，无法保存模型")
            return False
        if self.lda is None:
            self.logger.error("模型未训练，无法保存")
            return False
        joblib.dump(self.lda, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        self.logger.info(f"模型已保存至 {model_path}，向量化器保存至 {vectorizer_path}")
        return True

    def load_model(self, model_path, vectorizer_path):
        """加载已保存的模型和向量化器"""
        if not JOBLIB_AVAILABLE:
            self.logger.error("joblib 未安装，无法加载模型")
            return False
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            self.logger.error("模型文件不存在")
            return False
        self.lda = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.feature_names = self.vectorizer.get_feature_names_out()
        # 注意：加载后需要重新生成 doc_term_matrix 才能分配主题，此处只加载组件
        self.logger.info("模型加载完成")
        return True

    def compute_coherence(self, n_words=None):
        """
        计算已训练模型的主题一致性（需要 gensim）
        返回平均 coherence 分数
        """
        if not GENSIM_AVAILABLE:
            self.logger.error("gensim 未安装，无法计算 coherence")
            return None
        if self.lda is None:
            self.logger.error("模型未训练")
            return None
        if n_words is None:
            n_words = self.n_top_words

        # 获取过滤后的文本分词列表
        if not hasattr(self, '_filtered_texts_for_coherence'):
            self._prepare_filtered_texts_for_coherence()
        texts = [text.split() for text in self._filtered_texts_for_coherence]

        dictionary = Dictionary(texts)
        # 构建主题词列表
        topics_words = []
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words_idx = topic.argsort()[:-n_words - 1:-1]
            top_words = [self.feature_names[i] for i in top_words_idx]
            topics_words.append(top_words)

        cm = CoherenceModel(topics=topics_words, texts=texts,
                            dictionary=dictionary, coherence='c_v')
        coherence = cm.get_coherence()
        self.logger.info(f"主题平均一致性 (c_v): {coherence:.4f}")
        return coherence

    def run(self):
        """执行完整流程：加载数据、预处理、训练、输出主题、分配主题"""
        self.load_data()
        self.prepare_corpus()
        self.fit_model()
        topics = self.get_topics()
        self.logger.info("\n各主题关键词:")
        for topic in topics:
            self.logger.info(f"主题 {topic['topic_id']}: {topic['keywords_str']}")
        self.assign_topics()
        return self.df, topics