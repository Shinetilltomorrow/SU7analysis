# data_preprocessing/segment.py
import jieba
import jieba.posseg as pseg
import pandas as pd
import re
import os
import config


class TextSegmenter:
    def __init__(self, input_path, output_path, text_column='cleaned_text', use_stopwords=True, use_pos_filter=False):
        self.input_path = input_path
        self.output_path = output_path
        self.text_column = text_column
        self.use_stopwords = use_stopwords
        self.use_pos_filter = use_pos_filter
        self.df = None
        self.stopwords = set()

        if use_stopwords and os.path.exists(config.STOPWORDS_PATH):
            with open(config.STOPWORDS_PATH, 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f if line.strip())
            config.logger.info(f"已加载停用词表，共 {len(self.stopwords)} 个词")

        if os.path.exists(config.USER_DICT_PATH):
            jieba.load_userdict(config.USER_DICT_PATH)
            config.logger.info(f"已加载用户词典: {config.USER_DICT_PATH}")
        else:
            custom_words = [
                '三电系统', '电池', '电机', '电控', '续航', '充电', '快充', '慢充',
                '智能座舱', '自动驾驶', '辅助驾驶', '智驾', 'NOA', '高速领航',
                '外观', '内饰', '空间', '座椅', '方向盘', '中控屏', '仪表盘',
                '性能', '加速', '百公里加速', '操控', '底盘', '悬挂',
                '价格', '性价比', '定金', '交付', '等待', '提车',
                '小米SU7', 'SU7', '雷军', '小米汽车', '人车合一',
                'yyds', '绝绝子', '破防', '上头', '真香', '踩雷', '避坑',
                '冲', '蹲', '劝退', '安利', '种草', '拔草'
            ]
            for word in custom_words:
                jieba.add_word(word, freq=10000)
            config.logger.info("已加载内置用户词典")

        self.keep_pos = {'n', 'v', 'a', 'an', 'vn', 'ad', 'i', 'l', 'j'}  # 名词、动词、形容词等

    def load_data(self):
        self.df = pd.read_csv(self.input_path, encoding='utf-8-sig')
        config.logger.info(f"加载数据: {len(self.df)} 条")

    def _simplify_repeated_chars(self, text):
        return re.sub(r'(.)\1{2,}', r'\1', text)

    def segment(self):
        def cut(text):
            text = self._simplify_repeated_chars(text)
            if self.use_pos_filter:
                words = []
                for word, flag in pseg.cut(text):
                    if self.use_stopwords and (word in self.stopwords or len(word) < 2):
                        continue
                    if flag in self.keep_pos:
                        words.append(word)
            else:
                words = jieba.cut(text, cut_all=False)
                if self.use_stopwords:
                    words = [w for w in words if w not in self.stopwords and len(w) > 1]
                else:
                    words = [w for w in words if len(w) > 1]
            return ' '.join(words)

        self.df['segmented'] = self.df[self.text_column].apply(cut)
        config.logger.info("分词完成")

    def save(self):
        self.df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        config.logger.info(f"分词结果保存到 {self.output_path}")