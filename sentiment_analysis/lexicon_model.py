# sentiment_analysis/lexicon_model.py
import pandas as pd
import numpy as np
import os
import config


class LexiconSentimentAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.pos_words = set()
        self.neg_words = set()
        self.degree_words = {}
        self.negation_words = set()
        self._load_dictionary()

    def _load_dictionary(self):
        if os.path.exists(config.POS_DICT_PATH):
            with open(config.POS_DICT_PATH, 'r', encoding='utf-8') as f:
                self.pos_words = set(line.strip() for line in f if line.strip())
        else:
            self.pos_words = set()
        if os.path.exists(config.NEG_DICT_PATH):
            with open(config.NEG_DICT_PATH, 'r', encoding='utf-8') as f:
                self.neg_words = set(line.strip() for line in f if line.strip())
        else:
            self.neg_words = set()
        if os.path.exists(config.DEGREE_DICT_PATH):
            with open(config.DEGREE_DICT_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    if ' ' in line:
                        word, weight = line.strip().split()
                        self.degree_words[word] = float(weight)
        else:
            self.degree_words = {}
        if os.path.exists(config.NEGATION_DICT_PATH):
            with open(config.NEGATION_DICT_PATH, 'r', encoding='utf-8') as f:
                self.negation_words = set(line.strip() for line in f if line.strip())
        else:
            self.negation_words = set()

        config.logger.info(f"情感词典加载：积极词{len(self.pos_words)}，消极词{len(self.neg_words)}，程度词{len(self.degree_words)}，否定词{len(self.negation_words)}")

    def calculate_score(self, text):
        if not isinstance(text, str):
            return 0.5
        words = text.split()
        score = 0
        i = 0
        negation = 1
        degree = 1

        while i < len(words):
            word = words[i]
            if word in self.negation_words:
                negation *= -1
                i += 1
                continue
            if word in self.degree_words:
                degree = self.degree_words[word]
                i += 1
                continue
            if word in self.pos_words:
                score += 1 * degree * negation
                negation = 1
                degree = 1
            elif word in self.neg_words:
                score -= 1 * degree * negation
                negation = 1
                degree = 1
            i += 1

        normalized = (np.tanh(score) + 1) / 2
        normalized = max(0.0, min(1.0, normalized))
        return normalized

    def analyze(self):
        self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        self.df['segmented'] = self.df['segmented'].fillna('').astype(str)
        self.df['sentiment_score'] = self.df['segmented'].apply(self.calculate_score)
        self.df['sentiment_label'] = self.df['sentiment_score'].apply(
            lambda x: 'positive' if x >= config.POSITIVE_THRESHOLD else ('negative' if x <= config.NEGATIVE_THRESHOLD else 'neutral')
        )
        sentiment_counts = self.df['sentiment_label'].value_counts()
        config.logger.info(f"情感分布:\n{sentiment_counts}")
        return self.df