# sentiment_analysis/lexicon_model.py
# 基于情感词典的情感分析（支持外部词典文件）

import pandas as pd
import numpy as np
import os
import config


class LexiconSentimentAnalyzer:
    """基于情感词典的情感分析器（支持自定义词典）"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.pos_words = set()
        self.neg_words = set()
        self.degree_words = {}
        self.negation_words = set()
        self._load_dictionary()

    def _load_dictionary(self):
        """加载情感词典，优先从配置文件路径加载，否则使用内置词典"""
        # 积极词
        if os.path.exists(config.POS_DICT_PATH):
            with open(config.POS_DICT_PATH, 'r', encoding='utf-8') as f:
                self.pos_words = set(line.strip() for line in f if line.strip())
        else:
            self.pos_words = {
                '好', '棒', '赞', '优秀', '厉害', '牛逼', '惊艳', '完美',
                '喜欢', '爱', '值得', '推荐', '满意', '惊喜', '流畅',
                '稳定', '可靠', '省心', '划算', '超值', 'yyds', '真香'
            }

        # 消极词
        if os.path.exists(config.NEG_DICT_PATH):
            with open(config.NEG_DICT_PATH, 'r', encoding='utf-8') as f:
                self.neg_words = set(line.strip() for line in f if line.strip())
        else:
            self.neg_words = {
                '差', '烂', '垃圾', '失望', '后悔', '坑', '问题', '故障',
                '异响', '漏水', '卡顿', '慢', '贵', '不值', '劝退', '踩雷',
                '虚标', '缩水', '延迟', '不好', '不行', '糟糕'
            }

        # 程度副词
        if os.path.exists(config.DEGREE_DICT_PATH):
            with open(config.DEGREE_DICT_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    if ' ' in line:
                        word, weight = line.strip().split()
                        self.degree_words[word] = float(weight)
        else:
            self.degree_words = {
                '很': 1.5, '非常': 2.0, '特别': 2.0, '超级': 2.5,
                '太': 1.8, '极': 2.0, '最': 2.0, '有点': 0.7,
                '稍微': 0.5, '一般': 0.8, '比较': 1.2
            }

        # 否定词
        if os.path.exists(config.NEGATION_DICT_PATH):
            with open(config.NEGATION_DICT_PATH, 'r', encoding='utf-8') as f:
                self.negation_words = set(line.strip() for line in f if line.strip())
        else:
            self.negation_words = {'不', '没', '无', '非', '未', '别', '勿', '莫'}

        print(f"情感词典加载完成：积极词{len(self.pos_words)}，消极词{len(self.neg_words)}，"
              f"程度词{len(self.degree_words)}，否定词{len(self.negation_words)}")

    def calculate_score(self, text):
        """计算单条弹幕的情感得分，考虑否定词和程度副词的作用范围"""
        if not isinstance(text, str):
            return 0.5
        words = text.split()
        score = 0
        i = 0
        negation = 1      # 1表示无否定，-1表示有否定
        degree = 1        # 程度系数

        while i < len(words):
            word = words[i]

            # 检查是否是否定词
            if word in self.negation_words:
                # 否定词作用于其后直到下一个情感词或程度词（简化：影响后3个词）
                negation = -1
                # 这里可以设定一个作用范围，比如影响后面3个词，但为了简单，我们直接标记，然后在遇到情感词时应用
                i += 1
                # 继续循环，等待情感词
                continue

            # 检查是否是程度副词
            if word in self.degree_words:
                degree = self.degree_words[word]
                i += 1
                continue

            # 计算情感得分
            if word in self.pos_words:
                score += 1 * degree * negation
                # 重置否定和程度（假设只影响最近一个情感词）
                negation = 1
                degree = 1
            elif word in self.neg_words:
                score -= 1 * degree * negation
                negation = 1
                degree = 1

            i += 1

        # 将得分归一化到 [0, 1] 区间
        normalized = (np.tanh(score) + 1) / 2
        return normalized

    def analyze(self):
        """执行情感分析"""
        self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        # 确保 segmented 列是字符串
        self.df['segmented'] = self.df['segmented'].fillna('').astype(str)
        self.df['sentiment_score'] = self.df['segmented'].apply(self.calculate_score)

        # 情感分类
        self.df['sentiment_label'] = self.df['sentiment_score'].apply(
            lambda x: 'positive' if x >= config.POSITIVE_THRESHOLD
            else ('negative' if x <= config.NEGATIVE_THRESHOLD else 'neutral')
        )

        # 统计情感分布
        sentiment_counts = self.df['sentiment_label'].value_counts()
        print("情感分布:")
        print(sentiment_counts)

        return self.df

    def save(self, output_path):
        # 不再直接使用 output_path，而是交给 SaveData 处理
        config.SaveData(self.df, result_type="result", filename=os.path.basename(output_path)).save()
        print(f"情感分析结果保存到 {output_path}")


if __name__ == "__main__":
    analyzer = LexiconSentimentAnalyzer(config.SEGMENTED_COMMENTS_PATH)
    result_df = analyzer.analyze()
    analyzer.save("results/sentiment_lexicon.csv")