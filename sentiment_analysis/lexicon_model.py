# 基于情感词典的方法
# sentiment_analysis/lexicon_model.py
# 基于情感词典的情感分析

import pandas as pd
import numpy as np
import config


class LexiconSentimentAnalyzer:
    """基于情感词典的情感分析器"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.pos_words = set()  # 积极词库
        self.neg_words = set()  # 消极词库
        self.degree_words = {}  # 程度副词
        self.negation_words = set()  # 否定词

        self._load_dictionary()

    def _load_dictionary(self):
        """加载情感词典"""
        # 这里使用简化的词典，实际应加载完整的情感词典
        # 积极词
        self.pos_words = {
            '好', '棒', '赞', '优秀', '厉害', '牛逼', '惊艳', '完美',
            '喜欢', '爱', '值得', '推荐', '满意', '惊喜', '流畅',
            '稳定', '可靠', '省心', '划算', '超值', 'yyds', '真香'
        }

        # 消极词
        self.neg_words = {
            '差', '烂', '垃圾', '失望', '后悔', '坑', '问题', '故障',
            '异响', '漏水', '卡顿', '慢', '贵', '不值', '劝退', '踩雷',
            '虚标', '缩水', '延迟', '不好', '不行', '糟糕'
        }

        # 否定词
        self.negation_words = {
            '不', '没', '无', '非', '未', '别', '勿', '莫'
        }

        # 程度副词（简化）
        self.degree_words = {
            '很': 1.5, '非常': 2.0, '特别': 2.0, '超级': 2.5,
            '太': 1.8, '极': 2.0, '最': 2.0, '有点': 0.7,
            '稍微': 0.5, '一般': 0.8, '比较': 1.2
        }

    def calculate_score(self, text):
        """计算单条弹幕的情感得分"""
        words = text.split()
        score = 0
        i = 0
        negation = 1  # 否定标记，1表示无否定，-1表示有否定
        degree = 1  # 程度系数

        while i < len(words):
            word = words[i]

            # 检查是否是否定词
            if word in self.negation_words:
                negation = -1
                i += 1
                continue

            # 检查是否是程度副词
            if word in self.degree_words:
                degree = self.degree_words[word]
                i += 1
                continue

            # 计算情感得分
            if word in self.pos_words:
                score += 1 * degree * negation
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
        """保存结果"""
        self.df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"情感分析结果保存到 {output_path}")



# 使用示例
if __name__ == "__main__":
    # 基于词典的方法
    analyzer = LexiconSentimentAnalyzer(config.PROCESSED_DATA_PATH)
    result_df = analyzer.analyze()
    analyzer.save("results/sentiment_lexicon.csv")