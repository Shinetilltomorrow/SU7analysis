# data_preprocessing/segment.py
# 文本分词与自定义词典构建

import jieba
import pandas as pd
import os
import config


class TextSegmenter:
    """文本分词类，支持自定义词典和停用词"""

    def __init__(self, input_path, output_path, text_column='cleaned_text', use_stopwords=True):
        self.input_path = input_path
        self.output_path = output_path
        self.text_column = text_column
        self.use_stopwords = use_stopwords
        self.df = None
        self.stopwords = set()

        # 加载停用词表
        if use_stopwords and os.path.exists(config.STOPWORDS_PATH):
            with open(config.STOPWORDS_PATH, 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f)
            print(f"已加载停用词表，共 {len(self.stopwords)} 个词")
        else:
            # 默认停用词（备用）
            self.stopwords = {'的', '了', '是', '在', '和', '也', '都', '就', '不', '啊', '哦', '嗯'}

        # 加载用户自定义词典
        if os.path.exists(config.USER_DICT_PATH):
            jieba.load_userdict(config.USER_DICT_PATH)
            print(f"已加载用户词典: {config.USER_DICT_PATH}")
        else:
            # 内置汽车领域和网络流行语词典
            custom_words = [
                # 汽车领域术语
                '三电系统', '电池', '电机', '电控', '续航', '充电', '快充', '慢充',
                '智能座舱', '自动驾驶', '辅助驾驶', '智驾', 'NOA', '高速领航',
                '外观', '内饰', '空间', '座椅', '方向盘', '中控屏', '仪表盘',
                '性能', '加速', '百公里加速', '操控', '底盘', '悬挂',
                '价格', '性价比', '定金', '交付', '等待', '提车',
                # 小米SU7特定词汇
                '小米SU7', 'SU7', '雷军', '小米汽车', '人车合一',
                # 网络流行语
                'yyds', '绝绝子', '破防', '上头', '真香', '踩雷', '避坑',
                '冲', '蹲', '劝退', '安利', '种草', '拔草'
            ]
            for word in custom_words:
                jieba.add_word(word, freq=10000)

    def load_data(self):
        """加载清洗后的数据"""
        self.df = pd.read_csv(self.input_path, encoding='utf-8-sig')
        print(f"加载数据: {len(self.df)} 条")

    def segment(self):
        """进行分词，并过滤停用词"""
        def cut(text):
            # 使用精确模式分词
            words = jieba.cut(text, cut_all=False)
            if self.use_stopwords:
                # 过滤停用词
                words = [w for w in words if w not in self.stopwords and len(w.strip()) > 0]
            else:
                words = [w for w in words if len(w.strip()) > 0]
            return ' '.join(words)

        self.df['segmented'] = self.df[self.text_column].apply(cut)
        print("分词完成")

    def save(self):
        """保存分词结果"""
        self.df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        print(f"分词结果保存到 {self.output_path}")


if __name__ == "__main__":
    # 测试用例（需要先定义好路径）
    segmenter = TextSegmenter(
        input_path=config.CLEANED_COMMENTS_PATH,
        output_path=config.SEGMENTED_COMMENTS_PATH,
        text_column='cleaned_text'
    )
    segmenter.load_data()
    segmenter.segment()
    segmenter.save()