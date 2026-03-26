# 分词与词库构建
# data_preprocessing/segment.py
# 文本分词与自定义词典构建

import jieba
import pandas as pd
import config


class TextSegmenter:
    """文本分词类"""

    def __init__(self, input_path, output_path, text_column='cleaned_text'):
        self.input_path = input_path
        self.output_path = output_path
        self.text_column = text_column
        self.df = None

        # 自定义词典（汽车领域术语 + 网络流行语）
        self.custom_words = [
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

        # 加载自定义词典（修复 jieba 调用，提供频率参数）
        for word in self.custom_words:
            jieba.add_word(word, freq=10000)  # 增加频率，确保被识别

    def load_data(self):
        """加载清洗后的数据"""
        self.df = pd.read_csv(self.input_path, encoding='utf-8-sig')
        print(f"加载数据: {len(self.df)} 条")

    def segment(self):
        """进行分词"""
        def cut(text):
            words = jieba.cut(text)
            # 过滤停用词（这里简化，实际需要加载停用词表）
            stopwords = ['的', '了', '是', '在', '和', '也', '都', '就', '不', '啊', '哦', '嗯']
            words = [w for w in words if w not in stopwords and len(w) > 0]
            return ' '.join(words)

        self.df['segmented'] = self.df[self.text_column].apply(cut)
        print("分词完成")

    def save(self):
        """保存分词结果"""
        self.df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        print(f"分词结果保存到 {self.output_path}")


# 使用示例（仅用于测试，不依赖 config）
if __name__ == "__main__":
    # 注意：需要先定义 DataCleaner 或直接导入
    from data_preprocessing.clean import DataCleaner
    cleaner = DataCleaner(config.RAW_DATA_PATH, "data/processed/cleaned_comments.csv")
    cleaner.run()

    # 分词（弹幕使用 cleaned_text）
    segmenter = TextSegmenter("data/processed/cleaned_comments.csv", config.PROCESSED_DATA_PATH)
    segmenter.load_data()
    segmenter.segment()
    segmenter.save()