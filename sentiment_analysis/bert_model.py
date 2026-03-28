# sentiment_analysis/bert_model.py
# 基于BERT的情感分析（需要transformers库）
import os
import pandas as pd
import torch
import config

try:
    from transformers import BertTokenizer, BertForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("警告：未安装transformers库，BERT情感分析将不可用。请运行 pip install transformers")


class BERTSentimentAnalyzer:
    """基于BERT的情感分析器（三分类）"""

    def __init__(self, data_path, model_path=None):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers库未安装，无法使用BERT情感分析")
        self.data_path = data_path
        self.df = None
        self.model_path = model_path or config.BERT_MODEL_PATH

        # 加载分词器和模型
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=3  # 三分类
            )
            self.model.eval()
            print(f"已加载BERT模型: {self.model_path}")
        except Exception as e:
            print(f"加载BERT模型失败: {e}")
            print("请确保模型文件存在，或设置 USE_BERT=False 使用词典方法")
            raise

    def _predict_single(self, text, max_len=128):
        """单条文本的情感预测"""
        # 编码
        inputs = self.tokenizer(
            text,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
            # 假设标签顺序：0=消极, 1=中性, 2=积极
            score = probs[2]  # 取积极概率作为得分（与词典方法一致）
        return score

    def analyze(self):
        """执行情感分析"""
        self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        # 使用清洗后的文本或分词后的文本（BERT通常使用原始文本）
        text_col = 'cleaned_text' if 'cleaned_text' in self.df.columns else 'segmented'
        if text_col not in self.df.columns:
            raise ValueError("数据中缺少文本列（cleaned_text或segmented）")
        self.df[text_col] = self.df[text_col].fillna('').astype(str)

        # 预测情感得分（可能较慢，可以分批处理）
        print("正在进行BERT情感预测（可能需要一段时间）...")
        scores = []
        for text in self.df[text_col]:
            score = self._predict_single(text)
            scores.append(score)
        self.df['sentiment_score'] = scores

        # 情感分类
        self.df['sentiment_label'] = self.df['sentiment_score'].apply(
            lambda x: 'positive' if x >= config.POSITIVE_THRESHOLD
            else ('negative' if x <= config.NEGATIVE_THRESHOLD else 'neutral')
        )

        # 统计
        sentiment_counts = self.df['sentiment_label'].value_counts()
        print("情感分布:")
        print(sentiment_counts)

        return self.df

    def save(self, output_path):
        # 不再直接使用 output_path，而是交给 SaveData 处理
        config.SaveData(self.df, result_type="result", filename=os.path.basename(output_path)).save()
        print(f"BERT情感分析结果保存到 {output_path}")


if __name__ == "__main__":
    # 使用示例（需先下载模型或指定路径）
    # 如果 config.USE_BERT 为 True，则使用BERT；否则会提示
    if config.USE_BERT:
        analyzer = BERTSentimentAnalyzer(config.SEGMENTED_COMMENTS_PATH)
        result_df = analyzer.analyze()
        analyzer.save("results/sentiment_bert.csv")
    else:
        print("请在 config.py 中设置 USE_BERT=True 以使用BERT情感分析")