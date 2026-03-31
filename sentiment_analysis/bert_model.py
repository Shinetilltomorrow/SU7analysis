# sentiment_analysis/bert_model.py
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import config


class BERTSentimentAnalyzer:
    def __init__(self, data_path, model_path=None, batch_size=32):
        self.data_path = data_path
        self.batch_size = batch_size
        self.model_path = model_path or config.BERT_MODEL_PATH

        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)  # 自动读取 num_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        config.logger.info(f"BERT 模型已加载，设备: {self.device}")
        # 检查模型输出维度
        self.num_labels = self.model.config.num_labels
        config.logger.info(f"模型类别数: {self.num_labels}")

    def _predict_batch(self, texts, max_len=128):
        all_scores = []
        total = len(texts)
        for i in range(0, total, self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            inputs = self.tokenizer(
                batch_texts,
                max_length=max_len,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                if self.num_labels == 3:
                    # 假设类别顺序：0=negative, 1=neutral, 2=positive
                    # 将情感得分定义为正面概率
                    scores = probs[:, 2]
                else:
                    # 二分类情况，取正面概率
                    scores = probs[:, 1]
                all_scores.extend(scores)
            config.logger.debug(f"进度: {i+len(batch_texts)}/{total}")
        return all_scores

    def analyze(self):
        self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        if 'cleaned_text' in self.df.columns:
            text_col = 'cleaned_text'
        elif 'segmented' in self.df.columns:
            text_col = 'segmented'
        else:
            raise ValueError("数据中缺少文本列（cleaned_text或segmented）")
        texts = self.df[text_col].fillna('').astype(str).tolist()

        config.logger.info(f"开始 BERT 情感预测，共 {len(texts)} 条，批量大小 {self.batch_size}")
        scores = self._predict_batch(texts)
        self.df['sentiment_score'] = scores
        self.df['sentiment_label'] = self.df['sentiment_score'].apply(
            lambda x: 'positive' if x >= config.POSITIVE_THRESHOLD
            else ('negative' if x <= config.NEGATIVE_THRESHOLD else 'neutral')
        )
        config.logger.info("BERT 情感分析完成")
        return self.df