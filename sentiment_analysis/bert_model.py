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
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.num_labels = self.model.config.num_labels
        config.logger.info(f"BERT 模型已加载，设备: {self.device}，类别数: {self.num_labels}")

        # 定义标签映射（与训练时一致）
        if self.num_labels == 3:
            # 三分类：0=负面, 1=中性, 2=正面
            self.id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        elif self.num_labels == 2:
            # 二分类：0=负面, 1=正面
            self.id_to_label = {0: 'negative', 1: 'positive'}
        else:
            raise ValueError(f"不支持的类别数: {self.num_labels}，仅支持 2 或 3")
        config.logger.info(f"标签映射: {self.id_to_label}")

    def _predict_batch(self, texts, max_len=128):
        """
        批量预测，返回：
        - pred_labels: 预测的类别索引列表
        - pred_confidences: 最高概率值列表
        - positive_probs: 正面类别的概率列表（若为三分类则取索引2，二分类取索引1）
        """
        all_pred_labels = []
        all_pred_confidences = []
        all_positive_probs = []
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
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()  # (batch, num_labels)
                pred_labels = probs.argmax(axis=1).tolist()
                pred_confidences = probs.max(axis=1).tolist()
                # 正面概率：三分类取第2列（索引2），二分类取第1列（索引1）
                if self.num_labels == 3:
                    positive_probs = probs[:, 2].tolist()
                else:  # 二分类
                    positive_probs = probs[:, 1].tolist()
                all_pred_labels.extend(pred_labels)
                all_pred_confidences.extend(pred_confidences)
                all_positive_probs.extend(positive_probs)
            config.logger.debug(f"进度: {i+len(batch_texts)}/{total}")
        return all_pred_labels, all_pred_confidences, all_positive_probs

    def analyze(self):
        """对分词后的弹幕进行情感分析，添加 sentiment_label, sentiment_confidence, sentiment_score 三列"""
        self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        if 'cleaned_text' in self.df.columns:
            text_col = 'cleaned_text'
        elif 'segmented' in self.df.columns:
            text_col = 'segmented'
        else:
            raise ValueError("数据中缺少文本列（cleaned_text 或 segmented）")
        texts = self.df[text_col].fillna('').astype(str).tolist()

        config.logger.info(f"开始 BERT 情感预测，共 {len(texts)} 条，批量大小 {self.batch_size}")
        pred_labels, pred_confidences, positive_probs = self._predict_batch(texts)

        # 添加结果列
        self.df['sentiment_label'] = [self.id_to_label[lbl] for lbl in pred_labels]
        self.df['sentiment_confidence'] = pred_confidences   # 预测置信度
        self.df['sentiment_score'] = positive_probs          # 正面概率（兼容旧版）
        config.logger.info("BERT 情感分析完成")
        return self.df