import os
import json
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    TrainerCallback
)
from torch.utils.data import Dataset

# ==================== 配置参数 ====================
DATA_PATH = "data/pseudo_labeled_3class.csv"          # 数据路径
MODEL_SAVE_DIR = "models/bert-finetuned-3class"       # 保存目录
LOCAL_BERT_PATH = "models/bert-base-chinese"          # 本地预训练模型路径
MAX_LEN = 128                                         # 最大序列长度
BATCH_SIZE = 8                                        # 降低 batch size 适应 4GB 显存
EPOCHS = 5                                            # 最大 epoch，早停会提前停止
LEARNING_RATE = 1e-5                                  # 学习率
WARMUP_RATIO = 0.1                                    # 预热比例（占总步数的比例）
WEIGHT_DECAY = 0.05                                   # 权重衰减
USE_FOCAL_LOSS = True                                 # 使用 Focal Loss 处理类别不平衡
USE_OVERSAMPLING = False                              # 关闭过采样（已有 Focal Loss 和类别权重）
USE_TEXT_AUGMENTATION = False                         # 关闭文本增强（网络问题）
USE_HYPERPARAM_SEARCH = False                         # 关闭超参数搜索（节省显存和时间）
RANDOM_SEED = 42                                      # 随机种子
# =================================================

# 设置随机种子
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True   # 加速 GPU 训练

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 加载数据
df = pd.read_csv(DATA_PATH)
df['cleaned_text'] = df['cleaned_text'].fillna('').astype(str)

# 2. 标签编码
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])
texts = df['cleaned_text'].tolist()
labels = df['label_encoded'].tolist()
print(f"标签映射: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# 3. 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=RANDOM_SEED, stratify=labels
)

print(f"原始训练集大小: {len(train_texts)}")
print(pd.Series(train_labels).value_counts().sort_index())
print(f"验证集大小: {len(val_texts)}")
print(pd.Series(val_labels).value_counts().sort_index())

# 4. 过采样（可选，已关闭）
if USE_OVERSAMPLING:
    try:
        from imblearn.over_sampling import RandomOverSampler
        X_train = np.array(train_texts).reshape(-1, 1)
        y_train = np.array(train_labels)
        ros = RandomOverSampler(random_state=RANDOM_SEED)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
        train_texts = X_resampled.ravel().tolist()
        train_labels = y_resampled.tolist()
        print(f"过采样后训练集大小: {len(train_texts)}")
        print(pd.Series(train_labels).value_counts().sort_index())
    except ImportError:
        print("警告：未安装 imbalanced-learn，跳过过采样。")
        USE_OVERSAMPLING = False

# 5. 分词器
tokenizer = BertTokenizerFast.from_pretrained(LOCAL_BERT_PATH)

# 6. Dataset 类（返回列表，动态填充）
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            return_tensors=None
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': label
        }

train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LEN)

# 7. 数据整理器
data_collator = DataCollatorWithPadding(tokenizer)

# 8. 类别权重
unique_classes = np.unique(train_labels)
class_weights = compute_class_weight('balanced', classes=unique_classes, y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"类别权重: {class_weights}")

# 9. Focal Loss
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 10. 自定义 Trainer（支持类别权重和 Focal Loss）
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, use_focal=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal = use_focal
        if use_focal:
            self.focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.use_focal:
            loss = self.focal_loss(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 11. 评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    auc_macro = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'auc_macro': auc_macro
    }

# 12. 自定义回调（打印详细指标）
class DetailedLoggingCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if state.log_history:
            for log in reversed(state.log_history):
                if 'eval_loss' in log:
                    print(f"\nEpoch {state.epoch}: eval_loss={log['eval_loss']:.4f}, eval_f1_macro={log.get('eval_f1_macro', 0):.4f}")
                    break

# 13. 模型初始化函数（用于超参数搜索，当前未启用但保留）
def model_init():
    return BertForSequenceClassification.from_pretrained(LOCAL_BERT_PATH, num_labels=len(label_encoder.classes_))

# 14. 训练参数
total_steps = len(train_dataset) // BATCH_SIZE * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=warmup_steps,
    weight_decay=WEIGHT_DECAY,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    lr_scheduler_type="cosine",
    report_to="none",
    dataloader_num_workers=0 if os.name == 'nt' else 2,
    dataloader_pin_memory=torch.cuda.is_available(),
    fp16=torch.cuda.is_available(),   # GPU 时自动开启混合精度
    save_total_limit=2,
)

# 15. 超参数搜索（已关闭，如需开启请安装 optuna 并设为 True）
if USE_HYPERPARAM_SEARCH:
    try:
        import optuna
        def optuna_hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            }
        search_trainer = WeightedTrainer(
            class_weights=class_weights,
            use_focal=USE_FOCAL_LOSS,
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )
        best_run = search_trainer.hyperparameter_search(
            hp_space=optuna_hp_space,
            n_trials=5,
            direction="maximize",
            compute_objective=lambda metrics: metrics["eval_f1_macro"]
        )
        print(f"最佳超参数: {best_run.hyperparameters}")
        for param, value in best_run.hyperparameters.items():
            setattr(training_args, param, value)
    except ImportError:
        print("警告：未安装 optuna，跳过超参数搜索。请安装：pip install optuna")
        USE_HYPERPARAM_SEARCH = False

# 16. 最终 Trainer
trainer = WeightedTrainer(
    class_weights=class_weights,
    use_focal=USE_FOCAL_LOSS,
    model=model_init(),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# 17. 添加回调
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=2,
    early_stopping_threshold=0.001
)
trainer.add_callback(early_stopping)
trainer.add_callback(DetailedLoggingCallback())

# 18. 训练
print("开始三分类微调...")
trainer.train()

# 19. 保存模型、分词器、标签映射和配置
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
trainer.model.save_pretrained(MODEL_SAVE_DIR)
tokenizer.save_pretrained(MODEL_SAVE_DIR)

# 保存标签映射
label_mapping = {int(label): str(cls) for cls, label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
with open(os.path.join(MODEL_SAVE_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, ensure_ascii=False, indent=2)

# 保存配置
config = {
    "data_path": DATA_PATH,
    "max_len": MAX_LEN,
    "batch_size": training_args.per_device_train_batch_size,
    "epochs": training_args.num_train_epochs,
    "learning_rate": training_args.learning_rate,
    "warmup_ratio": WARMUP_RATIO,
    "weight_decay": training_args.weight_decay,
    "use_focal_loss": USE_FOCAL_LOSS,
    "use_oversampling": USE_OVERSAMPLING,
    "use_hyperparam_search": USE_HYPERPARAM_SEARCH,
    "random_seed": RANDOM_SEED,
    "label_mapping": label_mapping,
    "class_weights": class_weights.tolist(),
}
with open(os.path.join(MODEL_SAVE_DIR, "training_config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
print(f"模型及配置已保存至 {MODEL_SAVE_DIR}")

# 20. 最终评估
eval_results = trainer.evaluate()
print("\n验证集最终结果：")
for key, value in eval_results.items():
    print(f"  {key}: {value:.4f}")

# 21. 分类报告
predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
target_names = [str(cls) for cls in label_encoder.classes_]
report = classification_report(val_labels, preds, target_names=target_names)
print("\n分类报告：")
print(report)

with open(os.path.join(MODEL_SAVE_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)
print(f"分类报告已保存至 {os.path.join(MODEL_SAVE_DIR, 'classification_report.txt')}")