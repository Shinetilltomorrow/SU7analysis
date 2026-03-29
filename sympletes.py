# 在项目根目录下新建 quick_sentiment.py
import config
from steps import step_sentiment_analysis

# 确认 BERT 启用
config.USE_BERT = True
config.SENTIMENT_METHOD = 'bert'
# 根据需要修改分类方法
config.SENTIMENT_CLASSIFY_METHOD = 'percentile'
config.POSITIVE_PERCENTILE = 15
config.NEGATIVE_PERCENTILE = 15

sentiment_df = step_sentiment_analysis()
print(sentiment_df['sentiment_label'].value_counts())