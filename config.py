# config.py
import os
import pandas as pd
from datetime import datetime
import logging
import sys

# 采集配置
KEYWORDS = ["SU7", "小米SU7", "小米汽车SU7"]
START_DATE = "2024-04-01"
END_DATE = "2026-01-31"

# 情感分析配置
POSITIVE_THRESHOLD = 0.6
NEGATIVE_THRESHOLD = 0.4
USE_BERT = True                     # 启用 BERT
SENTIMENT_METHOD = 'bert'           # 使用 BERT 方法
BERT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bert-base-chinese")

# LDA 主题模型配置
N_TOPICS = None                     # 自动选择主题数
AUTO_SELECT_TOPICS = True
N_TOP_WORDS = 10
USE_POS_FILTER = True
LDA_MAX_DF = 0.85
LDA_MIN_DF = 3

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
SEGMENTED_VIDEOS_PATH = os.path.join(BASE_DIR, "data", "processed", "segmented_videos.csv")
SEGMENTED_COMMENTS_PATH = os.path.join(BASE_DIR, "data", "processed", "segmented_comments.csv")
SALES_DATA_PATH = os.path.join(BASE_DIR, "data", "sales", "xiaomi_su7_sales.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "results")

CLEANED_COMMENTS_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_comments.csv")
PROCESSED_VIDEOS_PATH = os.path.join(BASE_DIR, "data", "processed", "videos", "cleaned_videos.csv")

STOPWORDS_PATH = os.path.join(BASE_DIR, "data", "stopwords.txt")
USER_DICT_PATH = os.path.join(BASE_DIR, "data", "user_dict.txt")
POS_DICT_PATH = os.path.join(BASE_DIR, "data", "sentiment", "positive_words.txt")
NEG_DICT_PATH = os.path.join(BASE_DIR, "data", "sentiment", "negative_words.txt")
DEGREE_DICT_PATH = os.path.join(BASE_DIR, "data", "sentiment", "degree_words.txt")
NEGATION_DICT_PATH = os.path.join(BASE_DIR, "data", "sentiment", "negation_words.txt")

# 日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, 'analysis.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def print_step(step_name, is_start=True):
    if is_start:
        print(f"\n{'='*60}\n【{step_name}】开始\n{'='*60}")
    else:
        print(f"\n{'='*60}\n【{step_name}】完成\n{'='*60}\n")

def print_table(df, title=None):
    if title:
        print(f"\n{title}")
    print(df.to_string(index=False))

def ensure_directories():
    dirs = [
        os.path.dirname(SEGMENTED_VIDEOS_PATH),
        os.path.dirname(SEGMENTED_COMMENTS_PATH),
        RESULTS_PATH,
        os.path.dirname(SALES_DATA_PATH),
        RAW_DATA_DIR,
        os.path.join(BASE_DIR, "data", "processed", "videos"),
        os.path.join(BASE_DIR, "data", "processed", "danmaku"),
        os.path.dirname(STOPWORDS_PATH),
        os.path.dirname(USER_DICT_PATH),
        os.path.dirname(POS_DICT_PATH),
        os.path.dirname(BERT_MODEL_PATH),
    ]
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)

def create_default_dicts():
    # 停用词表（已扩充）
    if not os.path.exists(STOPWORDS_PATH):
        os.makedirs(os.path.dirname(STOPWORDS_PATH), exist_ok=True)
        default_stopwords = [
            '的', '了', '是', '在', '和', '也', '都', '就', '不', '啊', '哦', '嗯', '吧', '吗', '呢', '呀',
            '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿',
            '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们', '它们', '自己', '别人',
            '这', '那', '这些', '那些', '这个', '那个', '这么', '那么', '这样', '那样',
            '什么', '怎么', '为什么', '哪里', '哪儿', '谁', '什么时候',
            '会', '能', '可以', '可能', '应该', '需要', '想要', '觉得', '认为', '感觉',
            '说', '看', '听', '走', '去', '来', '做', '用', '给', '让', '把', '被',
            '有', '没有', '是', '不是', '也', '还', '都', '只', '就', '才', '又', '再',
            '很', '太', '非常', '特别', '十分', '更', '最', '比较', '稍微', '有点',
            '因为', '所以', '但是', '然而', '虽然', '尽管', '如果', '那么', '否则',
            'a', 'an', 'the', 'and', 'or', 'but', 'so', 'if', 'then', 'else', 'for', 'with',
            'to', 'of', 'in', 'on', 'at', 'by', 'from', 'as', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'this', 'that', 'these',
            'those', 'it', 'its', 'they', 'them', 'their', 'he', 'him', 'his', 'she', 'her',
            'we', 'us', 'our', 'you', 'your', 'i', 'me', 'my', 'mine',
            '哈', '嘻嘻', '呵呵', '哈哈', '嘿嘿', '呜呜', '喔', '嚯', '呼', '咩',
            '啦', '咯', '呵', '嘿', '哇', '哎', '哟', '喂', '嗯', '哦',
            '看', '瞧', '望', '瞅', '瞄', '盯', '瞪', '瞥',
            '说', '讲', '谈', '聊', '唠', '扯', '吹',
            '搞', '弄', '整', '干', '做', '办', '处理',
        ]
        with open(STOPWORDS_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(default_stopwords))
        logger.info(f"已创建停用词表（{len(default_stopwords)}词）: {STOPWORDS_PATH}")

    # 用户词典
    if not os.path.exists(USER_DICT_PATH):
        os.makedirs(os.path.dirname(USER_DICT_PATH), exist_ok=True)
        default_user_words = ['三电系统', '智能座舱', '辅助驾驶', '小米SU7', '雷军', 'yyds', '保时捷', '特斯拉', '续航', '充电', '交付']
        with open(USER_DICT_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(default_user_words))
        logger.info(f"已创建用户词典: {USER_DICT_PATH}")

    # 积极词
    if not os.path.exists(POS_DICT_PATH):
        os.makedirs(os.path.dirname(POS_DICT_PATH), exist_ok=True)
        default_pos = [
            '好', '棒', '赞', '优秀', '厉害', '惊艳', '完美', '喜欢', '爱', '值得', '推荐', '满意', '惊喜', '流畅', '稳定', '可靠', '省心', '划算', '超值', 'yyds', '真香',
            '舒服', '爽', '牛逼', '给力', '酷', '帅', '漂亮', '好看', '美观', '大气', '高端', '豪华', '舒适', '安静', '平顺', '强劲', '快', '迅速', '灵敏', '精准', '智能', '科技',
            '安全', '放心', '省油', '省钱', '经济', '环保', '先进', '创新', '独特', '新颖', '可靠', '耐用', '保值', '有面子', '有档次', '有品位', '有范儿', '拉风', '吸睛'
        ]
        with open(POS_DICT_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(default_pos))
        logger.info(f"已创建积极词表: {POS_DICT_PATH}")

    # 消极词
    if not os.path.exists(NEG_DICT_PATH):
        os.makedirs(os.path.dirname(NEG_DICT_PATH), exist_ok=True)
        default_neg = [
            '差', '烂', '垃圾', '失望', '后悔', '坑', '问题', '故障', '异响', '漏水', '卡顿', '慢', '贵', '不值', '劝退', '踩雷', '虚标', '缩水', '延迟', '不好', '不行', '糟糕',
            '恶心', '烦', '蠢', '傻', '笨', '差劲', '劣质', '粗糙', '简陋', '难用', '难开', '难坐', '不舒服', '不舒适', '不安静', '噪音大', '颠簸', '顿挫', '顿挫感', '续航短',
            '充电慢', '服务差', '售后差', '欺诈', '虚假', '夸大', '忽悠', '坑人', '骗人', '后悔', '不值', '太贵', '溢价', '加价', '提车慢', '等待久', '交付延迟'
        ]
        with open(NEG_DICT_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(default_neg))
        logger.info(f"已创建消极词表: {NEG_DICT_PATH}")

    # 程度副词
    if not os.path.exists(DEGREE_DICT_PATH):
        os.makedirs(os.path.dirname(DEGREE_DICT_PATH), exist_ok=True)
        default_degree = ['很 1.5', '非常 2.0', '特别 2.0', '超级 2.5', '太 1.8', '极 2.0', '最 2.0', '有点 0.7', '稍微 0.5', '一般 0.8', '比较 1.2']
        with open(DEGREE_DICT_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(default_degree))
        logger.info(f"已创建程度副词表: {DEGREE_DICT_PATH}")

    # 否定词
    if not os.path.exists(NEGATION_DICT_PATH):
        os.makedirs(os.path.dirname(NEGATION_DICT_PATH), exist_ok=True)
        default_negation = ['不', '没', '无', '非', '未', '别', '勿', '莫']
        with open(NEGATION_DICT_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(default_negation))
        logger.info(f"已创建否定词表: {NEGATION_DICT_PATH}")

create_default_dicts()
ensure_directories()

class SaveData:
    """统一数据保存工具类"""
    def __init__(self, data, result_type, add_some=None, filename=None, add_timestamp=True, keyword=None):
        self.data = data
        self.result_type = result_type
        self.add_some = add_some
        self.filename = filename
        self.add_timestamp = add_timestamp
        self.keyword = keyword

    def _add_some_(self, filepath):
        add = self.add_some
        dirname, filename = os.path.split(filepath)
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_{add}{ext}"
        return os.path.join(dirname, new_filename)

    def _add_timestamp_to_filename(self, filepath):
        dirname, filename = os.path.split(filepath)
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{name}_{timestamp}{ext}"
        return os.path.join(dirname, new_filename)

    def _get_full_path(self):
        result_type = self.result_type
        keyword = self.keyword
        add_some = self.add_some
        add_timestamp = self.add_timestamp

        if result_type == "videos":
            base_name = f"videos_{keyword}" if keyword else "videos"
            parts = [base_name]
            if add_timestamp:
                parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
            if add_some:
                parts.append(add_some)
            filename = "_".join(parts) + ".csv"
            if keyword:
                base_dir = os.path.join(RAW_DATA_DIR, keyword, "videos")
            else:
                base_dir = RAW_DATA_DIR
            full_path = os.path.join(base_dir, filename)

        elif result_type == "danmaku":
            base_name = f"danmaku_{keyword}" if keyword else "danmaku"
            parts = [base_name]
            if add_timestamp:
                parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
            if add_some:
                parts.append(add_some)
            filename = "_".join(parts) + ".csv"
            if keyword:
                base_dir = os.path.join(RAW_DATA_DIR, keyword, "danmaku")
            else:
                base_dir = RAW_DATA_DIR
            full_path = os.path.join(base_dir, filename)

        elif result_type == "processed":
            full_path = CLEANED_COMMENTS_PATH

        elif result_type == "sales":
            full_path = SALES_DATA_PATH

        elif result_type == "result":
            if not self.filename:
                raise ValueError("result_type 为 'result' 时必须提供 filename")
            full_path = os.path.join(RESULTS_PATH, self.filename)

        else:
            raise ValueError(f"不支持的结果类型: {result_type}")

        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        return full_path

    def save(self, **kwargs):
        data = self.data
        if isinstance(data, list):
            if not data:
                print("警告：数据为空，跳过保存")
                return
            data = pd.DataFrame(data)

        full_path = self._get_full_path()

        if isinstance(data, pd.DataFrame):
            data.to_csv(full_path, **kwargs)
        elif isinstance(data, str):
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(data)
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")

        relative_path = os.path.relpath(full_path, BASE_DIR)
        print(f"{self.result_type}数据已保存至：{relative_path}")