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
USE_BERT = True
SENTIMENT_METHOD = 'bert'
BERT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bert-finetuned-3class")

# LDA 主题模型配置
N_TOPICS = None
AUTO_SELECT_TOPICS = True
N_TOP_WORDS = 10
USE_POS_FILTER = True
LDA_MAX_DF = 0.85
LDA_MIN_DF = 3

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

SEGMENTED_VIDEOS_PATH = os.path.join(PROCESSED_DATA_DIR, "segmented_videos.csv")
SEGMENTED_COMMENTS_PATH = os.path.join(PROCESSED_DATA_DIR, "segmented_comments.csv")
SALES_DATA_PATH = os.path.join(BASE_DIR, "data", "sales", "xiaomi_su7_sales.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "results")

COMBINED_VIDEOS_PATH = os.path.join(PROCESSED_DATA_DIR, "combined_cleaned_videos.csv")
COMBINED_COMMENTS_PATH = os.path.join(PROCESSED_DATA_DIR, "combined_cleaned_comments.csv")

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
        PROCESSED_DATA_DIR,
        os.path.join(PROCESSED_DATA_DIR, "videos"),
        os.path.join(PROCESSED_DATA_DIR, "danmaku"),
        os.path.dirname(STOPWORDS_PATH),
        os.path.dirname(USER_DICT_PATH),
        os.path.dirname(POS_DICT_PATH),
        os.path.dirname(BERT_MODEL_PATH),
    ]
    for kw in KEYWORDS:
        dirs.append(os.path.join(RAW_DATA_DIR, "videos", kw))
        dirs.append(os.path.join(RAW_DATA_DIR, "danmaku", kw))
        dirs.append(os.path.join(PROCESSED_DATA_DIR, "videos", kw))
        dirs.append(os.path.join(PROCESSED_DATA_DIR, "danmaku", kw))
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)

def create_default_dicts():
    # 停用词表（已扩充，去除英文停用词）
    if not os.path.exists(STOPWORDS_PATH):
        os.makedirs(os.path.dirname(STOPWORDS_PATH), exist_ok=True)
        default_stopwords = [
            '在', '和', '也', '都', '就', '不', '啊', '哦', '嗯', '吧', '吗', '呢', '呀',
            '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们', '它们', '自己', '别人',
            '这', '那', '这些', '那些', '这个', '那个', '这么', '那么', '这样', '那样',
            '什么', '怎么', '为什么', '哪里', '哪儿', '谁', '什么时候',
            '会', '能', '可以', '可能', '应该', '需要', '想要', '觉得', '认为', '感觉',
            '说', '看', '听', '走', '去', '来', '做', '用', '给', '让', '把', '被',
            '有', '没有', '是', '不是', '也', '还', '都', '只', '就', '才', '又', '再',
            '很', '太', '非常', '特别', '十分', '更', '最', '比较', '稍微', '有点',
            '因为', '所以', '但是', '然而', '虽然', '尽管', '如果', '那么', '否则',
            '哈', '嘻', '呵', '嘿', '哇', '哎', '哟', '喂', '嗯', '哦', '咩', '嚯', '呼', '啦', '咯',
            '看', '瞧', '望', '瞅', '瞄', '盯', '瞪', '瞥', '讲', '谈', '聊', '唠', '扯', '吹',
            '搞', '弄', '整', '干', '做', '办', '处理',
            '真的', '一个', '这个', '那个', '什么', '怎么', '为什么', '哪里', '哪儿', '谁', '什么时候',
            '觉得', '感觉', '看', '说', '听', '走', '去', '来', '做', '用', '给', '让', '把', '被',
            '有', '没有', '是', '不是', '也', '还', '都', '只', '就', '才', '又', '再',
            '很', '太', '非常', '特别', '十分', '更', '最', '比较', '稍微', '有点',
            '视频', '看了', '懂了', '笑了', '哭了', '刷', '打卡', '路过', '收藏', '点赞', '投币', '三连', '关注',
            '转发', '支持',
            '666', '233', '哈哈哈', '哈哈哈哈'
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

    # 积极词（扩充汽车领域和网络用语）
    if not os.path.exists(POS_DICT_PATH):
        os.makedirs(os.path.dirname(POS_DICT_PATH), exist_ok=True)
        default_pos = [
            '好', '棒', '赞', '优秀', '厉害', '惊艳', '完美', '喜欢', '爱', '值得', '推荐', '满意', '惊喜',
            '流畅', '稳定', '可靠', '省心', '划算', '超值', 'yyds', '真香', '舒服', '爽', '牛逼', '给力',
            '酷', '帅', '漂亮', '好看', '美观', '大气', '高端', '豪华', '舒适', '安静', '平顺', '强劲',
            '快', '迅速', '灵敏', '精准', '智能', '科技', '安全', '放心', '省油', '省钱', '经济', '环保',
            '先进', '创新', '独特', '新颖', '可靠', '耐用', '保值', '有面子', '有档次', '有品位', '有范儿',
            '拉风', '吸睛',
            '丝滑', '扎实', '迅猛', '爆', '顶', '绝', '香', '种草', '安利', '惊艳', '可靠', '耐用', '省心',
            '静谧', '稳健', '从容', '干脆', '利落', '聪明', '智慧', '贴心', '人性化', '便捷', '炫酷', '时尚',
            '精致', '科技感', '未来感', '良心', '厚道', '雷总', '大定', '爆单', '破万', '遥遥领先',
            '绝绝子', 'yyds', 'awsl', '芜湖', '起飞', '冲', '买爆', '真香', '上头', '爱了爱了', '666', '牛批'
        ]
        with open(POS_DICT_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(default_pos))
        logger.info(f"已创建积极词表（{len(default_pos)}词）: {POS_DICT_PATH}")

    # 消极词（扩充汽车领域和网络用语）
    if not os.path.exists(NEG_DICT_PATH):
        os.makedirs(os.path.dirname(NEG_DICT_PATH), exist_ok=True)
        default_neg = [
            '差', '烂', '垃圾', '失望', '后悔', '坑', '问题', '故障', '异响', '漏水', '卡顿', '慢', '贵',
            '不值', '劝退', '踩雷', '虚标', '缩水', '延迟', '不好', '不行', '糟糕', '恶心', '烦', '蠢',
            '傻', '笨', '差劲', '劣质', '粗糙', '简陋', '难用', '难开', '难坐', '不舒服', '不舒适',
            '不安静', '噪音大', '颠簸', '顿挫', '顿挫感', '续航短', '充电慢', '服务差', '售后差', '欺诈',
            '虚假', '夸大', '忽悠', '坑人', '骗人', '不值', '太贵', '溢价', '加价', '提车慢', '等待久',
            '交付延迟',
            '肉', '顿挫', '颠', '抖', '散', '虚', '异响', '漏水', '生锈', '故障', '失灵', '出问题', '趴窝',
            '死机', '黑屏', '卡死', '智障', '不灵敏', '反应慢', '交付慢', '等车久', '售后差', '服务差',
            '加价', '变相加价', '黄牛', '加价提车', '减配', '偷工减料', '虚标', '缩水', '续航虚', '充电慢',
            '慢充', '充电桩少', '噪音', '风噪', '胎噪', '异响', '共振', '颠簸', '悬挂硬', '刹车软', '转向虚',
            '指向不准', '电耗高', '费电', '掉电快', '续航焦虑', '自动驾驶垃圾', '智驾拉胯',
            '翻车', '踩雷', '劝退', '拔草', '避坑', '辣鸡', '渣渣', '吐了', '雷', '坑爹', '智商税', '韭菜'
        ]
        with open(NEG_DICT_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(default_neg))
        logger.info(f"已创建消极词表（{len(default_neg)}词）: {NEG_DICT_PATH}")

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
            if not keyword:
                raise ValueError("保存 videos 时必须提供 keyword")
            base_dir = os.path.join(RAW_DATA_DIR, "videos", keyword)
            base_name = f"videos_{keyword}"
            parts = [base_name]
            if add_timestamp:
                parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
            if add_some:
                parts.append(add_some)
            filename = "_".join(parts) + ".csv"
            full_path = os.path.join(base_dir, filename)

        elif result_type == "danmaku":
            if not keyword:
                raise ValueError("保存 danmaku 时必须提供 keyword")
            base_dir = os.path.join(RAW_DATA_DIR, "danmaku", keyword)
            base_name = f"danmaku_{keyword}"
            parts = [base_name]
            if add_timestamp:
                parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
            if add_some:
                parts.append(add_some)
            filename = "_".join(parts) + ".csv"
            full_path = os.path.join(base_dir, filename)

        elif result_type == "processed":
            full_path = COMBINED_COMMENTS_PATH

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