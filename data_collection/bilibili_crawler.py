# B站弹幕爬虫
# data_collection/bilibili_crawler.py
# B站弹幕数据爬虫

import csv
import time
from datetime import datetime
from bilibili_api import video, sync, search
import config


class BilibiliCrawler:
    """B站弹幕爬虫类"""

    def __init__(self, keywords, start_date, end_date):
        self.keywords = keywords
        self.start_date = start_date
        self.end_date = end_date
        self.all_danmaku = []

    def search_videos(self, keyword):
        """搜索相关视频"""
        print(f"正在搜索关键词: {keyword}")
        result = []
        # 执行搜索
        resp = search.search_by_type(keyword, search_type=search.SearchObjectType.VIDEO)
        # 这里简化处理，实际需要处理分页和筛选
        for item in resp['result']:
            video_info = {
                'bv_id': item['bvid'],
                'title': item['title'],
                'pubdate': datetime.fromtimestamp(item['pubdate']).strftime('%Y-%m-%d'),
                'author': item['author'],
                'view': item['view'],
                'like': item['like'],
                'danmaku': item['danmaku']
            }
            result.append(video_info)
        return result

    def get_danmaku(self, bv_id):
        """获取单个视频的弹幕"""
        try:
            v = video.Video(bvid=bv_id)
            # 获取弹幕列表（需要处理分页）
            danmaku_list = sync(v.get_danmakus(0))

            danmaku_data = []
            for d in danmaku_list:
                danmaku_data.append({
                    'bv_id': bv_id,
                    'text': d.text,
                    'time': d.progress,  # 视频内时间戳（毫秒）
                    'date': d.date.strftime('%Y-%m-%d %H:%M:%S'),
                    'type': d.danmaku_type
                })
            return danmaku_data
        except Exception as e:
            print(f"获取弹幕失败 {bv_id}: {e}")
            return []

    def crawl(self):
        """执行爬虫"""
        for keyword in self.keywords:
            videos = self.search_videos(keyword)
            for video_info in videos:
                # 筛选时间范围
                if video_info['pubdate'] < self.start_date or video_info['pubdate'] > self.end_date:
                    continue

                print(f"正在处理视频: {video_info['title']}")
                danmakus = self.get_danmaku(video_info['bv_id'])
                for d in danmakus:
                    d.update(video_info)  # 合并视频元数据
                    self.all_danmaku.append(d)

                # 避免请求过快
                time.sleep(1)

        # 保存数据
        self.save_data()

    def save_data(self):
        """保存数据到CSV"""
        if not self.all_danmaku:
            print("没有采集到数据")
            return

        keys = self.all_danmaku[0].keys()
        with open(config.RAW_DATA_PATH, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.all_danmaku)

        print(f"数据已保存到 {config.RAW_DATA_PATH}，共 {len(self.all_danmaku)} 条弹幕")


# 使用示例
if __name__ == "__main__":
    crawler = BilibiliCrawler(
        keywords=config.KEYWORDS,
        start_date=config.START_DATE,
        end_date=config.END_DATE
    )
    crawler.crawl()