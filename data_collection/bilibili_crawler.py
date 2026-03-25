# data_collection/bilibili_crawler.py
# B站弹幕数据爬虫

import csv
import pandas
import os
import time
from datetime import datetime
from bilibili_api import video, sync, search
import config
import random


class BilibiliCrawler:
    """B站弹幕爬虫类"""

    def __init__(self, keywords, start_date, end_date):
        """
        :param keywords: 搜索关键词
        :param start_date: 开始日期
        :param end_date: 结束日期
        """
        self.keywords = keywords
        self.start_date = start_date
        self.end_date = end_date
        self.all_danmaku = []

    async def search_videos(self, keyword):
        """搜索相关视频"""
        print(f"正在搜索关键词: {keyword}")
        result = []

        resp = sync(search.search_by_type(
            keyword=keyword,
            search_type=search.SearchObjectType.VIDEO   #指定搜索视频类型
        ))


        # # 调试：打印第一条结果的结构（可选）
        # if resp.get('result') and len(resp['result']) > 0:
        #     print("搜索结果示例：", resp['result'][0])

        for item in resp.get('result', []):
            # 提取字段
            bvid = item.get('bvid')
            title = item.get('title')
            # 发布时间为时间戳（秒），转为日期字符串
            pubdate = datetime.fromtimestamp(item.get('pubdate', 0)).strftime('%Y-%m-%d')
            author = item.get('author')
            # 播放量、点赞数、弹幕数在顶层直接存在（从示例看）
            view = item.get('play', 0)          # 播放量字段为 'play'
            like = item.get('like', 0)          # 点赞数
            danmaku = item.get('danmaku', 0)    # 弹幕数（注意字段名小写）

            video_info = {
                'bv_id': bvid,
                'title': title,
                'pubdate': pubdate,
                'author': author,
                'view': view,
                'like': like,
                'danmaku': danmaku
            }
            result.append(video_info)
        return result

    def get_danmu(self, bv_id):
        """获取单个视频的弹幕"""
        try:
            v = video.Video(bvid=bv_id)
            # 获取弹幕列表（参数 0 表示第一页，实际可能有多页，此处简化）
            danmu_list = sync(v.get_danmus(0))  # 正确方法名

            danmu_data = []
            for d in danmu_list:
                # 动态获取弹幕文本（可能为 'text' 或 'content'）
                text = getattr(d, 'text', getattr(d, 'content', ''))
                # 动态获取视频内时间（可能为 'progress' 或 'time' 或 'timestamp'）
                time_val = None
                for attr in ['progress', 'time', 'timestamp']:
                    if hasattr(d, attr):
                        time_val = getattr(d, attr)
                        break
                # 弹幕发送日期
                if hasattr(d, 'date') and d.date:
                    date_str = d.date.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    date_str = ''
                # 弹幕类型
                dm_type = getattr(d, 'danmu_type', getattr(d, 'type', None))

                danmu_data.append({
                    'bv_id': bv_id,
                    'text': text,
                    'time': time_val,
                    'date': date_str,
                    'type': dm_type
                })
            return danmu_data
        except Exception as e:
            print(f"获取弹幕失败 {bv_id}: {e}")
            return []

    async def crawl(self):
        """执行爬虫"""
        # 遍历关键词
        for keyword in self.keywords:
            # 关键词查找
            videos = await self.search_videos(keyword)
            # 存储视频元数据
            config.save_data(videos,result_type="videos",result_count=len(videos))
            for video_info in videos:
                # 筛选时间范围
                if video_info['pubdate'] < self.start_date or video_info['pubdate'] > self.end_date:
                    continue
                # 筛选非广告
                elif video_info['bv_id'] == 0:
                    continue
                print(f"正在处理视频: {video_info['title']}\n")
                # 获取弹幕
                danmus = self.get_danmu(video_info['bv_id'])
                for d in danmus:
                    d.update(video_info)  # 合并视频元数据
                    self.all_danmaku.append(d)

                # 避免请求过快
                time.sleep(random.randint(1, 3))
            # 保存数据
            config.save_data(videos,result_type="videos",result_count=len(self.all_danmaku))




# 使用示例
if __name__ == "__main__":
    crawler = BilibiliCrawler(
        keywords=config.KEYWORDS,
        start_date=config.START_DATE,
        end_date=config.END_DATE
    )
    crawler.crawl()