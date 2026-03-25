# data_collection/bilibili_crawler.py
# B站弹幕数据爬虫

import asyncio
import csv
import pandas
import os
import random
from datetime import datetime
from bilibili_api import video, search
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

    async def search_videos(self, keyword, max_pages=10, max_videos=100):
        """
        搜索相关视频
        :param keyword: 搜索关键词
        :param max_pages: 最大翻页数（防止无限循环）
        :param max_videos: 最大获取视频数
        """
        print("=" * 50)
        print(f"正在搜索关键词: {keyword}")
        print("=" * 50+ "\n")
        all_videos = []
        page = 1
        total_fetched = 0

        while page <= max_pages and total_fetched < max_videos:
            resp = await search.search_by_type(
                keyword=keyword,
                search_type=search.SearchObjectType.VIDEO,
                page=page
            )

            items = resp.get('result', [])
            if not items:
                break  # 没有更多结果

            for item in items:
                bvid = item.get('bvid')
                title = item.get('title')
                pubdate = datetime.fromtimestamp(item.get('pubdate', 0)).strftime('%Y-%m-%d')
                author = item.get('author')
                view = item.get('play', 0)
                like = item.get('like', 0)
                danmaku = item.get('danmaku', 0)

                video_info = {
                    'bv_id': bvid,
                    'title': title,
                    'pubdate': pubdate,
                    'author': author,
                    'view': view,
                    'like': like,
                    'danmaku': danmaku
                }
                all_videos.append(video_info)

            total_fetched += len(items)
            page += 1
            # 避免请求过快，适当延时
            await asyncio.sleep(random.randint(1, 3))

        print(f"关键词 '{keyword}' 共获取 {len(all_videos)} 个视频")
        return all_videos


    async def get_danmu(self, bv_id):
        """获取单个视频的弹幕"""
        try:
            v = video.Video(bvid=bv_id)
            danmu_list = await v.get_danmakus()

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
                # 添加弹幕数据
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
            config.save_data(videos,keyword=keyword,result_type="videos",add_some=f"{keyword}")
            print("视频元数据已收集，正在开始获取视频弹幕数据\n")
            print("-" * 50)
            for video_info in videos:
                # 筛选时间范围
                if video_info['pubdate'] < self.start_date or video_info['pubdate'] > self.end_date:
                    continue
                if video_info['bv_id'] == 0:
                    continue
                print(f"正在处理视频: {video_info['title']}")
                # 获取弹幕
                danmus = await self.get_danmu(video_info['bv_id'])
                for d in danmus:
                    d.update(video_info)
                    self.all_danmaku.append(d)
                # 随机时停
                await asyncio.sleep(random.randint(1, 3))
            print("-" * 50)
            print("弹幕数据处理完毕")
            print("-" * 50)
                # 保存数据
            config.save_data(self.all_danmaku,keyword=keyword, result_type="danmaku",add_some=f"{keyword}")


# 测试函数
async def main():
    crawler = BilibiliCrawler(
        keywords=config.KEYWORDS,
        start_date=config.START_DATE,
        end_date=config.END_DATE
    )
    await crawler.crawl()

if __name__ == "__main__":
    asyncio.run(main())