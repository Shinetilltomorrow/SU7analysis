# data_collection/bilibili_crawler.py
# B站数据爬虫（拆分为视频元数据采集和弹幕采集两部分）

import asyncio
import math
import random
from datetime import datetime
from bilibili_api import video, search
import config


class BilibiliCrawler:
    """B站数据爬虫类（支持分离采集视频元数据和弹幕）"""

    def __init__(self, keywords, start_date, end_date, save_path=None):
        self.keywords = keywords
        self.start_date = start_date
        self.end_date = end_date
        self.save_path = save_path

    async def search_videos(self, keyword, max_pages=10, max_videos=100):
        """异步搜索指定关键词的视频"""
        print(f"正在搜索关键词: {keyword}")
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
                break

            for item in items:
                bvid = item.get('bvid')
                title = item.get('title')
                pubdate = datetime.fromtimestamp(item.get('pubdate', 0)).strftime('%Y-%m-%d')
                author = item.get('author')
                duration = item.get('duration')
                reply = item.get('reply', 0)
                view = item.get('play', 0)
                like = item.get('like', 0)
                danmaku = item.get('danmaku', 0)

                video_info = {
                    'bv_id': bvid,
                    'title': title,
                    'pubdate': pubdate,
                    'author': author,
                    'duration': duration,
                    'reply': reply,
                    'view': view,
                    'like': like,
                    'danmaku': danmaku
                }
                all_videos.append(video_info)

            total_fetched += len(items)
            page += 1
            await asyncio.sleep(random.uniform(1, 3))

        print(f"关键词 '{keyword}' 共获取 {len(all_videos)} 个视频")
        return all_videos

    async def crawl_videos(self, save_path=None, keyword=None):
        """
        采集视频元数据，若 keyword 为 None 则采集所有关键词并分别保存。
        """
        keywords_to_crawl = [keyword] if keyword else self.keywords
        all_videos = []

        for kw in keywords_to_crawl:
            videos = await self.search_videos(kw)
            # 日期筛选
            filtered = [v for v in videos if self.start_date <= v['pubdate'] <= self.end_date]
            if filtered:
                # 按关键词分别保存
                config.SaveData(
                    filtered,
                    result_type="videos",
                    add_some=kw,
                    add_timestamp=True,
                    keyword=kw
                ).save()
                all_videos.extend(filtered)
            else:
                print(f"关键词 '{kw}' 未采集到有效视频")

        return all_videos

    async def get_all_danmaku(self, bvid):
        """获取单个视频所有分P的全部弹幕"""
        v = video.Video(bvid=bvid)
        info = await v.get_info()
        pages = info.get('pages', [])
        all_danmaku = []

        for page_idx, page_info in enumerate(pages):
            cid = page_info['cid']
            duration = page_info['duration']
            seg_count = math.ceil(duration / 360)

            print(f"正在获取分P {page_idx + 1}/{len(pages)}，共 {seg_count} 段弹幕...")
            for seg in range(seg_count):
                danmu_list = await v.get_danmakus(
                    cid=cid,
                    date=None,
                    from_seg=seg,
                    to_seg=seg
                )
                for d in danmu_list:
                    danmu_type_map = {1: '滚动', 4: '底部', 5: '顶部', 6: '逆向', 7: '高级'}
                    all_danmaku.append({
                        'bv_id': bvid,
                        'page': page_idx + 1,
                        'cid': cid,
                        'text': d.text,
                        'time': d.dm_time,
                        'date': datetime.fromtimestamp(d.send_time).strftime('%Y-%m-%d %H:%M:%S') if d.send_time else '',
                        'type': danmu_type_map.get(d.mode, str(d.mode))
                    })
                await asyncio.sleep(0.5)

        return all_danmaku

    async def crawl_danmaku(self, video_ids, save_path=None, keyword=None):
        """根据视频ID列表采集弹幕，并保存到指定路径"""
        all_danmaku = []
        total = len(video_ids)
        for idx, bvid in enumerate(video_ids, 1):
            print(f"进度：{idx}/{total}，正在获取视频 {bvid} 的弹幕...")
            danmus = await self.get_all_danmaku(bvid)
            all_danmaku.extend(danmus)
            await asyncio.sleep(random.uniform(1, 3))

        # 保存弹幕
        config.SaveData(
            all_danmaku,
            result_type="danmaku",
            add_some=keyword,
            add_timestamp=True,
            keyword=keyword
        ).save()
        return all_danmaku