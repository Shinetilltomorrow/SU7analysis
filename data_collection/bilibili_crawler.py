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

    def __init__(self, keywords, start_date, end_date):
        """
        :param keywords: 搜索关键词列表
        :param start_date: 开始日期（格式：YYYY-MM-DD）
        :param end_date: 结束日期（格式：YYYY-MM-DD）
        """
        self.keywords = keywords
        self.start_date = start_date
        self.end_date = end_date


    """
    ---------------------------
    爬取视频数据
    """

    async def search_videos(self, keyword, max_pages=10, max_videos=100):
        """
        异步搜索指定关键词的视频（仅元数据，不包含弹幕）。

        :param keyword: 搜索关键词
        :param max_pages: 最大翻页数（默认10）
        :param max_videos: 最多获取的视频数量（默认100）
        :return: 视频信息列表，每个元素为字典
        """
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
        仅采集视频元数据，并保存到指定路径。

        :param save_path: 保存路径（如果为 None，则使用 config.RAW_VIDEOS_PATH）
        :param keyword: 指定关键词（若为 None，则使用 self.keywords 中的所有关键词）
        :return: 所有采集到的视频列表
        """
        keywords_to_crawl = [keyword] if keyword else self.keywords
        all_videos = []

        for kw in keywords_to_crawl:
            videos = await self.search_videos(kw)
            # 根据日期范围筛选
            filtered = []
            for v in videos:
                if self.start_date <= v['pubdate'] <= self.end_date:
                    filtered.append(v)
            all_videos.extend(filtered)

        # 保存数据
        if save_path is None:
            save_path = config.RAW_VIDEOS_PATH
        config.save_data(all_videos, result_type="videos", add_some=None,
                         add_timestamp=True, keyword=keyword, filename=None)
        print(f"视频元数据已保存至 {save_path}")
        return all_videos

    """
    -------------------------
    爬取弹幕数据
    """
    async def get_all_danmaku(self, bvid):
        """
        获取单个视频所有分P的全部弹幕。

        :param bvid: 视频BV号
        :return: 弹幕列表，每个元素为字典
        """
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
        """
        根据视频ID列表采集弹幕，并保存到指定路径。

        :param video_ids: 视频BV号列表
        :param save_path: 保存路径（如果为 None，则使用 config.RAW_DANMAKU_PATH）
        :param keyword: 可选的关键词，用于在文件名中标识
        :return: 所有采集到的弹幕列表
        """
        all_danmaku = []
        total = len(video_ids)
        for idx, bvid in enumerate(video_ids, 1):
            print(f"进度：{idx}/{total}，正在获取视频 {bvid} 的弹幕...")
            danmus = await self.get_all_danmaku(bvid)
            all_danmaku.extend(danmus)
            await asyncio.sleep(random.uniform(1, 3))

        if save_path is None:
            save_path = config.RAW_DANMAKU_PATH
        config.save_data(all_danmaku, result_type="danmaku", add_some=None,
                         add_timestamp=True, keyword=keyword, filename=None)
        print(f"弹幕数据已保存至 {save_path}")
        return all_danmaku


# 测试函数（可选）
async def main():
    crawler = BilibiliCrawler(
        keywords=config.KEYWORDS,
        start_date=config.START_DATE,
        end_date=config.END_DATE
    )
    # 仅采集视频
    videos = await crawler.crawl_videos()
    # 假设从视频中提取 bv_id 列表
    video_ids = [v['bv_id'] for v in videos]
    # 采集弹幕
    danmakus = await crawler.crawl_danmaku(video_ids)

if __name__ == "__main__":
    asyncio.run(main())