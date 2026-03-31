# data_collection/bilibili_crawler.py
import asyncio
import math
import random
from datetime import datetime
from bilibili_api import video, search
import config


class BilibiliCrawler:
    def __init__(self, keywords, start_date, end_date):
        self.keywords = keywords
        self.start_date = start_date
        self.end_date = end_date

    async def search_videos(self, keyword, max_pages=10, max_videos=100):
        config.logger.info(f"正在搜索关键词: {keyword}")
        all_videos = []
        page = 1
        total_fetched = 0

        while page <= max_pages and total_fetched < max_videos:
            try:
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
            except Exception as e:
                config.logger.error(f"搜索关键词 '{keyword}' 第{page}页失败: {e}")
                break

        config.logger.info(f"关键词 '{keyword}' 共获取 {len(all_videos)} 个视频")
        return all_videos

    async def crawl_videos(self, keyword=None):
        keywords_to_crawl = [keyword] if keyword else self.keywords
        all_videos = []

        for kw in keywords_to_crawl:
            print(f"\n{'='*60}")
            print(f"▶ 开始采集关键词: {kw}")
            print(f"{'='*60}")

            videos = await self.search_videos(kw)
            filtered = [v for v in videos if self.start_date <= v['pubdate'] <= self.end_date]
            if filtered:
                config.SaveData(
                    filtered,
                    result_type="videos",
                    add_some=kw,
                    add_timestamp=True,
                    keyword=kw
                ).save()
                all_videos.extend(filtered)
            else:
                config.logger.info(f"关键词 '{kw}' 未采集到有效视频")

            print(f"\n{'='*60}")
            print(f"✔ 关键词 '{kw}' 采集完成（共 {len(filtered)} 个有效视频）")
            print(f"{'='*60}\n")

        return all_videos

    async def get_all_danmaku(self, bvid):
        try:
            v = video.Video(bvid=bvid)
            info = await v.get_info()
            pages = info.get('pages', [])
            all_danmaku = []

            for page_idx, page_info in enumerate(pages):
                cid = page_info['cid']
                duration = page_info['duration']
                seg_count = math.ceil(duration / 360)

                config.logger.debug(f"获取 {bvid} 分P {page_idx+1}/{len(pages)}，共 {seg_count} 段弹幕")
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
        except Exception as e:
            config.logger.error(f"获取视频 {bvid} 弹幕失败: {e}")
            return []

    async def crawl_danmaku(self, video_ids, keyword=None):
        all_danmaku = []
        semaphore = asyncio.Semaphore(5)

        if keyword:
            print(f"\n{'='*60}")
            print(f"▶ 开始采集关键词 '{keyword}' 的弹幕")
            print(f"{'='*60}")

        async def fetch_one(bvid):
            async with semaphore:
                config.logger.info(f"获取视频 {bvid} 的弹幕...")
                return await self.get_all_danmaku(bvid)

        tasks = [fetch_one(bvid) for bvid in video_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                config.logger.error(f"视频 {video_ids[idx]} 弹幕获取失败: {res}")
            else:
                all_danmaku.extend(res)

        config.logger.info(f"共采集到 {len(all_danmaku)} 条弹幕")
        config.SaveData(
            all_danmaku,
            result_type="danmaku",
            add_some=keyword,
            add_timestamp=True,
            keyword=keyword
        ).save()

        if keyword:
            print(f"\n{'='*60}")
            print(f"✔ 关键词 '{keyword}' 弹幕采集完成（共 {len(all_danmaku)} 条）")
            print(f"{'='*60}\n")

        return all_danmaku