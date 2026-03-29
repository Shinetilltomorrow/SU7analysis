from bilibili_api import search


resp = search.search_by_type(
                keyword="keyword",
                search_type=search.SearchObjectType.VIDEO,
                page=1
            )
items = resp.get('result', [])