"""Crawler package - iFixit guide fetching."""
from .ifixit_crawler import crawl_guide, crawl_guides, search_guides, DEVICE_KEYWORDS

__all__ = ["crawl_guide", "crawl_guides", "search_guides", "DEVICE_KEYWORDS"]
