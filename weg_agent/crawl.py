import argparse
from pathlib import Path
from .crawler.ifixit_crawler import crawl, DEVICE_KEYWORDS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", required=True, choices=DEVICE_KEYWORDS.keys())
    p.add_argument("--keyword", required=False)
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--out", default="preweg_data")
    args = p.parse_args()

    keyword = args.keyword or DEVICE_KEYWORDS[args.device]
    crawl(Path(args.out), args.device, keyword, args.limit)


if __name__ == "__main__":
    main()
