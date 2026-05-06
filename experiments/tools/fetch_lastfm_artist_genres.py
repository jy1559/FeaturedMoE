#!/usr/bin/env python3
"""Fetch genre tags for LastFM artist MBIDs via Last.fm API and group into ~32 canonical genres.

Usage:
  # Step 1: fetch raw tags (writes cache JSON)
  python fetch_lastfm_artist_genres.py fetch --api-key YOUR_KEY --item-file path/to/lastfm.item --out-cache artist_tags.json

  # Step 2: map tags → canonical genres (writes artist_genre.tsv)
  python fetch_lastfm_artist_genres.py map --cache artist_tags.json --out artist_genre.tsv

  # Step 3 (optional): review unmapped tags
  python fetch_lastfm_artist_genres.py review --cache artist_tags.json

Canonical genre set matches lastfm0.03 labels (32 genres).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ITEM_FILE = REPO_ROOT / "Datasets" / "processed" / "basic" / "lastfm" / "lastfm.item"
DEFAULT_CACHE     = REPO_ROOT / "Datasets" / "raw" / "lastfm-dataset-1K" / "artist_tags_cache.json"
DEFAULT_OUT_TSV   = REPO_ROOT / "Datasets" / "raw" / "lastfm-dataset-1K" / "artist_genre.tsv"

LASTFM_API_URL = "http://ws.audioscrobbler.com/2.0/"

# Canonical genre set (32 labels matching lastfm0.03)
CANONICAL_GENRES = [
    "Alternative/Indie",
    "Blues",
    "Classic/Hard Rock",
    "Classical",
    "Country",
    "Country/Americana",
    "Dance/Club",
    "Disco",
    "Electronic",
    "Electronica/Downtempo",
    "Experimental",
    "Folk",
    "Folk/Acoustic",
    "Funk",
    "Hip-Hop/Rap",
    "Industrial",
    "Jazz",
    "Jazz Fusion",
    "K-Pop/J-Pop",
    "Latin",
    "Metal",
    "Others",
    "Pop",
    "Punk/Emo",
    "R&B",
    "R&B/Soul",
    "Reggae/Ska",
    "Rock",
    "Soundtrack",
    "Synthpop/New Wave",
    "Unknown",
    "World/Ethnic",
]

# Tag → canonical genre mapping (lowercase tag → genre label)
# Ordered by priority: more specific patterns first within groups.
TAG_TO_GENRE: List[tuple[str, str]] = [
    # Classical
    ("classical", "Classical"),
    ("opera", "Classical"),
    ("orchestral", "Classical"),
    ("chamber music", "Classical"),
    ("baroque", "Classical"),
    ("contemporary classical", "Classical"),

    # Jazz
    ("jazz fusion", "Jazz Fusion"),
    ("jazz", "Jazz"),
    ("bebop", "Jazz"),
    ("swing", "Jazz"),
    ("smooth jazz", "Jazz"),
    ("free jazz", "Jazz"),

    # Blues
    ("blues", "Blues"),
    ("chicago blues", "Blues"),
    ("delta blues", "Blues"),

    # Country/Americana
    ("americana", "Country/Americana"),
    ("bluegrass", "Country/Americana"),
    ("country", "Country"),
    ("country pop", "Country"),

    # Folk/Acoustic
    ("folk rock", "Folk/Acoustic"),
    ("acoustic", "Folk/Acoustic"),
    ("singer-songwriter", "Folk/Acoustic"),
    ("folk", "Folk"),

    # Electronic subtypes (before general Electronic)
    ("electronica", "Electronica/Downtempo"),
    ("downtempo", "Electronica/Downtempo"),
    ("ambient", "Electronica/Downtempo"),
    ("chillout", "Electronica/Downtempo"),
    ("chill", "Electronica/Downtempo"),
    ("trip-hop", "Electronica/Downtempo"),
    ("trip hop", "Electronica/Downtempo"),
    ("synthpop", "Synthpop/New Wave"),
    ("synth-pop", "Synthpop/New Wave"),
    ("new wave", "Synthpop/New Wave"),
    ("darkwave", "Synthpop/New Wave"),
    ("industrial", "Industrial"),
    ("noise", "Industrial"),
    ("ebm", "Industrial"),
    ("dance", "Dance/Club"),
    ("club", "Dance/Club"),
    ("house", "Dance/Club"),
    ("techno", "Dance/Club"),
    ("trance", "Dance/Club"),
    ("edm", "Dance/Club"),
    ("rave", "Dance/Club"),
    ("drum and bass", "Electronic"),
    ("drum'n'bass", "Electronic"),
    ("dubstep", "Electronic"),
    ("electronic", "Electronic"),
    ("electro", "Electronic"),
    ("electroclash", "Electronic"),
    ("disco", "Disco"),

    # Hip-Hop/Rap
    ("hip-hop", "Hip-Hop/Rap"),
    ("hip hop", "Hip-Hop/Rap"),
    ("rap", "Hip-Hop/Rap"),
    ("trap", "Hip-Hop/Rap"),
    ("gangsta rap", "Hip-Hop/Rap"),
    ("r&b", "R&B/Soul"),
    ("rnb", "R&B/Soul"),
    ("soul", "R&B/Soul"),
    ("neo soul", "R&B/Soul"),
    ("funk", "Funk"),

    # Rock subtypes (before general Rock)
    ("punk", "Punk/Emo"),
    ("emo", "Punk/Emo"),
    ("post-punk", "Punk/Emo"),
    ("hardcore", "Punk/Emo"),
    ("heavy metal", "Metal"),
    ("death metal", "Metal"),
    ("black metal", "Metal"),
    ("thrash metal", "Metal"),
    ("power metal", "Metal"),
    ("doom metal", "Metal"),
    ("metal", "Metal"),
    ("alternative rock", "Alternative/Indie"),
    ("indie rock", "Alternative/Indie"),
    ("indie pop", "Alternative/Indie"),
    ("alternative", "Alternative/Indie"),
    ("indie", "Alternative/Indie"),
    ("hard rock", "Classic/Hard Rock"),
    ("classic rock", "Classic/Hard Rock"),
    ("progressive rock", "Rock"),
    ("prog rock", "Rock"),
    ("psychedelic", "Rock"),
    ("grunge", "Rock"),
    ("rock", "Rock"),

    # Pop
    ("pop", "Pop"),
    ("k-pop", "K-Pop/J-Pop"),
    ("j-pop", "K-Pop/J-Pop"),
    ("k-rock", "K-Pop/J-Pop"),
    ("j-rock", "K-Pop/J-Pop"),
    ("korean", "K-Pop/J-Pop"),
    ("japanese", "K-Pop/J-Pop"),

    # Reggae/Ska
    ("reggae", "Reggae/Ska"),
    ("ska", "Reggae/Ska"),
    ("dancehall", "Reggae/Ska"),

    # Latin
    ("latin", "Latin"),
    ("salsa", "Latin"),
    ("bossa nova", "Latin"),
    ("samba", "Latin"),
    ("flamenco", "Latin"),

    # World/Ethnic
    ("world music", "World/Ethnic"),
    ("world", "World/Ethnic"),
    ("ethnic", "World/Ethnic"),
    ("celtic", "World/Ethnic"),
    ("african", "World/Ethnic"),
    ("middle eastern", "World/Ethnic"),
    ("indian", "World/Ethnic"),
    ("chanson", "World/Ethnic"),
    ("ye-ye", "World/Ethnic"),
    ("french", "World/Ethnic"),
    ("bossa", "World/Ethnic"),
    ("turkish", "World/Ethnic"),
    ("deutsch", "World/Ethnic"),

    # Experimental
    ("experimental", "Experimental"),
    ("avant-garde", "Experimental"),
    ("noise rock", "Experimental"),
    ("breakbeat", "Experimental"),
    ("mashup", "Experimental"),
    ("mash-up", "Experimental"),

    # Soundtrack
    ("soundtrack", "Soundtrack"),
    ("film score", "Soundtrack"),
    ("video game music", "Soundtrack"),
    ("anime", "Soundtrack"),
    ("audiobook", "Soundtrack"),
    ("spoken word", "Soundtrack"),

    # Blues (additional)
    ("rockabilly", "Blues"),

    # Electronica/Downtempo (additional)
    ("lounge", "Electronica/Downtempo"),
    ("eurobeat", "Dance/Club"),
    ("instrumental", "Electronica/Downtempo"),

    # Others (non-music / meta tags)
    ("comedy", "Others"),
    ("humor", "Others"),
    ("funny", "Others"),
    ("stand-up", "Others"),
]


def read_artist_ids_from_item(item_file: Path) -> List[str]:
    """Read unique artist UUIDs (category column) from .item file."""
    artist_ids: set[str] = set()
    with item_file.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            cat = (row.get("category:token") or row.get("category") or "").strip()
            if cat:
                artist_ids.add(cat)
    return sorted(artist_ids)


def fetch_top_tags(mbid: str, api_key: str, retries: int = 3) -> List[str]:
    """Return list of tag names for artist MBID (lowercased)."""
    params = urllib.parse.urlencode({
        "method": "artist.gettoptags",
        "mbid": mbid,
        "api_key": api_key,
        "format": "json",
        "autocorrect": "1",
    })
    url = f"{LASTFM_API_URL}?{params}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            toptags = data.get("toptags", {})
            tags = toptags.get("tag", [])
            if isinstance(tags, dict):
                tags = [tags]
            return [t["name"].lower() for t in tags if isinstance(t, dict) and "name" in t]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []
            time.sleep(1.0 * (attempt + 1))
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return []


def _tag_matches(pattern: str, tag: str) -> bool:
    """Check if pattern matches tag, not preceded/followed by alphanumeric or hyphen chars."""
    escaped = re.escape(pattern)
    return bool(re.search(r"(?<![a-z0-9\-])" + escaped + r"(?![a-z0-9\-])", tag))


def map_tags_to_genre(tags: List[str]) -> str:
    """Map a list of tags to canonical genre using vote-across-all-tags strategy.

    Each tag that matches a pattern casts a vote for that genre. The genre with
    the most votes wins. Ties broken by order of first match. Falls back to
    'Unknown' if no tag matches any pattern.
    """
    from collections import Counter
    votes: Counter = Counter()
    first_match_rank: dict = {}  # genre -> index of first matching tag (lower = earlier)
    for tag_idx, tag in enumerate(tags):
        tag_lower = tag.lower().strip()
        for pattern, genre in TAG_TO_GENRE:
            if _tag_matches(pattern, tag_lower):
                votes[genre] += 1
                if genre not in first_match_rank:
                    first_match_rank[genre] = tag_idx
                break  # one pattern match per tag
    if not votes:
        return "Unknown"
    # pick genre with most votes; break ties by earliest first match
    best = max(votes, key=lambda g: (votes[g], -first_match_rank[g]))
    return best


def cmd_fetch(args: argparse.Namespace) -> None:
    item_file = Path(args.item_file)
    cache_path = Path(args.cache)
    api_key = args.api_key

    artist_ids = read_artist_ids_from_item(item_file)
    print(f"Unique artist MBIDs: {len(artist_ids):,}")

    cache: Dict[str, List[str]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"  loaded {len(cache):,} cached entries")

    to_fetch = [aid for aid in artist_ids if aid not in cache]
    print(f"  fetching {len(to_fetch):,} new entries (rate ~5/s)")

    batch_save = 500
    for i, mbid in enumerate(to_fetch):
        tags = fetch_top_tags(mbid, api_key)
        cache[mbid] = tags
        if (i + 1) % batch_save == 0:
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  [{i+1}/{len(to_fetch)}] saved cache ({len(cache):,} total)")
        time.sleep(0.2)  # ~5 req/s — well under Last.fm 5/s limit

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done → {cache_path}  total={len(cache):,}")


def cmd_map(args: argparse.Namespace) -> None:
    cache_path = Path(args.cache)
    out_path = Path(args.out)

    cache: Dict[str, List[str]] = json.loads(cache_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(cache):,} entries from cache")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    from collections import Counter
    genre_counts: Counter = Counter()
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["artist_mbid", "genre"])
        for mbid, tags in sorted(cache.items()):
            genre = map_tags_to_genre(tags)
            genre_counts[genre] += 1
            writer.writerow([mbid, genre])

    print(f"Written: {out_path}")
    print(f"\nGenre distribution ({len(cache):,} artists):")
    for genre, cnt in sorted(genre_counts.items(), key=lambda x: -x[1]):
        print(f"  {cnt:6,}  {genre}")


def cmd_review(args: argparse.Namespace) -> None:
    """Print tags that map to 'Unknown' or 'Others' — useful for expanding TAG_TO_GENRE."""
    from collections import Counter
    cache_path = Path(args.cache)
    cache: Dict[str, List[str]] = json.loads(cache_path.read_text(encoding="utf-8"))

    unknown_tag_counts: Counter = Counter()
    for tags in cache.values():
        genre = map_tags_to_genre(tags)
        if genre in ("Unknown", "Others"):
            for t in tags[:5]:
                unknown_tag_counts[t.lower()] += 1

    print(f"Top unmapped tags ({len(unknown_tag_counts):,} unique):")
    for tag, cnt in unknown_tag_counts.most_common(50):
        print(f"  {cnt:5,}  {tag}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("fetch", help="Fetch raw tags from Last.fm API")
    pf.add_argument("--api-key", required=True, help="Last.fm API key")
    pf.add_argument("--item-file", type=Path, default=DEFAULT_ITEM_FILE)
    pf.add_argument("--cache", type=Path, default=DEFAULT_CACHE)

    pm = sub.add_parser("map", help="Map cached tags to canonical genres")
    pm.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    pm.add_argument("--out", type=Path, default=DEFAULT_OUT_TSV)

    pr = sub.add_parser("review", help="Review unmapped tags")
    pr.add_argument("--cache", type=Path, default=DEFAULT_CACHE)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "map":
        cmd_map(args)
    elif args.command == "review":
        cmd_review(args)


if __name__ == "__main__":
    main()
