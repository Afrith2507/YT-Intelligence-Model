from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape YouTube top-level comments into a CSV file."
    )
    parser.add_argument(
        "--channel-id",
        type=str,
        default=None,
        help="YouTube channel ID (preferred).",
    )
    parser.add_argument(
        "--channel-query",
        type=str,
        default="Ryan Trahan",
        help="Channel name search query if channel ID is not provided.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=30,
        help="Maximum number of videos to collect comments from (used in standard mode).",
    )
    parser.add_argument(
        "--max-comments-per-video",
        type=int,
        default=250,
        help="Maximum top-level comments to fetch per video (used in standard mode).",
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "single-ranked-video"],
        default="standard",
        help="standard: scrape multiple recent videos, single-ranked-video: scrape one most-viewed-ranked video.",
    )
    parser.add_argument(
        "--ranked-video-position",
        type=int,
        default=4,
        help="In single-ranked-video mode, choose Nth most-viewed video (1 = most viewed).",
    )
    parser.add_argument(
        "--scan-videos",
        type=int,
        default=400,
        help="In single-ranked-video mode, number of channel videos to scan before ranking by views.",
    )
    parser.add_argument(
        "--video-source",
        choices=["uploads-playlist", "search"],
        default="uploads-playlist",
        help="How to discover channel videos for ranking in single-ranked-video mode.",
    )
    parser.add_argument(
        "--single-video-comment-target",
        type=int,
        default=10000,
        help="In single-ranked-video mode, target number of comments from that single ranked video.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/youtube_comments_raw.csv",
        help="Output CSV path (absolute or relative to project root).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.1,
        help="Sleep between API requests to avoid aggressive rate bursts.",
    )
    parser.add_argument(
        "--exclude-ids-from",
        type=str,
        default=None,
        help="Path to existing CSV. Any comment_id already in this file will be skipped.",
    )
    parser.add_argument(
        "--video-order",
        choices=["date", "viewCount", "rating", "relevance"],
        default="date",
        help="Order to fetch videos from channel (standard mode only).",
    )
    parser.add_argument(
        "--target-new-comments",
        type=int,
        default=None,
        help="Stop collecting once this many NEW (non-excluded) comments are gathered.",
    )
    return parser.parse_args()


def build_youtube_client():
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing YOUTUBE_API_KEY. Add it to your .env file before running."
        )
    return build("youtube", "v3", developerKey=api_key)


def resolve_channel_id(youtube, channel_query: str) -> Optional[str]:
    response = (
        youtube.search()
        .list(
            part="snippet",
            q=channel_query,
            type="channel",
            maxResults=5,
        )
        .execute()
    )
    for item in response.get("items", []):
        channel_id = item.get("snippet", {}).get("channelId")
        if channel_id:
            return channel_id
    return None


def fetch_videos_for_scan(
    youtube,
    channel_id: str,
    max_videos: int,
    sleep_seconds: float,
    source: str = "uploads-playlist",
) -> List[Dict[str, str]]:
    if source == "uploads-playlist":
        return fetch_videos_from_uploads_playlist(
            youtube=youtube,
            channel_id=channel_id,
            max_videos=max_videos,
            sleep_seconds=sleep_seconds,
        )

    videos: List[Dict[str, str]] = []
    page_token = None

    while len(videos) < max_videos:
        response = (
            youtube.search()
            .list(
                part="id,snippet",
                channelId=channel_id,
                type="video",
                order=source if source not in ("uploads-playlist",) else "date",
                maxResults=min(50, max_videos - len(videos)),
                pageToken=page_token,
            )
            .execute()
        )

        for item in response.get("items", []):
            video_id = item.get("id", {}).get("videoId")
            snippet = item.get("snippet", {})
            if not video_id:
                continue
            videos.append(
                {
                    "video_id": video_id,
                    "video_title": snippet.get("title", ""),
                    "video_published_at": snippet.get("publishedAt", ""),
                }
            )
            if len(videos) >= max_videos:
                break

        page_token = response.get("nextPageToken")
        if not page_token:
            break

        time.sleep(sleep_seconds)

    return videos


def fetch_videos_from_uploads_playlist(
    youtube, channel_id: str, max_videos: int, sleep_seconds: float
) -> List[Dict[str, str]]:
    channel_response = (
        youtube.channels()
        .list(part="contentDetails", id=channel_id, maxResults=1)
        .execute()
    )
    items = channel_response.get("items", [])
    if not items:
        return []

    uploads_playlist_id = (
        items[0]
        .get("contentDetails", {})
        .get("relatedPlaylists", {})
        .get("uploads")
    )
    if not uploads_playlist_id:
        return []

    videos: List[Dict[str, str]] = []
    page_token = None
    seen_ids = set()

    while len(videos) < max_videos:
        response = (
            youtube.playlistItems()
            .list(
                part="contentDetails",
                playlistId=uploads_playlist_id,
                maxResults=min(50, max_videos - len(videos)),
                pageToken=page_token,
            )
            .execute()
        )

        for item in response.get("items", []):
            content = item.get("contentDetails", {})
            video_id = content.get("videoId")
            if not video_id or video_id in seen_ids:
                continue
            seen_ids.add(video_id)
            videos.append(
                {
                    "video_id": video_id,
                    "video_title": "",
                    "video_published_at": "",
                }
            )
            if len(videos) >= max_videos:
                break

        page_token = response.get("nextPageToken")
        if not page_token:
            break
        time.sleep(sleep_seconds)

    return videos


def attach_video_statistics(youtube, videos: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not videos:
        return []

    by_id: Dict[str, Dict[str, str]] = {v["video_id"]: v for v in videos}
    video_ids = list(by_id.keys())

    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i : i + 50]
        response = (
            youtube.videos()
            .list(part="statistics,snippet", id=",".join(batch_ids), maxResults=50)
            .execute()
        )

        for item in response.get("items", []):
            video_id = item.get("id")
            if not video_id or video_id not in by_id:
                continue

            stats = item.get("statistics", {})
            snippet = item.get("snippet", {})
            by_id[video_id]["view_count"] = int(stats.get("viewCount", 0))
            by_id[video_id]["comment_count"] = int(stats.get("commentCount", 0))
            if snippet.get("title"):
                by_id[video_id]["video_title"] = snippet.get("title", "")
            if snippet.get("publishedAt"):
                by_id[video_id]["video_published_at"] = snippet.get("publishedAt", "")

    enriched = list(by_id.values())
    for row in enriched:
        row.setdefault("view_count", 0)
        row.setdefault("comment_count", 0)
    return enriched


def fetch_channel_videos(
    youtube, channel_id: str, max_videos: int, sleep_seconds: float
) -> List[Dict[str, str]]:
    return fetch_videos_for_scan(
        youtube=youtube,
        channel_id=channel_id,
        max_videos=max_videos,
        sleep_seconds=sleep_seconds,
        source="uploads-playlist",
    )


def fetch_comments_for_video(
    youtube,
    video_meta: Dict[str, str],
    max_comments_per_video: int,
    sleep_seconds: float,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    page_token = None
    video_id = video_meta["video_id"]

    while len(rows) < max_comments_per_video:
        try:
            response = (
                youtube.commentThreads()
                .list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(100, max_comments_per_video - len(rows)),
                    textFormat="plainText",
                    pageToken=page_token,
                    order="time",
                )
                .execute()
            )
        except HttpError as err:
            status = getattr(err.resp, "status", None)
            if status in (403, 404):
                print(
                    f"Skipping video {video_id} due to comments restriction or unavailable comments.",
                    file=sys.stderr,
                )
                break
            raise

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            top_comment = snippet.get("topLevelComment", {}).get("snippet", {})
            comment_id = item.get("snippet", {}).get("topLevelComment", {}).get("id", "")
            text = top_comment.get("textDisplay") or top_comment.get("textOriginal") or ""

            if not comment_id or not text.strip():

                continue

            rows.append(
                {
                    "video_id": video_meta["video_id"],
                    "video_title": video_meta["video_title"],
                    "video_published_at": video_meta["video_published_at"],
                    "comment_id": comment_id,
                    "author": top_comment.get("authorDisplayName", ""),
                    "text": text,
                    "like_count": top_comment.get("likeCount", 0),
                    "published_at": top_comment.get("publishedAt", ""),
                    "updated_at": top_comment.get("updatedAt", ""),
                }
            )

            if len(rows) >= max_comments_per_video:
                break

        page_token = response.get("nextPageToken")
        if not page_token:
            break

        time.sleep(sleep_seconds)

    return rows


def load_existing_ids(path_str: str) -> set:
    path = Path(path_str)
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        path = project_root / path
    if not path.exists():
        print(f"Warning: --exclude-ids-from file not found at {path}. Skipping exclusion.")
        return set()
    df = pd.read_csv(path, usecols=lambda c: c in ("comment_id", "id"))
    id_col = "comment_id" if "comment_id" in df.columns else "id"
    ids = set(df[id_col].dropna().astype(str).tolist())
    print(f"Loaded {len(ids)} existing comment IDs to exclude from {path.name}.")
    return ids


def resolve_output_path(output_arg: str) -> Path:
    output_path = Path(output_arg)
    if output_path.is_absolute():
        return output_path  

    project_root = Path(__file__).resolve().parents[1]
    return project_root / output_path


def main() -> None:
    args = parse_args()
    youtube = build_youtube_client()

    channel_id = args.channel_id
    if not channel_id:
        channel_id = resolve_channel_id(youtube, args.channel_query)
        if not channel_id:
            raise RuntimeError(
                f"Could not resolve channel ID from query '{args.channel_query}'."
            )

    print(f"Using channel ID: {channel_id}")

    existing_ids: set = set()
    if args.exclude_ids_from:
        existing_ids = load_existing_ids(args.exclude_ids_from)

    all_rows: List[Dict[str, str]] = []

    if args.mode == "single-ranked-video":
        scanned_videos = fetch_videos_for_scan(
            youtube=youtube,
            channel_id=channel_id,
            max_videos=args.scan_videos,
            sleep_seconds=args.sleep_seconds,
            source=args.video_source,
        )
        print(f"Scanned {len(scanned_videos)} videos before ranking by views.")
        ranked = attach_video_statistics(youtube, scanned_videos)
        ranked = sorted(ranked, key=lambda x: int(x.get("view_count", 0)), reverse=True)

        if args.ranked_video_position < 1 or args.ranked_video_position > len(ranked):
            raise RuntimeError(
                f"ranked-video-position={args.ranked_video_position} is out of range "
                f"for {len(ranked)} scanned videos."
            )

        target_video = ranked[args.ranked_video_position - 1]
        print(
            "Target video:",
            f"rank={args.ranked_video_position},",
            f"title='{target_video['video_title']}',",
            f"views={target_video.get('view_count', 0)},",
            f"api_comment_count={target_video.get('comment_count', 0)},",
            f"video_id={target_video['video_id']}",
        )
        if int(target_video.get("comment_count", 0)) < args.single_video_comment_target:
            print(
                "Warning: target comments exceed the API-reported comment_count for this video. "
                "The scraper will collect all available comments, which may be less than the target."
            )

        all_rows = fetch_comments_for_video(
            youtube=youtube,
            video_meta=target_video,
            max_comments_per_video=args.single_video_comment_target,
            sleep_seconds=args.sleep_seconds,
        )
        print(
            f"Collected {len(all_rows)} comments from the selected ranked video "
            f"(target was {args.single_video_comment_target})."
        )
    else:
        videos = fetch_videos_for_scan(
            youtube=youtube,
            channel_id=channel_id,
            max_videos=args.max_videos,
            sleep_seconds=args.sleep_seconds,
            source=args.video_order,
        )
        print(f"Collected {len(videos)} videos.")

        target = args.target_new_comments
        new_count = 0

        for idx, video in enumerate(videos, start=1):
            if target and new_count >= target:
                print(f"Reached target of {target} new comments. Stopping early.")
                break
            print(f"[{idx}/{len(videos)}] Fetching: {video['video_title']}")
            rows = fetch_comments_for_video(
                youtube=youtube,
                video_meta=video,
                max_comments_per_video=args.max_comments_per_video,
                sleep_seconds=args.sleep_seconds,
            )
            new_rows = [r for r in rows if str(r["comment_id"]) not in existing_ids]
            all_rows.extend(new_rows)
            new_count += len(new_rows)
            print(f"  → {len(new_rows)} new comments (total new so far: {new_count})")

    if not all_rows:
        raise RuntimeError("No comments were collected. Check channel/videos/API quota.")

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].copy()
    df = df.drop_duplicates(subset=["comment_id"]).reset_index(drop=True)
    output_path = resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved {len(df)} unique comments to: {output_path}")


if __name__ == "__main__":
    main()
