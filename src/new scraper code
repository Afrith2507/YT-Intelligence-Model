from __future__ import annotations

import argparse
import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


print("SCRIPT STARTED")


# ---------------------------
# ARGUMENTS
# ---------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape YouTube top-level comments into a CSV file."
    )

    parser.add_argument(
        "--channel-query",
        type=str,
        default="Ryan Trahan",
        help="Channel name search query",
    )

    parser.add_argument(
        "--max-videos",
        type=int,
        default=30,
        help="Number of videos to scan",
    )

    parser.add_argument(
        "--max-comments-per-video",
        type=int,
        default=250,
        help="Max comments per video",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory",
    )

    return parser.parse_args()


# ---------------------------
# CLIENT
# ---------------------------
def build_youtube_client():
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        raise RuntimeError("Missing YOUTUBE_API_KEY in .env")

    return build("youtube", "v3", developerKey=api_key)


# ---------------------------
# CHANNEL RESOLVE
# ---------------------------
def resolve_channel_id(youtube, channel_query: str) -> Optional[str]:
    response = (
        youtube.search()
        .list(part="snippet", q=channel_query, type="channel", maxResults=5)
        .execute()
    )

    for item in response.get("items", []):
        return item["snippet"]["channelId"]

    return None


# ---------------------------
# GET VIDEOS
# ---------------------------
def fetch_channel_videos(youtube, channel_id: str, max_videos: int, sleep: float):
    response = (
        youtube.search()
        .list(
            part="id,snippet",
            channelId=channel_id,
            type="video",
            order="date",
            maxResults=min(50, max_videos),
        )
        .execute()
    )

    videos = []

    for item in response.get("items", []):
        video_id = item["id"].get("videoId")
        if not video_id:
            continue

        videos.append(
            {
                "video_id": video_id,
                "video_title": item["snippet"].get("title", ""),
                "video_published_at": item["snippet"].get("publishedAt", ""),
            }
        )

    time.sleep(sleep)
    return videos


# ---------------------------
# COMMENTS
# ---------------------------
def fetch_comments_for_video(
    youtube,
    video_meta: Dict[str, str],
    max_comments: int,
    sleep: float,
):
    rows = []
    page_token = None
    video_id = video_meta["video_id"]

    # RANDOMIZE API ORDER (slight variation per run)
    order_choice = random.choice(["relevance", "time"])

    while len(rows) < max_comments:
        try:
            response = (
                youtube.commentThreads()
                .list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(100, max_comments - len(rows)),
                    textFormat="plainText",
                    pageToken=page_token,
                    order=order_choice,
                )
                .execute()
            )

        except HttpError as err:
            status = getattr(err.resp, "status", None)
            if status in (403, 404):
                break
            raise

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comment_id = item["snippet"]["topLevelComment"]["id"]
            text = snippet.get("textDisplay", "")

            if not text.strip():
                continue

            rows.append(
                {
                    "video_id": video_id,
                    "video_title": video_meta["video_title"],
                    "comment_id": comment_id,
                    "author": snippet.get("authorDisplayName", ""),
                    "text": text,
                    "like_count": snippet.get("likeCount", 0),
                    "published_at": snippet.get("publishedAt", ""),
                }
            )

        page_token = response.get("nextPageToken")

        if not page_token:
            break

        time.sleep(sleep)

    # ADD RANDOMNESS IN FINAL SAMPLE
    random.shuffle(rows)

    return rows[:max_comments]


# ---------------------------
# MAIN
# ---------------------------
def main():
    args = parse_args()
    youtube = build_youtube_client()

    channel_id = resolve_channel_id(youtube, args.channel_query)

    if not channel_id:
        raise RuntimeError("Channel not found")

    print(f"Using channel: {channel_id}")

    videos = fetch_channel_videos(
        youtube,
        channel_id,
        args.max_videos,
        sleep=0.1,
    )

    # RANDOMIZE VIDEO ORDER
    random.shuffle(videos)

    all_rows = []

    for i, video in enumerate(videos, start=1):
        print(f"[{i}/{len(videos)}] {video['video_title']}")

        rows = fetch_comments_for_video(
            youtube,
            video,
            args.max_comments_per_video,
            sleep=0.1,
        )

        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No comments collected")

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["comment_id"])

    # ---------------------------
    # HUMAN-READABLE FILE NAMING
    # ---------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob("ryan_trahan_comments_run_*.csv"))

    next_run = len(existing) + 1
    filename = f"ryan_trahan_comments_run_{next_run:02d}.csv"

    output_path = output_dir / filename

    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nSaved {len(df)} comments → {output_path}")


if __name__ == "__main__":
    main()
