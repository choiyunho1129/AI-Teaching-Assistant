from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)


@dataclass
class VideoItem:
    video_id: str
    title: str
    index: Optional[int] = None


def sanitize_filename(name: str, max_len: int = 180) -> str:
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|\n\r\t]+", "_", name)
    name = re.sub(r"\s+", " ", name)
    return name[:max_len].rstrip()


def extract_playlist_videos(playlist_url: str) -> Tuple[str, List[VideoItem]]:
    """
    Returns (playlist_title, list_of_videos).
    Uses yt-dlp in flat mode to avoid downloading videos.
    """
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,   # fast playlist listing
        "noplaylist": False,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

    playlist_title = info.get("title") or "playlist"
    entries = info.get("entries") or []

    videos: List[VideoItem] = []
    for e in entries:
        # yt-dlp may yield dict entries (common) or URLs (rare). Handle dict primarily.
        if isinstance(e, dict):
            vid = e.get("id") or e.get("url")
            title = e.get("title") or f"video_{vid}"
            idx = e.get("playlist_index")
            if vid:
                videos.append(VideoItem(video_id=vid, title=title, index=idx))
        elif isinstance(e, str):
            # fallback: if entry is a URL, try to extract v=ID
            m = re.search(r"(?:v=|youtu\.be/)([0-9A-Za-z_-]{11})", e)
            if m:
                videos.append(VideoItem(video_id=m.group(1), title=f"video_{m.group(1)}"))

    # Sort by playlist index if available
    videos.sort(key=lambda x: (x.index is None, x.index if x.index is not None else 10**9))
    return playlist_title, videos


def pick_transcript(video_id: str, langs: list[str], translate_to: str | None):
    """
    Returns transcript as list of dicts:
      [{'text':..., 'start':..., 'duration':...}, ...]
    Works with youtube-transcript-api v1.x (instance API).
    """
    ytt_api = YouTubeTranscriptApi()

    # translate 안 쓰면 fetch()로 바로 (가장 간단/빠름)
    if not translate_to:
        fetched = ytt_api.fetch(video_id, languages=langs)
        return fetched.to_raw_data()

    # translate 쓰는 경우: list() -> find_transcript() -> translate() -> fetch()
    transcript_list = ytt_api.list(video_id)
    transcript = transcript_list.find_transcript(langs)
    translated = transcript.translate(translate_to)
    fetched = translated.fetch()
    return fetched.to_raw_data()

def sec_to_hhmmss(seconds: float) -> str:
    # 1시간 이상도 HH로 정상 표기
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcript_to_text_with_timestamps(segments: list[dict], mode: str) -> str:
    """
    segments: [{'text':..., 'start':..., 'duration':...}, ...]
    mode: none | start | range
    """
    if mode == "none":
        # 기존처럼 텍스트만 합치기
        parts = []
        for seg in segments:
            t = (seg.get("text") or "").replace("\n", " ").strip()
            if t:
                parts.append(t)
        return " ".join(parts).strip() + "\n"

    lines = []
    for seg in segments:
        text = (seg.get("text") or "").replace("\n", " ").strip()
        if not text:
            continue

        start = float(seg.get("start", 0.0))
        dur = float(seg.get("duration", 0.0))
        end = start + dur

        if mode == "start":
            lines.append(f"[{sec_to_hhmmss(start)}] {text}")
        elif mode == "range":
            lines.append(f"[{sec_to_hhmmss(start)}-{sec_to_hhmmss(end)}] {text}")

    return "\n".join(lines) + "\n"
    
def transcript_to_text(segments: Iterable[dict]) -> str:
    # Join lines; remove embedded newlines for cleaner plain text
    parts = []
    for s in segments:
        t = (s.get("text") or "").replace("\n", " ").strip()
        if t:
            parts.append(t)
    return " ".join(parts).strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "playlist_url",
        help="YouTube playlist URL (e.g., https://www.youtube.com/playlist?list=...)",
    )
    ap.add_argument("--timestamps", choices=["none", "start", "range"], default="start",
                help="Include timestamps per segment: start=[HH:MM:SS], range=[HH:MM:SS-HH:MM:SS]")
    ap.add_argument("--out-dir", default="playlist_transcripts", help="Output directory")
    ap.add_argument("--langs", nargs="+", default=["ko", "en"], help="Preferred language codes in order")
    ap.add_argument("--translate", default=None, help="Translate transcript to this language code (e.g., en, ko)")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between videos (rate-limit friendly)")
    ap.add_argument("--combined", action="store_true", help="Also write a combined_all.txt file")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    playlist_title, videos = extract_playlist_videos(args.playlist_url)
    safe_playlist_title = sanitize_filename(playlist_title)
    print(f"[Playlist] {playlist_title}")
    print(f"[Found] {len(videos)} videos")

    failed = []
    combined_lines = []

    for i, v in enumerate(videos, start=1):
        idx = v.index if v.index is not None else i
        safe_title = sanitize_filename(v.title)
        fname = f"{idx:02d}_{safe_title}_{v.video_id}.txt"
        fpath = out_dir / fname

        try:
            segments = pick_transcript(v.video_id, args.langs, args.translate)
            text = transcript_to_text_with_timestamps(segments, args.timestamps)

            fpath.write_text(text, encoding="utf-8")
            print(f"[OK] {idx:02d} {v.video_id} -> {fname}")

            if args.combined:
                combined_lines.append(f"===== {idx:02d} | {v.title} | {v.video_id} =====\n")
                combined_lines.append(text)
                combined_lines.append("\n")

        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, CouldNotRetrieveTranscript) as e:
            msg = f"{idx:02d} {v.video_id} | {v.title} | {type(e).__name__}"
            print(f"[FAIL] {msg}")
            failed.append(msg)

        time.sleep(max(0.0, args.sleep))

    # Write combined + failures
    if args.combined and combined_lines:
        combined_path = out_dir / f"{safe_playlist_title}_combined_all.txt"
        combined_path.write_text("".join(combined_lines), encoding="utf-8")
        print(f"[COMBINED] {combined_path}")

    if failed:
        fail_path = out_dir / f"{safe_playlist_title}_failed.txt"
        fail_path.write_text("\n".join(failed) + "\n", encoding="utf-8")
        print(f"[FAILED LIST] {fail_path}")

    print("Done.")


if __name__ == "__main__":
    main()