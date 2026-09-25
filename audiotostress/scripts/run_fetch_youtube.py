"""Fetch the audio of YouTube videos, playlists, channels or searches.

Audio only, best available, one folder per video under --out with the
stream as `audio.<ext>` and `meta.json` (id, title, uploader, duration,
url, upload date). A video already fetched is skipped, so a playlist can be
re-run to pick up what is new. Sources are lines in a file or arguments:
a URL, a playlist, a channel, or `ytsearchN:<query>`.

What is fetched here is under the same rule as the audiobooks: it feeds the
miner and never leaves this machine. Nothing derived from it carries more
than the word, its reading and one sentence of context.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import threading
import time
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", help="URLs, playlists, channels, ytsearchN:query")
    parser.add_argument("--list", type=Path, help="a file with one source per line")
    parser.add_argument("--out", type=Path, default=Path("youtube"))
    parser.add_argument("--min-minutes", type=float, default=10.0,
                        help="skip anything shorter: the miner works in ten-minute windows")
    parser.add_argument("--max-hours", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=0, help="stop after this many new videos")
    parser.add_argument("--cookies", type=Path,
                        help="a Netscape cookies.txt exported from a browser logged in "
                             "to YouTube; the way past 'Sign in to confirm you're not a bot'")
    parser.add_argument("--cookies-from-browser", metavar="BROWSER",
                        help="firefox, chrome, ...: read the cookies of a browser on this "
                             "machine that is logged in to YouTube")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="seconds between downloads (yt-dlp sleeps 1..2x this); "
                             "a few seconds keeps a long fetch under the bot check")
    parser.add_argument("--reverse", action="store_true",
                        help="take the list from the end: a second machine on the same "
                             "source meets the first in the middle")
    parser.add_argument("--min-free-gb", type=float, default=0.0,
                        help="wait while the output disk has less free space than this "
                             "(the Mac has 17 GB; the miner frees audio as it goes)")
    parser.add_argument("--workers", type=int, default=2,
                        help="parallel yt-dlp downloads; two keep a 5090 fed, more "
                             "invites the bot check")
    parser.add_argument("--stop-after-bot-checks", type=int, default=20,
                        help="after this many bot checks in a row, pause for --bot-pause "
                             "instead of burning the whole list; the block lifts by itself "
                             "(two workers earned one after ~45 min and it was gone 3 h later)")
    parser.add_argument("--bot-pause", type=float, default=1800.0,
                        help="seconds to wait after a run of bot checks before going on")
    parser.add_argument("--max-pauses", type=int, default=48,
                        help="give up after this many pauses (a day at the default)")
    parser.add_argument("--per-source", type=int, default=0,
                        help="fetch at most this many new videos from each source: "
                             "a pilot takes three from a channel before the channel")
    args = parser.parse_args()

    sources = list(args.sources)
    if args.list:
        sources += [l.strip() for l in args.list.read_text(encoding="utf-8").splitlines()
                    if l.strip() and not l.startswith("#")]
    if not sources:
        parser.error("no sources")
    args.out.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in args.out.iterdir() if (p / "meta.json").exists()}

    # IPv4 only: YouTube's bot check followed the IPv6 prefix, not the account
    # (four accounts clean over v4 from one machine, all blocked over v6 from
    # another), so every request goes over the shared v4 address.
    auth = ["--force-ipv4"]
    if args.cookies:
        auth += ["--cookies", str(args.cookies)]
    if args.cookies_from_browser:
        auth += ["--cookies-from-browser", args.cookies_from_browser]
    pace = (["--sleep-interval", str(args.sleep), "--max-sleep-interval", str(2 * args.sleep),
             "--sleep-requests", "1"] if args.sleep else [])
    bot_checks = 0

    # First pass: list entries with durations without downloading, so the
    # length filter runs before any bytes move.
    entries = []
    for source in sources:
        out = subprocess.run(
            ["yt-dlp", "--flat-playlist", "--dump-json", "--ignore-errors", *auth, source],
            capture_output=True, text=True)
        taken = 0
        for line in out.stdout.splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not e.get("id") or e["id"] in have:
                continue
            duration = e.get("duration") or 0
            if duration and not (args.min_minutes * 60 <= duration <= args.max_hours * 3600):
                continue
            if args.per_source and taken >= args.per_source:
                break
            e["source"] = source
            entries.append(e)
            taken += 1
        print(f"  {source[:60]:<60} {taken:>4} candidates", flush=True)
    print(f"{len(entries)} entries listed from {len(sources)} sources; {len(have)} already fetched",
          flush=True)

    # Downloads run in a few threads: one yt-dlp stream is ~40x realtime while
    # the 5090 mines at ~100x, so a single fetcher leaves it idle half the time.
    state = {"fetched": 0, "bot_checks": 0, "stop": False, "pauses": 0, "resume_at": 0.0}
    lock = threading.Lock()

    def fetch_one(e):
        vid = e["id"]
        if state["stop"]:
            return
        duration = e.get("duration") or 0
        if not duration:
            # The flat listing has no length for Shorts and some old uploads;
            # one metadata request is cheaper than a download to throw away.
            probe = subprocess.run(["yt-dlp", "--simulate", "--print", "%(duration)s",
                                    "--no-playlist", "--quiet", "--no-warnings", *auth,
                                    e.get("url") or f"https://www.youtube.com/watch?v={vid}"],
                                   capture_output=True, text=True)
            try:
                duration = float(probe.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError):
                duration = 0
        if not (args.min_minutes * 60 <= duration <= args.max_hours * 3600):
            print(f"  skip {vid} {(e.get('title') or '')[:50]!r}: {duration / 60:.0f} min", flush=True)
            return
        while time.time() < state["resume_at"] and not state["stop"]:
            time.sleep(30)
        while args.min_free_gb and shutil.disk_usage(args.out).free < args.min_free_gb * 2**30:
            time.sleep(60)   # the miner deletes audio it has pushed and mined
        if state["stop"]:
            return
        folder = args.out / vid
        with lock:
            if (folder / "meta.json").exists() or (folder / ".fetching").exists():
                return   # another worker or an earlier run has it
            folder.mkdir(exist_ok=True)
            (folder / ".fetching").touch()
        url = e.get("url") or e.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}"
        result = subprocess.run(
            ["yt-dlp", "-f", "bestaudio/best", "-x", "--audio-format", "m4a",
             "--no-playlist", "--quiet", "--no-warnings", *auth, *pace,
             "-o", str(folder / "audio.%(ext)s"), "--print-json", url],
            capture_output=True, text=True)
        (folder / ".fetching").unlink(missing_ok=True)
        if result.returncode != 0 or not result.stdout.strip():
            print(f"  FAILED {vid}: {result.stderr.strip()[:120]}", flush=True)
            with lock:
                if "not a bot" in result.stderr:
                    state["bot_checks"] += 1
                    if state["bot_checks"] >= args.stop_after_bot_checks and not state["stop"]:
                        state["bot_checks"] = 0
                        state["pauses"] += 1
                        if state["pauses"] > args.max_pauses:
                            print(f"=== bot check again after {args.max_pauses} pauses; "
                                  f"stopping", flush=True)
                            state["stop"] = True
                        else:
                            state["resume_at"] = time.time() + args.bot_pause
                            print(f"=== {args.stop_after_bot_checks} bot checks in a row "
                                  f"({time.strftime('%H:%M')}): pause {args.bot_pause / 60:.0f} min, "
                                  f"#{state['pauses']}", flush=True)
            return
        info = json.loads(result.stdout.splitlines()[-1])
        audio = next((p for p in folder.iterdir() if p.name.startswith("audio.")), None)
        if audio is None:
            print(f"  FAILED {vid}: no audio file", flush=True)
            return
        meta = {"id": vid, "source": e.get("source"), "title": info.get("title"), "uploader": info.get("uploader"),
                "channel": info.get("channel"), "duration_s": info.get("duration"),
                "upload_date": info.get("upload_date"), "url": info.get("webpage_url", url),
                "language": info.get("language"), "audio": audio.name}
        # meta.json marks a folder complete for the rsyncs, so it must appear whole
        (folder / "meta.json.tmp").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        (folder / "meta.json.tmp").replace(folder / "meta.json")
        with lock:
            state["bot_checks"] = 0
            state["fetched"] += 1
            if args.limit and state["fetched"] >= args.limit:
                state["stop"] = True
        print(f"  {vid}  {(info.get('duration') or 0) / 3600:.1f} h  {info.get('title', '')[:60]}", flush=True)

    todo = [e for e in entries if e["id"] not in have]
    if args.reverse:
        todo.reverse()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch_one, todo))
    fetched = state["fetched"]
    print(f"{fetched} fetched -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
