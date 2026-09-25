"""An ear check on the ranker-judge disagreements.

Takes the disagreements the judge wrote, samples them evenly over source
and reading, cuts a few seconds of audio around each word, and writes a
local page: the sentence with the word marked, the clip, and two buttons —
the ranker's reading and the judge's — plus "unclear". Answers accumulate
in the browser and export as JSON to paste back. The page and the clips
stay on this machine: the audio is the books' and the channels'.
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import random
import subprocess
import unicodedata
from pathlib import Path

VOWELS = "аеєиіїоуюяй"
ACUTE = "́"


def stressed(form: str, signature: str) -> str:
    """The form with the acute on the vowel the signature names."""
    target = int(signature) if signature.isdigit() else -1
    out, seen = [], -1
    for ch in unicodedata.normalize("NFC", form):
        out.append(ch)
        if ch.lower() in VOWELS:
            seen += 1
            if seen == target:
                out.append(ACUTE)
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", type=Path, default=Path("output/ml/judge_ranker"))
    parser.add_argument("--corpus", type=Path, default=Path("output/ml/corpus/tok-v6"))
    parser.add_argument("--plan", type=Path, default=Path("/home/devops/audiotostress/artifacts/books/plan.json"))
    parser.add_argument("--youtube", type=Path, default=Path("/home/devops/audiotostress/youtube"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--pad", type=float, default=1.5, help="seconds before and after the word")
    parser.add_argument("--out", type=Path, default=Path("output/ml/ear_check"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    disagreements = [json.loads(l) for l in (args.judge / "disagreements.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    cells: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for d in disagreements:
        cells[(d["source"], d["reading"])].append(d)
    rng = random.Random(args.seed)
    picked = []
    for key in sorted(cells):
        rng.shuffle(cells[key])
        picked += cells[key][: args.n // len(cells)]

    # Where in the audio: book rows kept start_s in the corpus, YouTube rows
    # did not, so both are looked up at the source by (book, sentence, start).
    wanted = {(d["book"], d["sentence"], d["start"]) for d in picked}
    position: dict[tuple, float] = {}
    for part in ("train", "dev", "test"):
        for line in (args.corpus / f"{part}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["book"], r["sentence"], r["start"])
            if key in wanted and r.get("start_s") is not None:
                position[key] = float(r["start_s"])
    for vid in {d["book"].split(":", 1)[1] for d in picked if d["book"].startswith("youtube:")}:
        rows = args.youtube / f"{vid}.rows4.jsonl"
        if not rows.exists():
            continue
        for line in rows.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (f"youtube:{vid}", r.get("sentence"), r.get("start"))
            if key in wanted:
                position[key] = float(r["start_s"])
    # the plan's paths are relative to the audiotostress checkout
    audio_of = {e["slug"]: str(args.plan.parent.parent.parent / e["audio"]) if not e["audio"].startswith("/") else e["audio"]
                for e in json.loads(args.plan.read_text(encoding="utf-8"))}

    clips = args.out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    items = []
    for i, d in enumerate(picked):
        key = (d["book"], d["sentence"], d["start"])
        if key not in position:
            continue
        if d["book"].startswith("youtube:"):
            vid = d["book"].split(":", 1)[1]
            meta = json.loads((args.youtube / vid / "meta.json").read_text(encoding="utf-8"))
            source = args.youtube / vid / meta["audio"]
        else:
            source = Path(audio_of.get(d["book"], ""))
            if source.is_dir() or not source.exists():
                # the plan names the folder's base; take the first audio file in it
                folder = source.parent if not source.is_dir() else source
                source = next((p for p in sorted(folder.glob("*")) if p.suffix.lower() in
                               (".m4b", ".m4a", ".mp3", ".webm", ".opus", ".flac", ".wav")), None)
                if source is None:
                    continue
        start = max(0.0, position[key] - args.pad)
        clip = clips / f"{i:03d}.mp3"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-i", str(source),
                        "-t", f"{2 * args.pad + 1.0:.2f}", "-ac", "1", "-b:a", "64k", str(clip)],
                       check=False)
        if not clip.exists():
            continue
        s = d["sentence"]
        marked = html.escape(s[:d["start"]]) + "<mark>" + html.escape(s[d["start"]:d["end"]]) + "</mark>" + html.escape(s[d["end"]:])
        items.append({"i": i, "clip": clip.name, "sentence": marked, "form": d["form"],
                      "ranker": stressed(d["form"], d["gold"]), "judge": stressed(d["form"], d["judge"]),
                      "source": d["source"], "reading": d["reading"], "book": d["book"],
                      "gold": d["gold"], "judge_sig": d["judge"]})
    (args.out / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")

    page = """<!doctype html><meta charset="utf-8"><title>Ear check</title>
<style>body{font:16px/1.5 system-ui;max-width:52rem;margin:2rem auto;padding:0 1rem}
.item{border-top:1px solid #ccc;padding:1rem 0}mark{background:#ffe08a}
button{font:inherit;padding:.3rem .8rem;margin:.2rem}button.on{background:#2b6;color:#fff}
.meta{color:#666;font-size:.85em}textarea{width:100%;height:8rem}</style>
<h1>Ear check: ranker vs judge</h1>
<p>Listen, then pick what the reader actually said. Answers are kept in this browser; the box at the bottom holds them as JSON to paste back.</p>
<div id="list"></div><h2>Answers</h2><textarea id="out" readonly></textarea>
<script>
const items = ITEMS;
const key = "ear_check_answers";
let answers = {}; try { answers = JSON.parse(localStorage.getItem(key) || "{}"); } catch (e) {}
const list = document.getElementById("list"), out = document.getElementById("out");
function save() { try { localStorage.setItem(key, JSON.stringify(answers)); } catch (e) {} out.value = JSON.stringify(answers); render(); }
function render() {
  list.innerHTML = items.map(it => {
    const a = answers[it.i] || "";
    const opts = [["ranker", it.ranker], ["judge", it.judge], ["unclear", "unclear / neither"]];
    return `<div class="item"><div class="meta">#${it.i} · ${it.source} · ${it.reading} reading · ${it.book}</div>
      <p>${it.sentence}</p><audio controls preload="none" src="clips/${it.clip}"></audio><div>` +
      opts.map(([v, label]) => `<button class="${a === v ? "on" : ""}" onclick="answers[${it.i}]='${v}';save()">${label}</button>`).join("") +
      `</div></div>`; }).join("");
  out.value = JSON.stringify(answers);
}
render();
</script>"""
    (args.out / "index.html").write_text(page.replace("ITEMS", json.dumps(items, ensure_ascii=False)), encoding="utf-8")
    print(f"{len(items)} items -> {args.out / 'index.html'}  (open with: python3 -m http.server -d {args.out} 8765)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
