"""The review pages, served on the laptop for the LAN.

Three pages under one root: the ear check on ranker-judge disagreements,
the modern-text gold draft (every stress mark editable by clicking the
vowel), and table A of the dictionary audit (one verdict per form). Each
page keeps its answers in the browser and also POSTs them to /save/<name>,
which lands in <root>/<name>.json, so nothing needs pasting back. The pages
and the audio stay on this machine.

    run_review_server.py --root output/ml/review --port 8765
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import unicodedata
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ACUTE = "́"
VOWELS = "аеєиіїоуюяАЕЄИІЇОУЮЯ"

STYLE = """<style>body{font:17px/1.6 system-ui;max-width:56rem;margin:2rem auto;padding:0 1rem;color:#222;background:#fff}
h1{font-size:1.4rem}nav a{margin-right:1rem}.item{border-top:1px solid #ddd;padding:.8rem 0}
mark{background:#ffe08a}button{font:inherit;padding:.25rem .7rem;margin:.15rem;border:1px solid #999;border-radius:4px;background:#f6f6f6;cursor:pointer}
button.on{background:#2b7a3a;color:#fff;border-color:#2b7a3a}.meta{color:#666;font-size:.85em}
.w{cursor:pointer;padding:0 1px;border-radius:3px}.w:hover{background:#eef}.v{cursor:pointer}
.form{font-weight:600;margin:1.2rem 0 .3rem;font-size:1.05em}.line{padding:.25rem 0;display:flex;gap:.5rem;align-items:baseline}
.line.del span.t{text-decoration:line-through;color:#999}.ex{color:#444;font-size:.9em;margin:.2rem 0 .2rem 1rem}
#status{position:fixed;top:.5rem;right:.8rem;font-size:.85em;color:#2b7a3a}
.line.edited{background:#fff4d1;border-left:4px solid #e0a800;padding-left:.4rem;border-radius:3px}
.w.changed{background:#ffcf85;font-weight:600}.counts{color:#555;margin:.4rem 0 1rem}</style>
<nav><a href="/">index</a><a href="/ear/">ear check</a><a href="/gold/">gold draft</a><a href="/tablea/">table A</a></nav><div id="status"></div>"""

SAVE_JS = """<script>
function post(name, data){document.getElementById("status").textContent="saving…";
 fetch("/save/"+name,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)})
 .then(r=>r.ok?"saved "+new Date().toLocaleTimeString():"save failed").catch(()=>"save failed (offline)")
 .then(t=>document.getElementById("status").textContent=t);}
function load(name, fallback){try{const v=localStorage.getItem("review_"+name); if(v) return JSON.parse(v);}catch(e){} return fallback;}
function keep(name, data){try{localStorage.setItem("review_"+name, JSON.stringify(data));}catch(e){} post(name, data);}
// What the server holds wins over nothing: a page opened in another browser
// starts empty locally, and its first save would otherwise overwrite every
// answer given elsewhere. Server answers fill whatever this browser lacks.
async function sync(name, state, render){
 try{const r=await fetch("/"+name+".json",{cache:"no-store"}); if(!r.ok) return;
  const server=await r.json(); let added=0;
  for(const k in server){ if(!(k in state)){ state[k]=server[k]; added++; } }
  if(added){ try{localStorage.setItem("review_"+name, JSON.stringify(state));}catch(e){} render();
   document.getElementById("status").textContent="loaded "+added+" saved answers"; }
 }catch(e){}
}
</script>"""


def gold_page(text: str) -> str:
    """Every stressed vowel is a click target; clicking a vowel moves the mark."""
    groups, current = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if re.match(r"#\s*\S+:\s", line):
                current = {"head": line.lstrip("# ").strip(), "lines": []}
                groups.append(current)
            continue
        if current is None:
            current = {"head": "", "lines": []}
            groups.append(current)
        current["lines"].append(unicodedata.normalize("NFC", line))
    payload = json.dumps(groups, ensure_ascii=False)
    return f"""<!doctype html><meta charset="utf-8"><title>Gold draft</title>{STYLE}
<h1>Gold draft — modern text</h1>
<p>Click a <b>vowel</b> to move the stress mark of that word there. Click ✕ to drop a line that is not a sentence (click again to restore). Everything is saved as you go; the corrected file is rebuilt from your answers.</p>
<div class="counts" id="counts"></div>
<label><input type="checkbox" id="onlyEdited" onchange="render()"> show only edited lines</label>
<div id="list"></div>{SAVE_JS}<script>
const groups = {payload};
const VOW = "{VOWELS}", ACUTE = "\\u0301";
let state = load("gold", {{}});   // key "g:l" -> {{text, deleted}}
function key(g, l) {{ return g + ":" + l; }}
function textOf(g, l) {{ const s = state[key(g, l)]; return s && s.text !== undefined ? s.text : groups[g].lines[l]; }}
function render() {{
  const out = [];
  const only = document.getElementById("onlyEdited").checked;
  let edited = 0, dropped = 0;
  groups.forEach((grp, g) => {{
    const lines = [];
    grp.lines.forEach((original, l) => {{
      const s = state[key(g, l)] || {{}};
      const t = textOf(g, l);
      const changed = t !== original;
      edited += changed; dropped += !!s.deleted;
      if (only && !changed && !s.deleted) return;
      const cls = ["line", s.deleted ? "del" : "", changed ? "edited" : ""].join(" ");
      lines.push(`<div class="${{cls}}"><button title="drop this line" onclick="toggleDel(${{g}},${{l}})">✕</button><span class="t">${{words(t, g, l, original)}}</span></div>`);
    }});
    if (lines.length) {{
      out.push(`<div class="form">${{grp.head ? esc(grp.head) : ""}}</div>`);
      out.push(...lines);
    }}
  }});
  document.getElementById("list").innerHTML = out.join("");
  document.getElementById("counts").textContent = `${{edited}} lines edited, ${{dropped}} dropped`;
}}
function esc(s) {{ return s.replace(/&/g, "&amp;").replace(/</g, "&lt;"); }}
function words(t, g, l, original) {{
  // split on spaces, keep punctuation attached; each vowel is clickable;
  // a word whose mark differs from the draft is highlighted
  const was = (original || "").split(" ");
  return t.split(" ").map((w, wi) => {{
    let html = "", vi = 0;
    for (const ch of w) {{
      if (ch === ACUTE) {{ html += "\\u0301"; continue; }}
      if (VOW.includes(ch)) {{ html += `<span class="v" onclick="setStress(${{g}},${{l}},${{wi}},${{vi}})">${{esc(ch)}}</span>`; vi++; }}
      else html += esc(ch);
    }}
    return `<span class="w${{was[wi] !== undefined && was[wi] !== w ? " changed" : ""}}">${{html}}</span>`;
  }}).join(" ");
}}
function setStress(g, l, wi, vi) {{
  const ws = textOf(g, l).split(" ");
  let out = "", seen = 0;
  for (const ch of ws[wi]) {{
    if (ch === ACUTE) continue;
    out += ch;
    if (VOW.includes(ch)) {{ if (seen === vi) out += ACUTE; seen++; }}
  }}
  ws[wi] = out;
  state[key(g, l)] = Object.assign({{}}, state[key(g, l)] || {{}}, {{text: ws.join(" ")}});
  keep("gold", state); render();
}}
function toggleDel(g, l) {{
  const s = state[key(g, l)] || {{}}; s.deleted = !s.deleted; state[key(g, l)] = s; keep("gold", state); render();
}}
render();
sync("gold", state, render);
</script>"""


def table_a_page(rows: list[dict], applied: dict[str, str]) -> str:
    """Table A in Ukrainian, one card per form, the choice spelled out.

    `applied` maps the forms whose narrators' reading is already served (the
    unanimous ones, applied automatically) to that reading, so the reviewer
    can skip them or overrule them.
    """
    payload = json.dumps(rows, ensure_ascii=False)
    applied_payload = json.dumps(applied, ensure_ascii=False)
    return f"""<!doctype html><meta charset="utf-8"><title>Таблиця A</title>{STYLE}
<style>.pair{{font-size:1.25em;margin:.3rem 0}}.pair b{{font-size:1.1em}}.lex{{color:#8a4b00}}.nar{{color:#0b5d8a}}
.legend{{background:#f4f6f8;border-radius:6px;padding:.6rem .9rem;margin:.6rem 0 1rem}}
.legend div{{margin:.2rem 0}}.badge{{display:inline-block;background:#e6f4ea;color:#1e6b34;border-radius:4px;padding:0 .4rem;font-size:.85em}}
.choices button{{display:block;width:100%;text-align:left;margin:.25rem 0}}.why{{color:#666;font-size:.85em}}</style>
<h1>Таблиця A: словник має одне читання, диктори кажуть інше</h1>
<div class="legend">
<div>Для кожного слова: <span class="lex">ліворуч — як наголошує словник зараз</span>,
<span class="nar">праворуч — як його читають диктори</span> в аудіокнигах і на YouTube (скільки прочитань і скільки різних дикторів).
Три приклади — речення, де слово трапилось.</div>
<div><b>Обидва правильні</b> — лишити словникове читання й додати читання дикторів; обиратиме контекст.</div>
<div><b>Правильно лише читання дикторів</b> — словник помиляється, замінити.</div>
<div><b>Правильно лише словникове</b> — диктори помиляються, нічого не міняти.</div>
<div><b>Не знаю</b> — пропустити.</div>
<div><span class="badge">вже застосовано</span> — читання дикторів уже подається автоматично (одностайні випадки); вердикт тут його змінить.</div>
</div>
<div class="counts" id="counts"></div>
<label><input type="checkbox" id="hideDone" onchange="render()"> сховати слова з вердиктом</label>
<label style="margin-left:1rem"><input type="checkbox" id="hideApplied" onchange="render()"> сховати вже застосовані</label>
<div id="list"></div>{SAVE_JS}<script>
const rows = {payload};
const applied = {applied_payload};
let state = load("tablea", {{}});
function esc(s) {{ return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;"); }}
function render() {{
  const hideDone = document.getElementById("hideDone").checked;
  const hideApplied = document.getElementById("hideApplied").checked;
  let done = 0;
  const out = [];
  rows.forEach((r, i) => {{
    const a = state[r.form] || "";
    if (a) done++;
    if ((hideDone && a) || (hideApplied && applied[r.form])) return;
    const ex = (r.examples || []).slice(0, 3).map(e => `<div class="ex">${{esc(e).replace(/«([^»]+)»/g, "<mark>$1</mark>")}}</div>`).join("");
    const badge = applied[r.form] ? ` <span class="badge">вже застосовано: ${{esc(applied[r.form])}}</span>` : "";
    const choices = [
      ["add", `Обидва правильні — ${{esc(r.lexicon)}} і ${{esc(r.narrators)}}`, "додати читання дикторів, обиратиме контекст"],
      ["exclusive", `Правильно лише ${{esc(r.narrators)}}`, "словник помиляється — замінити"],
      ["keep", `Правильно лише ${{esc(r.lexicon)}}`, "диктори помиляються — нічого не міняти"],
      ["unsure", "Не знаю", "пропустити"]];
    out.push(`<div class="item"><div class="meta">#${{i + 1}} · диктори: ${{r.rows}} з ${{r.total}} прочитань (${{Math.round(r.share * 100)}}%) · ${{r.books}} різних дикторів${{badge}}</div>
      <div class="pair"><span class="lex">словник: <b>${{esc(r.lexicon)}}</b></span> &nbsp;→&nbsp; <span class="nar">диктори: <b>${{esc(r.narrators)}}</b></span></div>${{ex}}
      <div class="choices">` + choices.map(([v, label, why]) =>
        `<button class="${{a === v ? "on" : ""}}" onclick="state['${{r.form}}']='${{v}}';keep('tablea',state);render()">${{label}} <span class="why">— ${{why}}</span></button>`).join("") + `</div></div>`);
  }});
  document.getElementById("list").innerHTML = out.join("");
  document.getElementById("counts").textContent = `вердиктів: ${{done}} з ${{rows.length}}; вже застосовано автоматично: ${{Object.keys(applied).length}}`;
}}
render();
sync("tablea", state, render);
</script>"""


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        name = self.path.rsplit("/", 1)[-1]
        if not self.path.startswith("/save/") or not re.fullmatch(r"[a-z_]+", name):
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            data = json.loads(body)
        except ValueError:
            self.send_error(400)
            return
        (Path(self.directory) / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        self.send_response(204)
        self.end_headers()

    def end_headers(self):
        # answers change on every click; never let a browser cache them
        if self.path.endswith(".json"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):  # quiet
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/ml/review"))
    parser.add_argument("--ear", type=Path, default=Path("output/ml/ear_check"))
    parser.add_argument("--gold", type=Path, default=Path("ml/data/gold_modern_draft_1.txt"))
    parser.add_argument("--audit", type=Path, default=Path("/home/devops/audiotostress/DICTIONARY_AUDIT_ROUND4.json"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    if args.ear.exists():
        if (root / "ear").exists():
            shutil.rmtree(root / "ear")
        shutil.copytree(args.ear, root / "ear")
    (root / "gold").mkdir(exist_ok=True)
    (root / "gold" / "index.html").write_text(gold_page(args.gold.read_text(encoding="utf-8")), encoding="utf-8")
    (root / "tablea").mkdir(exist_ok=True)
    rows = json.loads(args.audit.read_text(encoding="utf-8"))["A"]
    rows.sort(key=lambda r: (-r["books"], -r["share"]))
    applied: dict[str, str] = {}
    for name in ("decisions_round4_auto.json", "decisions_round4_family.json", "decisions_round4_90.json"):
        path = Path("ml/data") / name
        if path.exists():
            for d in json.loads(path.read_text(encoding="utf-8")):
                applied[d["form"]] = d["stressed"]
    (root / "tablea" / "index.html").write_text(table_a_page(rows, applied), encoding="utf-8")
    (root / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Review</title>{STYLE}
<h1>Review</h1><ul>
<li><a href="/ear/">Ear check</a> — 100 ranker-vs-judge clips (done)</li>
<li><a href="/gold/">Gold draft</a> — modern-text gold, {sum(1 for l in args.gold.read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#'))} sentences over 50 forms: fix any mark, drop non-sentences</li>
<li><a href="/tablea/">Table A</a> — {len(rows)} forms where narrators read what the lexicon lacks: one verdict each</li>
</ul><p class="meta">Answers save to this machine on every click (and stay in the browser as a backup).</p>""", encoding="utf-8")
    print(f"pages -> {root}")
    if args.build_only:
        return 0
    server = ThreadingHTTPServer(("0.0.0.0", args.port), lambda *a, **k: Handler(*a, directory=str(root), **k))
    print(f"serving {root} on :{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
