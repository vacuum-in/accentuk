"""Browse the ML datasets in a browser.

    ml/.venv/bin/python ml/scripts/run_dataset_browser.py --port 7870

Files reach 275 MB, so nothing is loaded to render a page: each is indexed once
into byte offsets and afterwards a row is a seek. The first open of a large file
pays for the scan; every later one is instant.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import time
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset browser</title>
<style>
:root{--ground:#eef1f4;--surface:#fff;--sunken:#e6ebf0;--ink:#16202b;--muted:#5d6b7a;
--line:#d5dce4;--accent:#1f6f8b;--hit:#fdf0c0;--chosen:#2e6d4f;--other:#8a8477;
--wrong:#a8453a;--recheck:#8a6212;--ok:#2e6d4f;--tag:#4a5a86;
--body:ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
@media(prefers-color-scheme:dark){:root{--ground:#0f151b;--surface:#171f27;--sunken:#131a21;
--ink:#e4eaf0;--muted:#93a2b1;--line:#28323c;--accent:#5fb3cf;--hit:#3b3418;
--chosen:#6cbb92;--other:#7d8794;--wrong:#e0897c;--recheck:#d6ad55;--ok:#6cbb92;--tag:#8fa1cf}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--body);
font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{display:grid;grid-template-columns:21rem 1fr;min-height:100vh}
aside{background:var(--surface);border-right:1px solid var(--line);overflow-y:auto;
max-height:100vh;position:sticky;top:0}
.side-head{padding:.9rem 1rem .6rem;border-bottom:1px solid var(--line);position:sticky;top:0;
background:var(--surface);z-index:2}
.side-head input{width:100%}
.grp{font-family:var(--mono);font-size:.64rem;letter-spacing:.14em;text-transform:uppercase;
color:var(--accent);padding:.9rem 1rem .3rem}
.ds{padding:.5rem 1rem .55rem;border-bottom:1px solid var(--line);cursor:pointer}
.ds:hover{background:var(--sunken)}
.ds.on{background:var(--sunken);box-shadow:inset 3px 0 0 var(--accent)}
.ds b{display:block;font-weight:600;font-size:.84rem;line-height:1.3}
.ds i{display:block;font-style:normal;font-size:.76rem;color:var(--muted);margin-top:.1rem}
.ds span{font-family:var(--mono);font-size:.67rem;color:var(--muted)}
main{padding:1.2rem 1.6rem 4rem;min-width:0}
h2{font-size:1.3rem;margin:0 0 .2rem;line-height:1.25}
.detail{color:var(--muted);max-width:74ch;margin:0 0 .9rem}
.stats{display:flex;flex-wrap:wrap;gap:.35rem .5rem;margin-bottom:1rem}
.stat{background:var(--surface);border:1px solid var(--line);border-radius:4px;
padding:.3rem .6rem;font-family:var(--mono);font-size:.73rem;color:var(--muted)}
.stat b{color:var(--ink);font-variant-numeric:tabular-nums}
.bar{display:flex;flex-wrap:wrap;gap:.45rem;align-items:center;margin-bottom:.9rem;
position:sticky;top:0;background:var(--ground);padding:.5rem 0;z-index:1}
input,button,select{font:inherit;padding:.34rem .6rem;border:1px solid var(--line);
border-radius:4px;background:var(--surface);color:var(--ink)}
button{cursor:pointer}button:hover{border-color:var(--accent)}
.pos{font-family:var(--mono);font-size:.74rem;color:var(--muted);min-width:9rem}
.chips{display:flex;flex-wrap:wrap;gap:.3rem;margin-bottom:1rem}
.chip{font-family:var(--mono);font-size:.71rem;padding:.16rem .5rem;border:1px solid var(--line);
border-radius:99px;background:var(--surface);cursor:pointer;color:var(--muted)}
.chip:hover{border-color:var(--accent);color:var(--ink)}
.chip b{color:var(--ink)}
.row{background:var(--surface);border:1px solid var(--line);border-radius:6px;
padding:.85rem 1rem;margin-bottom:.6rem}
.key{font-family:var(--mono);font-size:.95rem;font-weight:600;margin-bottom:.35rem}
.sent{font-size:1.06rem;line-height:1.72}
mark{background:var(--hit);color:inherit;padding:.05em .18em;border-radius:3px;font-weight:600}
.opts{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.6rem}
.opt{border:1px solid var(--line);border-radius:4px;padding:.28rem .55rem;font-size:.82rem;
background:var(--ground);max-width:34rem;cursor:pointer}
.opt:hover{border-color:var(--accent)}
.opt[data-picked="1"]{border-color:var(--wrong);box-shadow:inset 0 0 0 1px var(--wrong)}
.opt.on{border-color:var(--chosen);box-shadow:inset 0 0 0 1px var(--chosen)}
.opt .w{font-weight:600}
.opt.on .w{color:var(--chosen)}
.opt .w:not(.on){color:var(--other)}
.opt .d{display:block;color:var(--muted);font-size:.75rem;line-height:1.4;margin-top:.1rem}
.opt .tick{font-family:var(--mono);font-size:.68rem;color:var(--chosen);margin-left:.3rem}
.kv{display:flex;flex-wrap:wrap;gap:.2rem .8rem;font-family:var(--mono);font-size:.71rem;
color:var(--muted);margin-top:.55rem;border-top:1px solid var(--line);padding-top:.45rem}
.kv b{color:var(--ink);font-weight:600}
pre{font-family:var(--mono);font-size:.73rem;background:var(--sunken);padding:.55rem;
border-radius:4px;overflow-x:auto;margin:.3rem 0 0;max-height:22rem}
.empty{color:var(--muted);padding:2.5rem 0}
.review{display:flex;flex-wrap:wrap;gap:.35rem;align-items:center;margin-top:.6rem;
border-top:1px solid var(--line);padding-top:.5rem}
.vb{font-family:var(--mono);font-size:.71rem;padding:.2rem .55rem;border:1px solid var(--line);
border-radius:99px;background:var(--ground);cursor:pointer;color:var(--muted)}
.vb:hover{border-color:var(--accent);color:var(--ink)}
.vb.on-wrong{background:var(--wrong);border-color:var(--wrong);color:#fff}
.vb.on-recheck{background:var(--recheck);border-color:var(--recheck);color:#fff}
.vb.on-ok{background:var(--ok);border-color:var(--ok);color:#fff}
.review .hint{font-family:var(--mono);font-size:.68rem;color:var(--muted)}
select,.band input{font-family:var(--mono);font-size:.78rem;padding:.25rem .4rem;
border:1px solid var(--line);border-radius:4px;background:var(--ground);color:var(--ink)}
.band{font-family:var(--mono);font-size:.72rem;color:var(--muted);
display:inline-flex;align-items:center;gap:.3rem}
.band input{width:4.5rem}
.freq{font-family:var(--mono);font-size:.68rem;color:var(--muted);
border:1px solid var(--line);border-radius:99px;padding:.05rem .45rem;margin-left:.4rem}
.freq b{color:var(--accent);font-variant-numeric:tabular-nums}
.review .sep{width:1px;align-self:stretch;background:var(--line);margin:0 .3rem}
.tb{font-family:var(--mono);font-size:.71rem;padding:.2rem .55rem;border:1px dashed var(--line);
border-radius:3px;background:var(--ground);cursor:pointer;color:var(--muted)}
.tb:hover{border-color:var(--tag);color:var(--ink)}
.tb.on{background:var(--tag);border-color:var(--tag);border-style:solid;color:#fff}
.badge.tag{background:var(--tag)}
.badge{font-family:var(--mono);font-size:.67rem;padding:.1rem .4rem;border-radius:3px;
color:#fff;margin-left:.4rem}
.badge.wrong{background:var(--wrong)}.badge.recheck{background:var(--recheck)}
.badge.ok{background:var(--ok)}
.row[data-v="wrong"]{border-left:3px solid var(--wrong)}
.row[data-v="recheck"]{border-left:3px solid var(--recheck)}
.row[data-v="ok"]{border-left:3px solid var(--ok)}
.row[data-v=""]:not([data-tags=""]){border-left:3px dashed var(--tag)}
label.tog{font-family:var(--mono);font-size:.73rem;color:var(--muted);display:flex;
align-items:center;gap:.3rem;cursor:pointer}
</style></head><body>
<div class="wrap">
<aside>
  <div class="side-head"><input id="dsq" placeholder="filter datasets…"></div>
  <div id="list"></div>
  <div id="audits"></div>
</aside>
<main>
  <h2 id="title">…</h2>
  <p class="detail" id="detail"></p>
  <div class="stats" id="stats"></div>
  <div class="bar">
    <input id="q" placeholder="find a word, e.g. замок" size="20">
    <button id="find">Find</button>
    <button id="prev">&larr;</button><span class="pos" id="pos"></span><button id="next">&rarr;</button>
    <select id="size"><option>25</option><option>50</option><option>100</option></select>
    <select id="sort" title="row order">
      <option value="file">file order</option>
      <option value="freq">most frequent first</option>
      <option value="freq_asc">rarest first</option>
    </select>
    <span class="band">freq
      <input id="minfreq" type="number" min="0" placeholder="min" size="4">
      <input id="maxfreq" type="number" min="0" placeholder="max" size="4">
    </span>
    <label class="tog"><input type="checkbox" id="raw"> raw JSON</label>
    <label class="tog"><input type="checkbox" id="onlymarked"> reviewed only</label>
    <a id="dl" class="vb" href="#" download>export reviews</a>
  </div>
  <div id="chips" class="chips"></div>
  <div id="rows"></div>
</main></div>
<script>
let all=[], current=null, offset=0, mode="page";
const $=s=>document.querySelector(s);
const fmt=n=>n>=1e6?(n/1e6).toFixed(1)+" MB":n>=1e3?(n/1e3).toFixed(0)+" kB":n+" B";
const esc=s=>(s??"").toString().replace(/[<>&]/g,c=>({"<":"&lt;",">":"&gt;","&":"&amp;"}[c]));

async function boot(){
  all=await (await fetch("/api/datasets")).json();
  paint();
  const audits=await (await fetch("/api/audits")).json();
  if(audits.length){
    $("#audits").innerHTML=`<div class="grp">audits</div>`+audits.map(a=>
      `<a class="ds" href="/audits/${encodeURIComponent(a.name)}" target="_blank">
         <b>${esc(a.title)}</b><i>${esc(a.name)} · ${esc(a.when)}</i></a>`).join("");
  }
  if(all.length) open(all[0].name);
}
function paint(){
  const f=$("#dsq").value.trim().toLowerCase();
  const shown=all.filter(d=>!f||d.name.toLowerCase().includes(f)||
    (d.title||"").toLowerCase().includes(f));
  let html="", group=null;
  for(const d of shown){
    if(d.group!==group){group=d.group;html+=`<div class="grp">${esc(group)}</div>`;}
    html+=`<div class="ds${d.name===current?" on":""}" data-n="${esc(d.name)}">
      <b>${esc(d.title||d.name)}</b>
      ${d.title?`<i>${esc(d.name)}</i>`:""}
      <span>${fmt(d.size)}</span></div>`;
  }
  $("#list").innerHTML=html||'<div class="empty" style="padding:1rem">Nothing matches.</div>';
  document.querySelectorAll(".ds").forEach(el=>el.onclick=()=>open(el.dataset.n));
}
async function open(name){
  current=name; offset=0; mode="page"; paint();
  $("#rows").innerHTML='<div class="empty">Indexing — a large file is scanned once, then cached.</div>';
  $("#chips").innerHTML=""; $("#stats").innerHTML="";
  const s=await (await fetch(`/api/datasets/${encodeURIComponent(name)}`)).json();
  $("#title").textContent=s.title||name;
  $("#detail").textContent=s.detail||"";
  if(s.error){$("#rows").innerHTML=`<div class="empty">${esc(s.error)}</div>`;return;}
  const bits=[`<span class="stat"><b>${s.rows.toLocaleString()}</b> rows</span>`,
    `<span class="stat">${fmt(s.size)}</span>`,
    `<span class="stat">${esc(s.name)}</span>`];
  if(s.labels.length) bits.push(`<span class="stat">labels ${s.labels.map(([k,v])=>
    `<b>${esc(k)}</b>·${v}`).join(" ")}</span>`);
  if(s.keys.length) bits.push(`<span class="stat">${s.keys.length} fields</span>`);
  bits.push(`<span class="stat" id="revcount"></span>`);
  $("#stats").innerHTML=bits.join("");
  counts(s.annotations);
  $("#dl").href=`/api/datasets/${encodeURIComponent(name)}/annotations`;
  $("#dl").setAttribute("download", name.replace(/\.[^.]+$/,"")+"-reviews.jsonl");
  $("#chips").innerHTML=s.forms.slice(0,30).map(([f,n])=>
    `<span class="chip" data-f="${esc(f)}">${esc(f)} <b>${n}</b></span>`).join("");
  document.querySelectorAll(".chip").forEach(c=>c.onclick=()=>{$("#q").value=c.dataset.f;find();});
  load();
}
function band(){
  const p=new URLSearchParams();
  p.set("sort", $("#sort").value);
  const lo=$("#minfreq").value.trim(), hi=$("#maxfreq").value.trim();
  if(lo!=="") p.set("minfreq", lo);
  if(hi!=="") p.set("maxfreq", hi);
  return p;
}
async function load(){
  const n=+$("#size").value;
  const p=band(); p.set("offset", offset); p.set("limit", n);
  const r=await (await fetch(`/api/datasets/${encodeURIComponent(current)}/rows?${p}`)).json();
  render(r.rows);
  const filtered=r.unsorted_total&&r.total!==r.unsorted_total
    ? ` (filtered from ${r.unsorted_total.toLocaleString()})` : "";
  $("#pos").textContent=r.total
    ? `${(offset+1).toLocaleString()}–${(offset+r.rows.length).toLocaleString()} of ${r.total.toLocaleString()}${filtered}`
    : `no rows in this band${filtered}`;
}
async function find(){
  const f=$("#q").value.trim();
  if(!f){mode="page";offset=0;return load();}
  mode="find"; $("#rows").innerHTML='<div class="empty">Scanning…</div>';
  const r=await (await fetch(`/api/datasets/${encodeURIComponent(current)}/find?form=${encodeURIComponent(f)}&limit=${+$("#size").value}`)).json();
  render(r.rows);
  $("#pos").textContent=r.rows.length?`${r.rows.length} × ${f}`:`no ${f} in ${r.scanned.toLocaleString()}`;
}
function render(rows){
  if($("#onlymarked").checked) rows=(rows||[]).filter(r=>r._note);
  if(!rows||!rows.length){$("#rows").innerHTML=
    '<div class="empty">Nothing here'+($("#onlymarked").checked?" — no reviewed rows on this page.":".")+'</div>';return;}
  const raw=$("#raw").checked;
  $("#rows").innerHTML=rows.map(r=>{
    if(raw) return `<div class="row"><pre>${esc(JSON.stringify(r,null,1))}</pre></div>`;
    let body="";
    const freq=r._freq!==undefined
      ? `<span class="freq" title="occurrences in the corpus sample"><b>${r._freq.toLocaleString()}</b>×</span>`
      : "";
    const head=r._key!==undefined?r._key:(r.form||"");
    if(head||freq) body+=`<div class="key">${esc(head)}${freq}</div>`;
    if(r._target!==undefined){
      const w=r._stressed||r._target;
      body+=`<div class="sent">${esc(r._before)}<mark>${esc(w)}</mark>${esc(r._after)}</div>`;
    }
    const picked=r._note&&r._note.proposed||"";
    if(r._options){
      body+=`<div class="opts">`+r._options.map(o=>
        `<div class="opt${o.chosen?" on":""}" data-sig="${esc(o.signature)}"
              data-picked="${o.signature===picked?1:0}" title="click to propose this reading">
           <span class="w${o.chosen?" on":""}">${esc(o.stressed)}</span>
           ${o.chosen?'<span class="tick">chosen</span>':""}
           ${o.signature===picked?'<span class="tick" style="color:var(--wrong)">proposed</span>':""}
           ${o.definition?`<span class="d">${esc(o.definition)}</span>`:""}
         </div>`).join("")+`</div>`;
    }
    if(!body) body=`<pre>${esc(JSON.stringify(r,null,1)).slice(0,1500)}</pre>`;
    const skip=new Set(["sentence","_before","_target","_after","_stressed","_options","_key","_signature"]);
    const kv=Object.entries(r).filter(([k,v])=>!skip.has(k)&&(typeof v!=="object"||v===null))
      .map(([k,v])=>`<span>${esc(k)} <b>${esc(v)}</b></span>`).join("");
    const v=r._note?r._note.verdict:"";
    const tags=(r._note&&r._note.tags)||[];
    const review=`<div class="review">
      ${["wrong","recheck","ok"].map(k=>
        `<span class="vb${v===k?" on-"+k:""}" data-v="${k}">${k}</span>`).join("")}
      <span class="sep"></span>
      ${[["not-frequent","not frequent"],["toponym","toponym"],
         ["alt-needed","needs alt variant"]].map(([k,label])=>
        `<span class="tb${tags.includes(k)?" on":""}" data-t="${k}">${label}</span>`).join("")}
      <span class="sep"></span>
      <span class="hint">${r._options?"or click a reading to propose it":""}</span>
      ${r._note&&r._note.note?`<span class="hint">note: ${esc(r._note.note)}</span>`:""}
      <span class="vb" data-v="note">note…</span>
    </div>`;
    return `<div class="row" data-v="${v}" data-tags="${tags.join(" ")}" data-row="${r._row}"
      data-form="${esc(r.form||r._key||"")}"
      data-was="${esc(r._signature??r.gold_signature??r.gold??"")}"
      data-sent="${esc(r.sentence||"")}">${body}${kv?`<div class="kv">${kv}</div>`:""}${review}</div>`;
  }).join("");
  wire();
}
async function mark(el, verdict, proposed, note, tags){
  const row=el.closest(".row");
  if(tags===undefined) tags=(row.dataset.tags||"").split(" ").filter(Boolean);
  const payload={row:+row.dataset.row, verdict, tags, proposed:proposed||"",
    note:note||"", form:row.dataset.form, sentence:row.dataset.sent,
    was:row.dataset.was};
  const r=await (await fetch(`/api/datasets/${encodeURIComponent(current)}/annotate`,
    {method:"POST",headers:{"content-type":"application/json"},
     body:JSON.stringify(payload)})).json();
  if(r.error){alert(r.error);return;}
  row.dataset.v=verdict;
  row.dataset.tags=tags.join(" ");
  row.querySelectorAll(".vb[data-v]").forEach(b=>{
    b.className="vb"+(b.dataset.v===verdict?" on-"+verdict:"");});
  row.querySelectorAll(".tb[data-t]").forEach(b=>{
    b.className="tb"+(tags.includes(b.dataset.t)?" on":"");});
  if(proposed){
    row.querySelectorAll(".opt").forEach(o=>
      o.dataset.picked=o.dataset.sig===proposed?"1":"0");
  }
  counts(r.counts);
}
function counts(c){
  const el=document.querySelector("#revcount");
  if(el) el.innerHTML="reviewed "+Object.entries(c||{})
    .map(([k,v])=>`<b>${v}</b> ${k}`).join(" · ");
}
function wire(){
  document.querySelectorAll(".vb[data-v]").forEach(b=>b.onclick=()=>{
    if(b.dataset.v==="note"){
      const n=prompt("Note for this row:");
      if(n!==null) mark(b, b.closest(".row").dataset.v||"recheck","",n);
    } else mark(b,b.dataset.v,"","");
  });
  document.querySelectorAll(".tb[data-t]").forEach(b=>b.onclick=()=>{
    const row=b.closest(".row");
    const have=(row.dataset.tags||"").split(" ").filter(Boolean);
    const next=have.includes(b.dataset.t)
      ? have.filter(t=>t!==b.dataset.t) : have.concat([b.dataset.t]);
    mark(b, row.dataset.v||"", "", "", next);
  });
  document.querySelectorAll(".opt").forEach(o=>o.onclick=()=>{
    // A numeric signature is a vowel ordinal: picking one says the row's own
    // reading was wrong. A named one is the answer itself — "counted" is not
    // a correction to a mistake, so the row is simply marked reviewed.
    const named = !/^\d/.test(o.dataset.sig);
    mark(o, named ? "ok" : "wrong", o.dataset.sig, "");
  });
}
$("#next").onclick=()=>{if(mode==="page"){offset+=+$("#size").value;load();}};
$("#prev").onclick=()=>{if(mode==="page"){offset=Math.max(0,offset-(+$("#size").value));load();}};
$("#find").onclick=find;
$("#q").onkeydown=e=>{if(e.key==="Enter")find();};
$("#dsq").oninput=paint;
$("#size").onchange=()=>{offset=0;mode==="find"?find():load();};
$("#raw").onchange=()=>{mode==="find"?find():load();};
$("#onlymarked").onchange=()=>{mode==="find"?find():load();};
// A new order or band restarts paging: page 40 of the old order means nothing.
for(const id of ["sort","minfreq","maxfreq"])
  $("#"+id).onchange=()=>{mode="page";offset=0;load();};
boot();
</script></body></html>"""


class Review(BaseModel):
    """One reviewer verdict on one row.

    Declared at module level because FastAPI resolves a handler's type hints
    against the module's globals; a class defined inside the factory is invisible
    there and the body silently degrades into a query parameter.
    """

    row: int
    verdict: str = ""
    tags: list[str] = []
    proposed: str = ""
    note: str = ""
    form: str = ""
    sentence: str = ""
    was: str = ""


AUDIT_PAGE = r"""<!doctype html>
<html lang="uk"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--ground:#eef1f4;--surface:#fff;--ink:#16202b;--muted:#5d6b7a;--line:#d5dce4;--accent:#1f6f8b;--hit:#fdf0c0}
@media(prefers-color-scheme:dark){:root{--ground:#0f151b;--surface:#171f27;--ink:#e4eaf0;--muted:#93a2b1;--line:#28323c;--accent:#5fb3cf;--hit:#3b3418}}
body{margin:0;background:var(--ground);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,sans-serif}
main{max-width:100rem;margin:0 auto;padding:1.5rem 2rem}
h1{font-size:1.4rem;margin:.2rem 0 1rem}h2{font-size:1.1rem;margin:1.6rem 0 .5rem;color:var(--accent)}
p{max-width:70rem;color:var(--muted)}
.tbl{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:6px;margin:.6rem 0 1.2rem}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:.35rem .6rem;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}
th{position:sticky;top:0;background:var(--surface);cursor:pointer;white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr:hover td{background:var(--hit)}
td:last-child input{width:7rem}
b{font-weight:650}
</style></head><body><main>__BODY__</main>
<script>
// Click a header to sort; numbers sort as numbers.
document.querySelectorAll("th").forEach((th,i)=>th.addEventListener("click",()=>{
  const tb=th.closest("table").tBodies[0],rows=[...tb.rows],asc=th.dataset.asc!=="1";
  const num=v=>parseFloat(v.replace(/[^0-9.+-]/g,""));
  rows.sort((a,b)=>{const x=a.cells[i].innerText,y=b.cells[i].innerText,nx=num(x),ny=num(y);
    const c=isNaN(nx)||isNaN(ny)?x.localeCompare(y):nx-ny;return asc?c:-c});
  rows.forEach(r=>tb.appendChild(r));th.dataset.asc=asc?"1":"0";}));
</script></body></html>"""


def render_markdown(text: str) -> str:
    """Enough Markdown for the audit files: headings, paragraphs, bold, and
    pipe tables (with `<br>` inside cells kept as line breaks)."""
    out, table, para = [], [], []
    number = re.compile(r"^[+-]?[\d,.]+ ?(pp|%|h)?$|^\d+/\d+$")

    def inline(cell: str) -> str:
        cell = html.escape(cell).replace("&lt;br&gt;", "<br>")
        cell = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", cell)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", cell)

    def flush_table():
        if not table:
            return
        head, rows = table[0], [r for r in table[1:] if not set(r.replace("|", "").strip()) <= set("-: ")]
        cells = lambda line: [c.strip() for c in line.strip().strip("|").split("|")]
        out.append('<div class="tbl"><table><thead><tr>' + "".join(
            f"<th>{inline(c)}</th>" for c in cells(head)) + "</tr></thead><tbody>")
        for r in rows:
            out.append("<tr>" + "".join(
                f'<td class="{"num" if number.match(c.strip()) else ""}">{inline(c)}</td>'
                for c in cells(r)) + "</tr>")
        out.append("</tbody></table></div>")
        table.clear()

    def flush_para():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()

    for line in text.splitlines():
        if line.startswith("|"):
            flush_para()
            table.append(line)
            continue
        flush_table()
        if line.startswith("# "):
            flush_para(); out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("## "):
            flush_para(); out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.strip() == "":
            flush_para()
        else:
            para.append(line)
    flush_table(); flush_para()
    return "\n".join(out)


def create_app(root: Path, cache: Path):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, PlainTextResponse

    from ukstress_ml import browser, catalogue

    app = FastAPI(title="Dataset browser", docs_url=None, redoc_url=None)
    indexes: dict[str, object] = {}
    manifest: dict = {}
    frequency = browser.load_frequency(root / "ambiguous_frequency.json")
    notes = browser.Annotations(root / ".annotations")

    def forms() -> dict:
        """The serving manifest, used to show what a label was chosen against."""
        if not manifest:
            for name in ("serving_manifest_v25.json", "serving_manifest_v22.json"):
                path = root / name
                if path.is_file():
                    import json as _json

                    manifest.update(
                        _json.loads(path.read_text(encoding="utf-8")).get("forms", {}))
                    break
        return manifest

    def index_for(name: str):
        path = (root / name).resolve()
        # Confine every read to the dataset directory; the name arrives from a URL.
        if root.resolve() not in path.parents or not path.is_file():
            return None
        if name not in indexes:
            indexes[name] = browser.load_index(path, cache)
        return indexes[name]

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return PAGE

    audits_dir = root / "audits"

    def audit_files() -> list[Path]:
        if not audits_dir.is_dir():
            return []
        return sorted(audits_dir.glob("*.md"), key=lambda p: -p.stat().st_mtime)

    @app.get("/api/audits")
    def audits() -> list[dict]:
        out = []
        for path in audit_files():
            first = next((l[2:] for l in path.read_text(encoding="utf-8").splitlines()
                          if l.startswith("# ")), path.stem)
            out.append({"name": path.name, "title": first,
                        "when": time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime))})
        return out

    @app.get("/audits/{name}", response_class=HTMLResponse)
    def audit(name: str) -> str:
        path = (audits_dir / name).resolve()
        if audits_dir.resolve() not in path.parents or not path.is_file():
            return HTMLResponse("no such audit", status_code=404)
        return AUDIT_PAGE.replace("__TITLE__", html.escape(name)).replace(
            "__BODY__", render_markdown(path.read_text(encoding="utf-8")))

    @app.get("/api/datasets")
    def datasets() -> list[dict]:
        found = []
        for item in browser.list_datasets(root):
            entry = catalogue.describe(item["name"])
            found.append({**item, "group": entry.group, "title": entry.title,
                          "detail": entry.detail})
        # Group order first, then size, so the corpora a reader wants are at the
        # top of their own section rather than buried by a bigger neighbour.
        rank = {g: i for i, g in enumerate(catalogue.GROUP_ORDER)}
        found.sort(key=lambda d: (rank.get(d["group"], 99), -d["size"]))
        return found

    @app.get("/api/datasets/{name}")
    def summary(name: str) -> dict:
        index = index_for(name)
        if index is None:
            return {"error": f"no dataset named {name}"}
        entry = catalogue.describe(name)
        return {"annotations": notes.counts(name),
                "name": name, "rows": index.rows, "size": index.size,
                "kind": index.kind, "keys": index.keys,
                "forms": index.forms, "labels": index.labels,
                "group": entry.group, "title": entry.title, "detail": entry.detail}

    # An order is a list of row numbers, so it costs one pass over the index and
    # is reusable across pages. Cached per (dataset, sort, band) because paging
    # through 478k rows would otherwise rebuild it on every click.
    orders: dict[tuple, list[int]] = {}

    def ordering(name: str, index, sort: str,
                 low: int | None, high: int | None) -> list[int] | None:
        if sort == "file" and low is None and high is None:
            return None
        if not frequency:
            return None
        key = (name, sort, low, high)
        if key not in orders:
            if len(orders) > 8:
                orders.clear()
            orders[key] = browser.frequency_order(
                index, frequency, descending=sort != "freq_asc",
                low=low, high=high)
        return orders[key]

    @app.get("/api/datasets/{name}/rows")
    def rows(name: str, offset: int = 0, limit: int = 25, sort: str = "file",
             minfreq: int | None = None, maxfreq: int | None = None) -> dict:
        index = index_for(name)
        if index is None:
            return {"error": f"no dataset named {name}", "rows": []}
        order = ordering(name, index, sort, minfreq, maxfreq)
        total = index.rows if order is None else len(order)
        window = browser.read_rows(index, max(0, offset),
                                   min(max(limit, 1), 200), order)
        marked = notes.latest(name)
        rows = [browser.decorate(r, forms()) for r in window]
        for row in rows:
            existing = marked.get(row.get("_row"))
            if existing:
                row["_note"] = existing
            count = browser.frequency_of(index, row.get("_row", -1), frequency)
            if count is not None:
                row["_freq"] = count
        return {"rows": rows, "total": total, "unsorted_total": index.rows}

    @app.get("/api/datasets/{name}/find")
    def find(name: str, form: str, limit: int = 25) -> dict:
        index = index_for(name)
        if index is None:
            return {"error": f"no dataset named {name}", "rows": [], "scanned": 0}
        found, scanned = browser.find_rows(index, form, min(max(limit, 1), 200))
        marked = notes.latest(name)
        rows = [browser.decorate(r, forms()) for r in found]
        for row in rows:
            existing = marked.get(row.get("_row"))
            if existing:
                row["_note"] = existing
        return {"rows": rows, "scanned": scanned}

    @app.post("/api/datasets/{name}/annotate")
    def annotate(name: str, body: Review) -> dict:
        if body.verdict and body.verdict not in browser.VERDICTS:
            return {"error": f"verdict must be one of {browser.VERDICTS}"}
        bad = [t for t in body.tags if t not in browser.TAGS]
        if bad:
            return {"error": f"unknown tag {bad[0]!r}; tags are {browser.TAGS}"}
        if index_for(name) is None:
            return {"error": f"no dataset named {name}"}
        notes.add(browser.Annotation(
            dataset=name, row=body.row, verdict=body.verdict,
            tags=sorted(set(body.tags), key=browser.TAGS.index),
            proposed=body.proposed, note=body.note[:2000], form=body.form,
            sentence=body.sentence[:2000], was=body.was, at=time.time()))
        return {"ok": True, "counts": notes.counts(name)}

    @app.get("/api/datasets/{name}/annotations", response_class=PlainTextResponse)
    def download(name: str) -> str:
        """Every current verdict as JSONL, ready to feed back into a pipeline."""
        return notes.export(name)

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/ml"))
    parser.add_argument("--cache", type=Path, default=Path("output/ml/.index-cache"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7870)
    args = parser.parse_args(argv)

    import uvicorn

    print(f"datasets in {args.root.resolve()}  ->  http://localhost:{args.port}", flush=True)
    uvicorn.run(create_app(args.root, args.cache), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
