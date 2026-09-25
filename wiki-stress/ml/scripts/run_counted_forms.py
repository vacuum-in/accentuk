"""Build the counted-form list from the Orthoepic Dictionary.

After `два/дві/три/чотири` some Ukrainian nouns take the *counted form*
(рахункова форма, historically a dual), stressed differently from the
nominative plural they are spelled like. Which nouns is lexical: `три сестри́`
but `дві вели́кі площи́ни`, and no parse can tell them apart because both are
tagged `Case=Nom|Number=Plur`.

Nothing infers this. The Orthoepic Dictionary prints the numeral phrase with its
stress for exactly the nouns that take it, and prints nothing for the rest, so
this is a lookup:

    сестра   -> "чотири сестри́"      counted
    площина  -> no numeral phrase     nominative

That criterion was checked against lang-uk's gold on five words and matched on
four, both negatives included. See STATE.md for why a rule over the accent class
is wrong and why no language model gets this right.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import unicodedata
import urllib.parse
from pathlib import Path
from typing import Any

ACUTE = "́"
SOURCE = "https://slovnyk.me/dict/orthoepy/"
NUMERALS = ("два", "дві", "три", "чотири", "обидва", "обидві")
NUMERAL_PHRASE = re.compile(
    rf"(?:{'|'.join(NUMERALS)})\s+([^\W\d_]+{ACUTE}[^\W\d_]*)", re.UNICODE
)
TAGS = re.compile(r"<[^>]+>")
SCRIPTS = re.compile(r"<script.*?</script>|<style.*?</style>", re.DOTALL)


def page_text(html_text: str) -> str:
    stripped = TAGS.sub(" ", html.unescape(SCRIPTS.sub("", html_text)))
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", stripped))


def numeral_forms(text: str) -> list[str]:
    """Every stressed word the page shows directly after a 2/3/4 numeral."""
    return list(dict.fromkeys(NUMERAL_PHRASE.findall(text)))


def fetch(client: Any, lemma: str, retries: int = 3) -> str | None:
    url = SOURCE + urllib.parse.quote(lemma)
    for attempt in range(retries):
        try:
            response = client.get(url, timeout=30, follow_redirects=True)
        except Exception:  # noqa: BLE001 - retried, then reported as a miss
            time.sleep(1 + attempt)
            continue
        if response.status_code == 200:
            return response.text
        if response.status_code == 404:
            return None
        time.sleep(1 + attempt)
    return None


def lemmas_for(forms: list[str], database_url: str) -> dict[str, list[str]]:
    """Map each surface form to the lemmas the lexicon records for it."""
    import psycopg

    out: dict[str, list[str]] = {}
    with psycopg.connect(database_url) as connection:
        rows = connection.execute(
            "SELECT form_normalized, array_agg(DISTINCT lemma_normalized) "
            "FROM stress_lookup WHERE form_normalized = ANY(%s) "
            "GROUP BY form_normalized",
            (forms,),
        ).fetchall()
    for form, lemmas in rows:
        out[form] = list(lemmas)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path,
                        default=Path("output/ml/counted_form_worklist.json"))
    parser.add_argument("--out", type=Path, default=Path("output/ml/counted_forms.json"))
    parser.add_argument("--database-url",
                        default="postgresql://ukstress_owner:ukstress_owner@localhost:5432/ukstress")
    parser.add_argument("--delay", type=float, default=0.7,
                        help="seconds between requests; this is someone else's server")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    import httpx

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    if args.limit:
        candidates = candidates[: args.limit]
    forms = [c["form"] for c in candidates]
    lemmas = lemmas_for(forms, args.database_url)
    print(f"{len(candidates)} candidates, lemmas known for {len(lemmas)}", flush=True)

    results: list[dict[str, Any]] = []
    counted = nominative = unresolved = 0
    with httpx.Client(headers={"user-agent": "ukstress-counted-form/1.0"}) as client:
        for index, candidate in enumerate(candidates, 1):
            form = candidate["form"]
            entry: dict[str, Any] = {
                "form": form,
                "nominative_plural": candidate["nominative_plural"],
                "genitive_singular": candidate["genitive_singular"],
            }
            found: list[str] = []
            checked: list[str] = []
            for lemma in lemmas.get(form, []):
                body = fetch(client, lemma)
                checked.append(lemma)
                time.sleep(args.delay)
                if body is None:
                    continue
                found.extend(numeral_forms(page_text(body)))
            entry["lemmas_checked"] = checked
            entry["numeral_phrases"] = list(dict.fromkeys(found))

            # The dictionary prints the phrase only for nouns that take the
            # form. Absence is an answer, not a gap — but only if a page was
            # actually read.
            if not checked:
                entry["verdict"] = "unresolved"
                entry["reason"] = "no lemma for this form in the lexicon"
                unresolved += 1
            elif entry["numeral_phrases"]:
                stressed = next(
                    (p for p in entry["numeral_phrases"]
                     if p.replace(ACUTE, "") == form), entry["numeral_phrases"][0]
                )
                entry["verdict"] = "counted"
                entry["stressed"] = stressed
                entry["source"] = SOURCE + urllib.parse.quote(checked[0])
                counted += 1
            else:
                entry["verdict"] = "nominative"
                entry["stressed"] = candidate["nominative_plural"]
                entry["source"] = SOURCE + urllib.parse.quote(checked[0])
                nominative += 1
            results.append(entry)
            if index % 10 == 0 or index == len(candidates):
                print(f"  {index}/{len(candidates)} counted={counted} "
                      f"nominative={nominative} unresolved={unresolved}", flush=True)

    args.out.write_text(
        json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}: {counted} counted, {nominative} nominative, "
          f"{unresolved} unresolved", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
