#!/usr/bin/env python3
"""Benchmark accentuk performance: trie size, latency, throughput, RSS."""

from __future__ import annotations

import argparse
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

from accentuk import Accentor, Lexicon


def get_rss_mb() -> float:
    # Linux ru_maxrss is in kilobytes
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024.0


def benchmark_lexicon(trie_path: Path, sample_words: list[str]) -> dict:
    trie_size_bytes = trie_path.stat().st_size
    trie_size_mb = trie_size_bytes / (1024 * 1024)

    # Cold init
    t0 = time.perf_counter()
    lex = Lexicon(trie_path)
    cold_init_ms = (time.perf_counter() - t0) * 1000

    num_keys = len(lex)

    # Lookup latencies
    latencies = []
    # Warmup
    for w in sample_words[:50]:
        lex.lookup(w)

    for w in sample_words:
        t0 = time.perf_counter()
        _ = lex.lookup(w)
        latencies.append((time.perf_counter() - t0) * 1_000_000)  # microseconds

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    mean_us = float(np.mean(latencies))

    return {
        "trie_size_bytes": trie_size_bytes,
        "trie_size_mb": round(trie_size_mb, 2),
        "num_keys": num_keys,
        "cold_init_ms": round(cold_init_ms, 2),
        "lookups_tested": len(sample_words),
        "lookup_p50_us": round(p50, 2),
        "lookup_p95_us": round(p95, 2),
        "lookup_p99_us": round(p99, 2),
        "lookup_mean_us": round(mean_us, 2),
    }


def benchmark_engine(
    asset_dir: Path,
    sentences: list[str],
    combiner: bool = False,
) -> dict:
    rss_before = get_rss_mb()

    # Cold engine init
    t0 = time.perf_counter()
    engine = Accentor(asset_dir=asset_dir, enable_combiner=combiner)
    cold_init_sec = time.perf_counter() - t0

    # First sentence warms up lazy model weights
    t0 = time.perf_counter()
    _ = engine.stress(sentences[0], combiner=combiner)
    first_call_warmup_sec = time.perf_counter() - t0

    rss_after = get_rss_mb()

    latencies_ms = []
    total_words = 0

    for sent in sentences:
        total_words += len(sent.split())
        t0 = time.perf_counter()
        res = engine.stress(sent, combiner=combiner)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    total_time_sec = sum(latencies_ms) / 1000
    throughput_sents_sec = len(sentences) / total_time_sec if total_time_sec > 0 else 0
    throughput_words_sec = total_words / total_time_sec if total_time_sec > 0 else 0

    p50_ms = float(np.percentile(latencies_ms, 50))
    p95_ms = float(np.percentile(latencies_ms, 95))
    p99_ms = float(np.percentile(latencies_ms, 99))

    return {
        "cold_init_sec": round(cold_init_sec, 3),
        "first_call_warmup_sec": round(first_call_warmup_sec, 3),
        "rss_before_mb": round(rss_before, 1),
        "rss_after_mb": round(rss_after, 1),
        "rss_delta_mb": round(rss_after - rss_before, 1),
        "sentences_tested": len(sentences),
        "total_words": total_words,
        "throughput_sentences_per_sec": round(throughput_sents_sec, 1),
        "throughput_words_per_sec": round(throughput_words_sec, 1),
        "latency_p50_ms": round(p50_ms, 1),
        "latency_p95_ms": round(p95_ms, 1),
        "latency_p99_ms": round(p99_ms, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark accentuk")
    parser.add_argument("--assets", default="/tmp/accentuk-bundle-test", help="Asset directory")
    parser.add_argument("--limit", type=int, default=30, help="Number of sentences for inference benchmark")
    parser.add_argument("--combiner", action="store_true", help="Enable combiner")
    args = parser.parse_args()

    assets_path = Path(args.assets)
    trie_path = assets_path / "lexicon.marisa"
    if not trie_path.is_file():
        trie_path = assets_path / "lexicon" / "lexicon.marisa"
    if not trie_path.is_file():
        print(f"Error: {trie_path} not found.")
        return 1

    print("=" * 65)
    print("ACCENTUK BENCHMARK SUITE")
    print("=" * 65)
    print(f"Platform:       {platform.platform()}")
    print(f"Python:         {platform.python_version()}")
    print(f"Asset path:     {assets_path}")
    print()

    # 1. Lexicon benchmark
    print("1. Lexicon MARISA Trie Benchmark:")
    words_sample = [
        "мова", "слово", "книга", "день", "ніч", "людина", "життя",
        "сестри", "руки", "води", "землі", "голови", "вікна",
        "ходити", "робити", "бачити", "знати", "жити", "писати",
        "великий", "новий", "добрий", "гарний", "перший",
    ] * 200  # 5,000 lookups

    lex_res = benchmark_lexicon(trie_path, words_sample)
    print(f"  Trie file size:       {lex_res['trie_size_mb']} MB ({lex_res['trie_size_bytes']:,} bytes)")
    print(f"  Total keys in trie:   {lex_res['num_keys']:,}")
    print(f"  Cold init time:       {lex_res['cold_init_ms']} ms")
    print(f"  Lookup p50 latency:   {lex_res['lookup_p50_us']} µs")
    print(f"  Lookup p95 latency:   {lex_res['lookup_p95_us']} µs")
    print(f"  Lookup p99 latency:   {lex_res['lookup_p99_us']} µs")
    print(f"  Lookup mean latency:  {lex_res['lookup_mean_us']} µs")
    print()

    # 2. Engine benchmark
    print("2. Full Accentor Inference Benchmark (CPU):")
    sample_sentences = [
        "Спе́ршу йому́ здало́ся, що закре́слено найкра́щі місця́, які́ він та́к стара́нно випи́сував, ни́жучи сло́во до сло́ва.",
        "Ти́ нічо́го не відчува́єш, бо ми не ру́хаємося в буква́льному розумі́нні цього́ сло́ва.",
        "Оки́нувши ву́лицю по́глядом короле́ви, що поверта́ється у́ свої́ володі́ння, мі́сіс Ві́льсон узя́ла на ру́ки пе́сика.",
        "Він відчинив двері ключем і побачив старовинний замок на горі.",
        "Мої сестри приїхали додому після довгої подорожі.",
    ]
    # Remove accents for test input
    import unicodedata, re
    clean_sents = [
        unicodedata.normalize("NFC", re.sub(r"[\u0300\u0301\u0342]", "", unicodedata.normalize("NFC", s)))
        for s in sample_sentences
    ]
    test_batch = (clean_sents * 10)[: args.limit]

    eng_res = benchmark_engine(assets_path, test_batch, combiner=args.combiner)
    print(f"  Engine cold init:     {eng_res['cold_init_sec']} s")
    print(f"  Model load warmup:    {eng_res['first_call_warmup_sec']} s")
    print(f"  Process RSS (initial):{eng_res['rss_before_mb']} MB")
    print(f"  Process RSS (loaded): {eng_res['rss_after_mb']} MB (delta: +{eng_res['rss_delta_mb']} MB)")
    print(f"  Sentences tested:     {eng_res['sentences_tested']} ({eng_res['total_words']} words)")
    print(f"  Throughput:           {eng_res['throughput_sentences_per_sec']} sentences/s ({eng_res['throughput_words_per_sec']} words/s)")
    print(f"  Latency p50:          {eng_res['latency_p50_ms']} ms/sentence")
    print(f"  Latency p95:          {eng_res['latency_p95_ms']} ms/sentence")
    print(f"  Latency p99:          {eng_res['latency_p99_ms']} ms/sentence")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
