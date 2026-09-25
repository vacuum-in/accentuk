# Mining throughput: what was slow, and why

A 120-second window cost 234–312 seconds to mine. It now costs about 14. Every
one of the three causes was the same shape — a correct result computed many
times where once would do — and none of them was where I first looked.

## The measurements

| stage | before | after |
| --- | ---: | ---: |
| reading the audio | 118 s | ~0 s |
| prosody | 81 s | 3 s |
| ASR | 23–102 s | 7–8 s |
| word alignment | 4–44 s | 3 s |
| **window total** | **234–312 s** | **~14 s** |

## Reading the audio: `soundfile` cannot seek into an mp3

It decodes from the start to reach the offset, at about 9.7 ms per second of
offset:

| offset | read 120 s |
| ---: | ---: |
| 600 s | 7.0 s |
| 6,000 s | 59.2 s |
| 12,000 s | 117.3 s |
| 24,000 s | 232.4 s |

Mining walks forward, so every window paid for all the audio before it. Near
the end of a 32-hour book one window would have cost nineteen minutes.

This also explains a difference I had misattributed: a run starting at 600 s
averaged 2.4 minutes a window and one starting at 12,000 s took 5.3. I put that
down to denser speech. It was the offset.

`scripts/run_decode_audio.py` writes a 16-kHz mono WAV once — 3.7 GB, 22.7
minutes — after which reads take 0.01 s at any offset.

## Prosody: one F0 track per word instead of one per window

`extract_prosodic_features` sliced the word out for the energy measurements and
then handed `f0_track` the **whole waveform**:

```python
word_samples = _slice(waveform, ..., word_start_s, word_end_s)   # the word
f0_times, f0_values = f0_track(waveform, ...)                    # the lot
```

The track is the same for every word in a window, and it was recomputed for
each. On a two-minute window with 115 words that is 3.8 hours of
autocorrelation to describe 60 seconds of speech. The result was right; 99.6%
of the work was thrown away. This is why the stage cost 81 s regardless of the
material or the ASR model.

`extract_prosodic_features` now takes an optional `f0_cache`, and the miner
computes one track per window: 81 s to 1 s.

## What was not the bottleneck

The fine aligner re-ran WhisperX **once per word** — 121 forward passes over a
window where one suffices — and `align()` already computed character timings
and discarded them. Removing that was worth **6%**, measured A/B on identical
windows. Cheap short passes are not the same as expensive long ones, and the
count of calls said nothing useful about the cost.

Three hypotheses about the bottleneck were wrong before the fourth was right,
and each was refuted by measurement rather than by reading the code.

## A trap worth naming

Four times, a shell loop of the form

```bash
until ! ps -eo args | grep -q '[r]un_something'; do sleep 15; done
```

never exited, because the waiting shell's own command line contains the
pattern. The `[r]` trick stops grep matching itself; it does nothing about the
parent. One of these ran for 25 minutes while I read its elapsed time as
progress — the giveaways were 0.0% CPU and 3 MB of RSS, impossible for a
process with numpy loaded.
