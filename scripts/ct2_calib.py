"""Calibrate the CT2 INT8 cross-encoder against the FP32 model it was converted from.

Both engines stress the same sentences (silver corpus sentences, not any test set); every cross-encoder decision
is recorded with its top signature and margin. The INT8 margin threshold is then chosen so that INT8 selects
the same targets FP32 selects (FP32 at its served threshold 0.5), and the logit scale is reported.

    ct2_calib.py bundle_dir n_sentences
"""
import os
import collections, json, random, sys
import numpy as np
from accentuk import Accentor

BUNDLE, N = sys.argv[1], int(sys.argv[2])
rows = json.load(open(os.environ.get("ACCENTUK_CALIB_SENTENCES", "silver_mined_v10.json"), encoding="utf-8"))
random.seed(11)
sents = sorted({r["sentence"] for r in random.sample(rows, N * 3)})[:N]

def record(engine, name):
    engine.stress("Мої сестри прийшли додому.")          # load the models
    model = engine._context_model
    seen = {}
    orig = model.resolve
    def wrapped(targets):
        out = orig(targets)
        for t, d in zip(targets, out):
            if d and d.get("scores"):
                key = (getattr(t, "sentence", None) or t["sentence"], getattr(t, "start", None) if not isinstance(t, dict) else t["start"])
                seen[key] = (d["signature"], d["margin"], d["scores"])
        return out
    model.resolve = wrapped
    for s in sents:
        engine.stress(s)
    print(f"{name}: {len(seen)} cross-encoder decisions", flush=True)
    return seen

full = record(Accentor(asset_dir=BUNDLE, mode="full", device="cuda"), "fp32")
light = record(Accentor(asset_dir=BUNDLE, mode="light"), "int8")
keys = [k for k in full if k in light]
same_top = np.array([full[k][0] == light[k][0] for k in keys])
m32 = np.array([full[k][1] for k in keys]); m8 = np.array([light[k][1] for k in keys])
print(f"paired decisions {len(keys)}; same top reading {same_top.mean():.1%}")
slope, icpt = np.polyfit(m8, m32, 1)
print(f"margin fp32 ~ {slope:.3f} * margin int8 + {icpt:.3f}; correlation {np.corrcoef(m8, m32)[0, 1]:.3f}")
sel32 = m32 >= 0.5
print(f"fp32 selects {sel32.mean():.1%} at threshold 0.5; int8 at 0.5 selects {(m8 >= 0.5).mean():.1%}")
best = None
for t in np.arange(0.0, 1.01, 0.02):
    sel8 = m8 >= t
    agree = ((sel8 == sel32) & (~sel8 | same_top)).mean()   # same select/abstain, and same reading when selecting
    if best is None or agree > best[1]:
        best = (round(float(t), 2), float(agree))
    if abs(t - 0.5) < 1e-9 or abs(t - 0.2) < 1e-9 or abs(t - 0.3) < 1e-9:
        print(f"   int8 threshold {t:.2f}: decision agreement with fp32 {agree:.2%}")
print(f"best int8 threshold {best[0]} -> agreement {best[1]:.2%}")
json.dump({"threshold_int8": best[0], "agreement": best[1], "pairs": len(keys), "slope": slope}, open("/tmp/ct2_calib.json", "w"))
