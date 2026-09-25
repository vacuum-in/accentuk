#!/usr/bin/env bash
# Fetch the Ukrainian morphological tagger the serving path uses.
#
# `pip install` of the published wheel does not work here: it pins spaCy 3.7.5,
# whose `blis` dependency has no wheel for Python 3.14 and fails to build. The
# model data itself is version-independent, so the wheel is unpacked and loaded
# by path instead. spaCy warns that a 3.7.0 model is running on 3.8.x; measured
# on lang-uk's benchmark it scores 73.50% heteronym accuracy against Stanza's
# 72.96%, so the warning is noted rather than acted on.
#
# Why this model and not Stanza: same accuracy from 15 MB instead of ~500 MB
# and a 0.8s load instead of ~30s. The size is what makes a parser pool viable,
# and the pool is what stops one instance serialising the API — under eight
# concurrent callers Stanza dropped word accuracy from 87.89% to 72.16%.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${1:-uk_core_news_sm}"
DEST="models/${MODEL}"
URL="https://huggingface.co/spacy/${MODEL}/resolve/main/${MODEL}-any-py3-none-any.whl"

if [ -f "${DEST}/config.cfg" ]; then
    echo "${DEST} already present"
    exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
echo "fetching ${MODEL}…"
curl -fL --retry 3 -o "${tmp}/model.whl" "${URL}"
unzip -qo "${tmp}/model.whl" -d "${tmp}/unpacked"

inner="$(find "${tmp}/unpacked" -maxdepth 3 -name config.cfg -printf '%h\n' | head -1)"
if [ -z "${inner}" ]; then
    echo "no config.cfg inside the wheel — layout changed?" >&2
    exit 1
fi
mkdir -p models
rm -rf "${DEST}"
cp -r "${inner}" "${DEST}"
echo "installed ${DEST}"
echo
echo "point the serving stack at it with:"
echo "  SPACY_UK_MODEL=${PWD}/${DEST}"
