#!/usr/bin/env bash
# Pack the three repositories, the lexicon, the models and the artifacts for
# a move. See docs/migration.md for what each item is and how to restore.
set -eu
STAGING="${1:?usage: pack_for_move.sh <staging dir> [--with-books] [--with-cv]}"
shift || true
WITH_BOOKS=0; WITH_CV=0
for flag in "$@"; do
  case "$flag" in
    --with-books) WITH_BOOKS=1 ;;
    --with-cv) WITH_CV=1 ;;
    *) echo "unknown flag $flag"; exit 2 ;;
  esac
done
HOME_DIR=/home/devops
mkdir -p "$STAGING"
cd "$STAGING"
: > MANIFEST.txt
note() { printf '%-60s %s\n' "$1" "$2" | tee -a MANIFEST.txt; }

echo "=== git bundles (the repositories have no remote; this is the history)"
for repo in wiki-stress audiotostress uk-tts-frontend; do
  ( cd "$HOME_DIR/$repo" && git bundle create "$STAGING/$repo.bundle" --all >/dev/null 2>&1 )
  git bundle verify "$repo.bundle" >/dev/null
  note "$repo.bundle" "$(du -h "$repo.bundle" | cut -f1)  $(cd "$HOME_DIR/$repo" && git rev-parse --short HEAD), $(cd "$HOME_DIR/$repo" && git status --short | wc -l) uncommitted"
done

echo "=== the lexicon (pg_dump from the running container)"
PG=$(docker ps --format '{{.Names}}' | grep -E 'postgres' | head -1)
if [ -n "$PG" ]; then
  docker exec "$PG" pg_dump -U ukstress_owner -d ukstress -Fc > lexicon.dump
  note "lexicon.dump" "$(du -h lexicon.dump | cut -f1)  from container $PG"
else
  note "lexicon.dump" "SKIPPED: no postgres container running"
fi

echo "=== .env (secrets included)"
cp "$HOME_DIR/uk-tts-frontend/deploy/.env" env.txt
chmod 600 env.txt
note "env.txt" "edit every path on the new host"

echo "=== directories"
sync_dir() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "$dst")"
  rsync -a --info=progress2 "$src/" "$dst/"
  note "$dst" "$(du -sh "$dst" | cut -f1)"
}
sync_dir "$HOME_DIR/wiki-stress/models"          wiki-stress/models
sync_dir "$HOME_DIR/wiki-stress/output"          wiki-stress/output
sync_dir "$HOME_DIR/audiotostress/artifacts"     audiotostress/artifacts
sync_dir "$HOME_DIR/verbolizer/marian-uk-verbalizer-openspec/outputs/e39-people-counts/final" verbolizer/e39-people-counts/final
for m in models--ukr-models--xlm-roberta-base-uk models--Yehor--wav2vec2-xls-r-300m-uk-with-small-lm \
         models--mobiuslabsgmbh--faster-whisper-large-v3-turbo models--Systran--faster-whisper-large-v3 \
         models--skypro1111--m2m100-ukr-verbalization-ct2; do
  [ -d "$HOME_DIR/.cache/huggingface/hub/$m" ] && sync_dir "$HOME_DIR/.cache/huggingface/hub/$m" "huggingface/hub/$m"
done
[ "$WITH_BOOKS" = 1 ] && sync_dir "$HOME_DIR/books" books
[ "$WITH_CV" = 1 ] && sync_dir "$HOME_DIR/audiotostress/data/cv-corpus-26.0-2026-06-12" audiotostress/data/cv-corpus-26.0-2026-06-12

echo "=== checksums"
find . -type f ! -name SHA256SUMS ! -name MANIFEST.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
note "TOTAL" "$(du -sh . | cut -f1)"
echo; cat MANIFEST.txt
