#!/usr/bin/env bash
# Move the stack to another machine with the same user and home path.
# Bundles and the lexicon dump go through a small local staging directory;
# the large directories rsync straight across. Idempotent: re-run to resume.
set -eu
HOST="${1:?usage: migrate_to_host.sh user@host}"
HOME_DIR=/home/devops
STAGING="$HOME_DIR/move-staging"
mkdir -p "$STAGING"
log() { echo "=== $(date +%H:%M:%S) $*"; }

log "bundles"
for repo in wiki-stress audiotostress uk-tts-frontend; do
  ( cd "$HOME_DIR/$repo" && git bundle create "$STAGING/$repo.bundle" --all 2>/dev/null )
  git bundle verify "$STAGING/$repo.bundle" >/dev/null
done

log "lexicon dump"
if [ ! -s "$STAGING/lexicon.dump" ]; then
  PG=$(docker ps --format '{{.Names}}' | grep postgres | head -1)
  docker exec "$PG" pg_dump -U ukstress_owner -d ukstress -Fc > "$STAGING/lexicon.dump"
fi
ls -la "$STAGING/lexicon.dump" | awk '{print "    dump", $5/1e9, "GB"}'

log "env"
cp "$HOME_DIR/uk-tts-frontend/deploy/.env" "$STAGING/env.txt"; chmod 600 "$STAGING/env.txt"

log "staging -> $HOST"
rsync -a --info=progress2 "$STAGING/" "$HOST:$STAGING/"

log "large directories -> $HOST (same absolute paths)"
sync_dir() { ssh "$HOST" "mkdir -p '$(dirname "$1")'"; rsync -a --info=progress2 "$1/" "$HOST:$1/"; }
sync_dir "$HOME_DIR/wiki-stress/models"
sync_dir "$HOME_DIR/wiki-stress/output"
sync_dir "$HOME_DIR/audiotostress/artifacts"
sync_dir "$HOME_DIR/audiotostress/data/cv-corpus-26.0-2026-06-12"
sync_dir "$HOME_DIR/verbolizer/marian-uk-verbalizer-openspec/outputs/e39-people-counts/final"
for m in models--ukr-models--xlm-roberta-base-uk models--Yehor--wav2vec2-xls-r-300m-uk-with-small-lm \
         models--mobiuslabsgmbh--faster-whisper-large-v3-turbo models--Systran--faster-whisper-large-v3 \
         models--skypro1111--m2m100-ukr-verbalization-ct2; do
  sync_dir "$HOME_DIR/.cache/huggingface/hub/$m"
done
sync_dir "$HOME_DIR/books"

log "TRANSFER DONE"
