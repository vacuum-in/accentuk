#!/usr/bin/env bash
# Restart services only after the serving inventory checks out, and return
# only once every tier answers.
#   deploy/restart.sh [--fix] [service ...]     default: stress-model api
# The check runs the API's own candidate computation over every manifest form
# and compares it with what the model accepts, the inventory hash everywhere,
# and each signature against its spelling (wiki-stress: etl check-serving).
# --fix repairs the manifest (backed up) and reloads the database first.
#
# /health/ready turns green before the model service has warmed its parser,
# and until then the API quietly degrades every contextual decision to the
# dictionary default: a lang-uk run taken in that window read 58% on
# heteronyms instead of 85%. So the script waits for a probe sentence to be
# decided by the morphology tier before it declares the stack ready.
set -euo pipefail
cd "$(dirname "$0")"
set -a; . ./.env; set +a
FIX=""
if [ "${1:-}" = "--fix" ]; then FIX="--fix"; shift; fi
if [ $# -gt 0 ]; then SERVICES=("$@"); else SERVICES=(stress-model api); fi
ETL="${WIKI_STRESS:-$HOME/wiki-stress}/etl"
DB="postgresql://ukstress_owner:${POSTGRES_PASSWORD:-ukstress_owner}@127.0.0.1:${POSTGRES_PORT:-5432}/ukstress"
echo "=== check-serving"
if [ -x "$ETL/.venv/bin/python" ]; then
  CHECK=("$ETL/.venv/bin/python" -m ukstress.cli check-serving $FIX --database-url "$DB"
         --manifest "$STRESS_MODEL_DIR/serving_manifest.json" --api-manifest "$SERVING_MANIFEST"
         ${TOKEN_MODEL_DIR:+--token-model-dir "$TOKEN_MODEL_DIR"})
else
  # No ukstress checkout with a venv on this host (a deployment bundle): run
  # the check in the model image, which carries the same package, against
  # the same mounts. --fix needs a writable manifest and is not offered here.
  [ -n "$FIX" ] && { echo "--fix needs the wiki-stress checkout; running the check only" >&2; }
  CHECK=(docker compose run --rm --no-deps --entrypoint python stress-model -m ukstress.cli check-serving
         --database-url "postgresql://ukstress_owner:${POSTGRES_PASSWORD:-ukstress_owner}@postgres:5432/ukstress"
         --manifest /app/models/v3-xenc/serving_manifest.json
         ${TOKEN_MODEL_DIR:+--token-model-dir /app/models/tok-v2})
fi
if ! "${CHECK[@]}" --active-hash "$ACTIVE_INVENTORY_HASH" \
    --supplementary "${SUPPLEMENTARY_DATASETS:-}" \
    --exclusive "${REVIEWED_EXCLUSIVE_DATASETS:-13}"; then
  echo "=== check failed: not restarting. Run with --fix to repair, or fix by hand." >&2
  exit 1
fi
# The model first, then the API, so the API never starts against a cold model.
# `up --force-recreate`, not `restart`: a restart keeps the old container,
# so a rebuilt image or a changed .env would not reach it.
for service in stress-model api; do
  printf '%s\n' "${SERVICES[@]}" | grep -qx "$service" || continue
  echo "=== recreate $service"
  docker compose up -d --no-deps --force-recreate "$service"
done
others=()
for service in "${SERVICES[@]}"; do
  case "$service" in stress-model|api) ;; *) others+=("$service") ;; esac
done
[ ${#others[@]} -gt 0 ] && docker compose up -d --no-deps --force-recreate "${others[@]}"

probe() {
  curl -sf -X POST "http://127.0.0.1:${STRESS_API_PORT:-8080}/v1/stress" -H 'Content-Type: application/json' \
    -d '{"text": "Мої сестри прийшли додому."}' 2>/dev/null \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); t=[x for x in d["tokens"] if x["text"]=="сестри"]; sys.exit(0 if t and t[0].get("status") in ("morphology","combiner") and not d.get("warnings") else 1)'
}
# The probe also goes through the combiner when it is on: its first call loads
# the classifier over every form and can outlast the API's budget, and a
# warning in the answer ("combiner unavailable") means the tier order served.
for i in $(seq 1 90); do
  if probe; then echo "=== ready: the morphology tier answers"; exit 0; fi
  sleep 2
done
echo "=== not warm after 180 s: contextual tiers are not answering" >&2
exit 1
