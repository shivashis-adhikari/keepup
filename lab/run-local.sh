#!/usr/bin/env bash
# run a lab scenario against a throwaway cluster built from local postgres
# binaries, no docker needed. the data directory is deleted afterwards.
#
#   lab/run-local.sh lab/scenarios/basic.toml
#   PGBIN=/usr/lib/postgresql/17/bin lab/run-local.sh lab/scenarios/basic.toml
set -euo pipefail

scenario=${1:?usage: run-local.sh scenario.toml [out-dir]}
name=$(basename "$scenario" .toml)
out=${2:-lab/runs/$name-local}
pgbin=${PGBIN:-$(ls -d /usr/lib/postgresql/*/bin 2>/dev/null | sort -V | tail -1)}
port=${PORT:-54399}
dir=$(mktemp -d)
mkdir -p "$out"
log=$(realpath "$out")/postgres.log

cleanup() {
    "$pgbin/pg_ctl" -D "$dir/data" stop -m fast >/dev/null 2>&1 || true
    rm -rf "$dir"
}
trap cleanup EXIT

"$pgbin/initdb" -D "$dir/data" -U postgres --auth=trust >/dev/null
cat >> "$dir/data/postgresql.conf" <<CONF
port = $port
listen_addresses = ''
unix_socket_directories = '$dir'
log_autovacuum_min_duration = 0
track_io_timing = on
log_line_prefix = '%m [%p] '
shared_buffers = 256MB
CONF
"$pgbin/pg_ctl" -D "$dir/data" -l "$log" start -w >/dev/null

keepup lab run "$scenario" --dsn "host=$dir port=$port user=postgres dbname=postgres" \
    --out "$out" --log "$log"
