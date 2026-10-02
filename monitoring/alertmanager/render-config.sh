#!/bin/sh
# Render the Alertmanager config template, then exec Alertmanager.
#
# Alertmanager does not expand environment variables, so destinations are substituted here before
# the process starts. Two renderers are implemented because the Prometheus images are busybox-based
# and busybox does not ship envsubst, which comes from GNU gettext. envsubst is preferred when
# present; otherwise awk substitutes from ENVIRON. Both produce identical output for this template.
#
# Deliberately POSIX sh with no bashisms and no coreutils beyond awk, so it runs on whatever shell
# the image happens to provide rather than depending on a specific base image layout.

set -eu

TEMPLATE="${ALERTMANAGER_TEMPLATE:-/etc/alertmanager/alertmanager.yml.template}"
OUTPUT="${ALERTMANAGER_CONFIG:-/tmp/alertmanager.yml}"

# Defaults applied here as well as in docker-compose.yml, deliberately in two places. Compose
# covers the normal path; this covers a bare `docker run`, a CI job, or a compose file that has
# drifted, so that an unwired container always starts instead of dying on an empty destination.
# The values are RFC 2606 reserved names and the IANA discard port, so "started but delivering
# nothing" is the failure mode rather than a crash loop or mail to a stranger.
ALERT_WEBHOOK_URL="${ALERT_WEBHOOK_URL:-http://127.0.0.1:9/replace-me}"
SMTP_SMARTHOST="${SMTP_SMARTHOST:-localhost:587}"
SMTP_FROM="${SMTP_FROM:-limbot-alerts@example.invalid}"
SMTP_TO="${SMTP_TO:-limbot-alerts@example.invalid}"
SMTP_HELLO="${SMTP_HELLO:-limbot-alertmanager}"
SMTP_AUTH_USERNAME="${SMTP_AUTH_USERNAME:-}"
SMTP_AUTH_PASSWORD="${SMTP_AUTH_PASSWORD:-}"

# Escape for a YAML double-quoted scalar. Every placeholder in the template sits inside double
# quotes, and a value containing " or \ would otherwise end the scalar or start an escape and
# produce a file that does not parse. A generated SMTP password containing either is entirely
# ordinary, so this is not hypothetical.
#
# The escaping happens once, here, and both renderers then insert the already-escaped text
# verbatim. Doing it in the renderer instead would mean envsubst and awk could drift apart, since
# envsubst has no way to escape anything it substitutes.
yaml_escape() {
    printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'
}

export ALERT_WEBHOOK_URL="$(yaml_escape "$ALERT_WEBHOOK_URL")"
export SMTP_SMARTHOST="$(yaml_escape "$SMTP_SMARTHOST")"
export SMTP_FROM="$(yaml_escape "$SMTP_FROM")"
export SMTP_TO="$(yaml_escape "$SMTP_TO")"
export SMTP_HELLO="$(yaml_escape "$SMTP_HELLO")"
export SMTP_AUTH_USERNAME="$(yaml_escape "$SMTP_AUTH_USERNAME")"
export SMTP_AUTH_PASSWORD="$(yaml_escape "$SMTP_AUTH_PASSWORD")"

if [ ! -r "$TEMPLATE" ]; then
    echo "render-config: cannot read template $TEMPLATE" >&2
    echo "render-config: is ./monitoring/alertmanager mounted into the container?" >&2
    exit 1
fi

# The output must not live under the template directory. That directory is mounted read-only, so
# writing there fails with EROFS and the container crash-loops. /tmp is writable in this image.
OUT_DIR=$(dirname "$OUTPUT")
if [ ! -d "$OUT_DIR" ] || [ ! -w "$OUT_DIR" ]; then
    echo "render-config: output directory $OUT_DIR is not writable" >&2
    exit 1
fi

# Renderer selection. "auto" prefers envsubst and falls back to awk. The explicit values exist so
# the fallback can be exercised directly rather than only on a machine that happens to lack
# envsubst, and so a broken renderer can be pinned while it is fixed. Both must produce identical
# output; tests/test_alert_rules.py renders both ways and compares them.
RENDERER_MODE="${ALERTMANAGER_RENDERER:-auto}"

render_with_envsubst() {
    envsubst <"$TEMPLATE" >"$OUTPUT"
}

render_with_awk() {
    # POSIX awk: repeatedly find the next ${NAME} and splice in ENVIRON[NAME]. An unset variable
    # renders as the empty string, which is what the quoted placeholders in the template expect.
    # Values are inserted verbatim and never evaluated, so a password containing $ or a backtick
    # cannot execute anything.
    awk '
    {
      line = $0
      while (match(line, /\$\{[A-Za-z_][A-Za-z0-9_]*\}/)) {
        name = substr(line, RSTART + 2, RLENGTH - 3)
        value = (name in ENVIRON) ? ENVIRON[name] : ""
        line = substr(line, 1, RSTART - 1) value substr(line, RSTART + RLENGTH)
      }
      print line
    }
    ' "$TEMPLATE" >"$OUTPUT"
}

case "$RENDERER_MODE" in
    envsubst)
        render_with_envsubst
        ;;
    awk)
        render_with_awk
        ;;
    auto)
        if command -v envsubst >/dev/null 2>&1; then
            render_with_envsubst
        else
            render_with_awk
        fi
        ;;
    *)
        echo "render-config: ALERTMANAGER_RENDERER must be auto, envsubst or awk," >&2
        echo "render-config: not '$RENDERER_MODE'" >&2
        exit 1
        ;;
esac

# A template that failed to substitute still parses as valid YAML, because every placeholder sits
# inside quotes. Alertmanager would then start happily with an empty destination and quietly
# deliver nothing. This is the check that turns that into a visible warning instead.
if grep -q '\${[A-Za-z_][A-Za-z0-9_]*}' "$OUTPUT"; then
    echo "render-config: WARNING unsubstituted placeholders remain in $OUTPUT:" >&2
    grep -n '\${[A-Za-z_][A-Za-z0-9_]*}' "$OUTPUT" >&2 || true
fi

# Same reasoning for the shipped defaults: alerting that evaluates correctly and notifies nobody
# looks from the outside exactly like alerting that works, so say so on every start.
#
# Only real config lines are inspected. Grepping the whole file would also match the prose in the
# template's own header, which names these very addresses when explaining them, and would fire
# this warning forever even once real destinations were configured.
if grep -E '^[[:space:]]*(smtp_smarthost|smtp_from|smtp_hello|url|- url|to|- to):' "$OUTPUT" \
    | grep -Eq 'example\.invalid|127\.0\.0\.1:9'; then
    echo "render-config: WARNING notification destinations are still the reserved" >&2
    echo "render-config: WARNING placeholders (example.invalid / 127.0.0.1:9). Alerts will" >&2
    echo "render-config: WARNING be evaluated and then discarded. Set ALERT_WEBHOOK_URL," >&2
    echo "render-config: WARNING SMTP_SMARTHOST, SMTP_FROM and SMTP_TO to wire real" >&2
    echo "render-config: WARNING destinations. See README.md section 'Alerting'." >&2
fi

echo "render-config: rendered $TEMPLATE -> $OUTPUT using ${RENDERER_MODE}"

# A leading "--" is dropped rather than forwarded. Go's flag package treats a bare "--" as "stop
# parsing here" and silently ignores every argument after it, so forwarding it would drop
# --storage.path and --data.retention.time on the floor and quietly lose the silence retention that
# keeps a planned deploy from paging. Everything after the separator is still forwarded.
case "${1:-}" in
    --) shift ;;
esac

# exec so Alertmanager becomes PID 1 and receives SIGTERM directly. Without exec the shell stays
# as PID 1, swallows the signal, and the container waits out the full stop grace period on every
# deploy.
#
# Flags are passed in the order default-then-override on purpose. Go's flag package takes the last
# occurrence of a repeated flag, so the defaults above act as fallbacks for whatever a caller appends.
exec "${ALERTMANAGER_BIN:-/bin/alertmanager}" \
    --config.file="$OUTPUT" \
    --storage.path="${ALERTMANAGER_STORAGE_PATH:-/alertmanager}" \
    --data.retention.time="${ALERTMANAGER_RETENTION:-120h}" \
    --web.listen-address="${ALERTMANAGER_LISTEN_ADDRESS:-0.0.0.0:9093}" \
    "$@"