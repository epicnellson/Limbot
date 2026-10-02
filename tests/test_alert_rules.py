"""Guard the Prometheus alert rules against the two ways they silently die.

A rule file has no compiler and no test runner of its own, so both failure modes below
produce a system that looks correctly configured and notifies nobody:

* a rule naming a metric the application never emits evaluates to an empty vector forever,
  which is indistinguishable from "no problem";
* a matcher naming a label that metric does not carry silently matches nothing.

The first of these is not hypothetical. The original specification for this work asked for
``limkobot_provider_fallback_total{provider="groq"}``, and the application emits
``limbot_provider_fallbacks_total{from_provider=...}`` — wrong prefix, singular instead of
plural, and a label name that does not exist. Written as specified, that alert would never
have fired and nobody would have noticed.

So the rule file is checked here against the live metric registry instead.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from app import metrics
from prometheus_client import Counter, Gauge, Histogram

MONITORING = Path(__file__).resolve().parents[1] / "monitoring"
RULES_FILE = MONITORING / "prometheus" / "alert_rules.yml"
# The Alertmanager config is a template, not a config: render-config.sh substitutes ${VARS} at
# container start. Every placeholder is inside quotes, so the template is still valid YAML and the
# structural assertions below work on it unmodified.
ALERTMANAGER_TEMPLATE = MONITORING / "alertmanager" / "alertmanager.yml.template"
RENDER_SCRIPT = MONITORING / "alertmanager" / "render-config.sh"
COMPOSE_FILE = Path(__file__).resolve().parents[1] / "docker-compose.yml"

# ${NAME} as written in the template. Anchored on the braced form because that is the only form
# either renderer substitutes; a bare $NAME would be left alone by both and ship a literal.
_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
# The seven variables render-config.sh substitutes into the config. Everything else in the
# script (paths, retention, renderer choice) is read by the script itself and never appears in
# the rendered YAML.
DESTINATION_VARS = frozenset(
    {
        "ALERT_WEBHOOK_URL",
        "SMTP_SMARTHOST",
        "SMTP_FROM",
        "SMTP_TO",
        "SMTP_HELLO",
        "SMTP_AUTH_USERNAME",
        "SMTP_AUTH_PASSWORD",
    }
)

# Series Prometheus itself publishes. These never appear in the application exposition, so
# they are allowed without being declared as application metrics.
PROMETHEUS_INTERNAL = frozenset({"up", "scrape_duration_seconds", "scrape_samples_scraped"})

# Any bare identifier in an expression is a candidate metric name. Identifying candidates
# generically rather than only looking for a `limbot_` prefix is deliberate: the original
# specification for this work asked for `limkobot_provider_fallback_total`, and a check that
# only recognised correctly-prefixed names would report "no metrics referenced" for that
# expression and pass, which is precisely the bug it exists to catch.
_IDENTIFIER_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")
# `{result="accepted",provider!="none"}` — label matchers, not metric names.
_MATCHER_BLOCK_RE = re.compile(r"\{[^{}]*\}")
# by/le, on(job), without(...), ignoring(...), group_left(...) — label name lists.
_GROUPING_RE = re.compile(r"\b(?:by|on|without|ignoring|group_left|group_right)\s*\([^()]*\)")
_STRING_RE = re.compile(r"\"[^\"]*\"|'[^']*'")
# A label matcher, e.g. result="accepted" or provider!~"groq|gemini".
_MATCHER_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:=~|!~|!=|=)\s*(?:\"[^\"]*\"|'[^']*')")
# PromQL functions and keywords. Anything left over after these are removed is a metric name.
_PROMQL = frozenset(
    {
        "absent",
        "absent_over_time",
        "and",
        "avg",
        "bool",
        "bottomk",
        "by",
        "clamp_max",
        "clamp_min",
        "count",
        "count_values",
        "delta",
        "end",
        "group",
        "group_left",
        "group_right",
        "histogram_quantile",
        "idelta",
        "ignoring",
        "increase",
        "label_replace",
        "le",
        "max",
        "min",
        "offset",
        "on",
        "or",
        "quantile",
        "rate",
        "start",
        "step",
        "stddev",
        "stdvar",
        "sum",
        "topk",
        "unless",
        "without",
    }
)


def _exposition() -> dict[str, set[str]]:
    """Metric name -> label names, for every metric the application declares.

    Introspects the collectors in ``app.metrics`` rather than parsing the rendered exposition,
    because ``prometheus_client`` only emits a series once it has been observed at least
    once. A counter that has never incremented is absent from ``/metrics``, so reading the
    exposition would make exactly the metrics this file exists to watch — an outage counter
    that has not fired yet — look undefined.
    """
    parsed: dict[str, set[str]] = {}
    for value in vars(metrics).values():
        if isinstance(value, Counter):
            base, labels, suffixes = value._name, value._labelnames, ("_total",)
        elif isinstance(value, Gauge):
            base, labels, suffixes = value._name, value._labelnames, ()
        elif isinstance(value, Histogram):
            base, labels = value._name, value._labelnames
            suffixes = ("_bucket", "_sum", "_count")
            labels = (*labels, "le")
        else:
            continue
        for suffix in suffixes or ("",):
            parsed.setdefault(base + suffix, set()).update(labels)
    return parsed


def _load_rules() -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for group in yaml.safe_load(RULES_FILE.read_text(encoding="utf-8"))["groups"]:
        for rule in group["rules"]:
            rules.append({**rule, "group": group["name"]})
    return rules


def _referenced_metrics(expr: str) -> set[str]:
    """Metric names referenced by a PromQL expression.

    Label matchers, grouping clauses and string literals are stripped first, so what remains
    is metric names and nothing else. Histogram suffixes are kept, because a rule selecting
    ``_bucket`` is selecting a real series and it should be checked like any other.
    """
    stripped = _MATCHER_BLOCK_RE.sub(" ", expr)
    stripped = _GROUPING_RE.sub(" ", stripped)
    stripped = _STRING_RE.sub(" ", stripped)
    return {token for token in _IDENTIFIER_RE.findall(stripped) if token not in _PROMQL}


RULES = _load_rules()
EXPOSITION = _exposition()


def test_the_rule_file_parses_and_is_not_empty() -> None:
    assert RULES, "alert_rules.yml parsed to zero rules"
    assert all(rule.get("alert") for rule in RULES), "every entry needs an alert name"


def test_the_four_required_alerts_are_present() -> None:
    """The alert names this work was commissioned for must all exist."""
    names = {rule["alert"] for rule in RULES}
    required = {
        "AllProvidersExhausted",
        "PrimaryLLMDown",
        "HighLatency",
        "WebhookVerificationFailures",
    }
    assert required <= names, f"missing required alerts: {sorted(required - names)}"


def test_the_guard_would_catch_the_bug_it_was_written_for() -> None:
    """Negative control. A guard that cannot fail is not a guard.

    This work was specified against metric names the application does not emit. Written as
    specified, every alert would have evaluated to an empty vector forever and looked
    configured. These cases assert the detection logic actually rejects each of those
    mistakes, and still accepts the expression that is correct, so the guard cannot be
    weakened into always-passing.
    """
    should_reject = {
        "wrong prefix": 'rate(limkobot_provider_fallback_total{provider="groq"}[5m]) > 0',
        "singular instead of plural": "rate(limbot_provider_fallback_total[5m]) > 0",
        "typo in the metric name": "rate(limbot_all_providers_exhausted_totals[5m]) > 0",
    }
    for name, expr in should_reject.items():
        unknown = _referenced_metrics(expr) - set(EXPOSITION) - PROMETHEUS_INTERNAL
        assert unknown, f"the guard failed to reject {name}: {expr}"

    correct = 'sum(rate(limbot_provider_fallbacks_total{from_provider="groq"}[5m])) > 0'
    assert not _referenced_metrics(correct) - set(EXPOSITION) - PROMETHEUS_INTERNAL, (
        "the guard rejected a correct expression, so it would block legitimate rules"
    )


@pytest.mark.parametrize("rule", RULES, ids=lambda r: str(r["alert"]))
def test_every_referenced_metric_is_actually_exposed(rule: dict[str, Any]) -> None:
    """The check that catches the wrong-prefix, wrong-plurality, wrong-name class of bug."""
    referenced = _referenced_metrics(rule["expr"])
    unknown = referenced - set(EXPOSITION) - PROMETHEUS_INTERNAL
    assert not unknown, (
        f"{rule['alert']} references metrics that do not exist: {sorted(unknown)}. "
        f"A rule naming a missing metric evaluates to an empty vector forever, which looks "
        f"identical to a healthy system. Known metrics: {sorted(EXPOSITION)}"
    )


@pytest.mark.parametrize("rule", RULES, ids=lambda r: str(r["alert"]))
def test_every_label_matcher_exists_on_the_metrics_it_filters(rule: dict[str, Any]) -> None:
    """A matcher on a label the metric lacks matches nothing, silently."""
    referenced = _referenced_metrics(rule["expr"]) & set(EXPOSITION)
    if not referenced:
        return
    known_labels = set().union(*(EXPOSITION[name] for name in referenced)) | {"job", "instance"}
    matchers = {
        name
        for name in _MATCHER_RE.findall(rule["expr"])
        if name not in _PROMQL and not name.startswith("__")
    }
    unknown = matchers - known_labels
    assert not unknown, (
        f"{rule['alert']} filters on labels absent from its metrics: {sorted(unknown)}. "
        f"Present on them: {sorted(known_labels)}"
    )


@pytest.mark.parametrize("rule", RULES, ids=lambda r: str(r["alert"]))
def test_every_alert_is_actionable(rule: dict[str, Any]) -> None:
    """An alert without a severity, summary and runbook cannot be routed or acted on."""
    assert rule.get("for") is not None, f"{rule['alert']} has no for: clause"
    labels = rule.get("labels", {})
    assert labels.get("severity") in {"critical", "warning", "info"}, (
        f"{rule['alert']} has severity {labels.get('severity')!r}, "
        "which Alertmanager has no route for"
    )
    annotations = rule.get("annotations", {})
    assert annotations.get("summary"), f"{rule['alert']} has no summary"
    assert annotations.get("runbook_url"), f"{rule['alert']} has no runbook_url"


def test_exhaustion_is_not_derived_from_the_fallback_counter() -> None:
    """Pin the decision that exhaustion needed its own metric.

    PROVIDER_FALLBACKS deliberately stops counting one tier before the end of the chain, so a
    query built on it cannot distinguish "the backup saved this request" from "nothing could
    answer". If this ever regresses, this test explains why the separate counter still has to
    exist rather than letting someone delete it as redundant.
    """
    exhaustion_rules = [r for r in RULES if r["alert"] == "AllProvidersExhausted"]
    assert exhaustion_rules, "AllProvidersExhausted disappeared"
    expr = exhaustion_rules[0]["expr"]
    assert "limbot_all_providers_exhausted_total" in expr
    assert "limbot_provider_fallbacks_total" not in expr, (
        "exhaustion must come from its own counter: the fallback counter stops one tier early "
        "by design, so it cannot represent a total outage"
    )


def test_the_latency_alert_watches_completion_not_the_acknowledgement() -> None:
    """Guard the most consequential judgement call in the file.

    The webhook route acknowledges Meta and returns before any AI work happens. HTTP latency
    there is therefore always small, so an alert on it would stay green while students waited
    for answers. If someone "simplifies" this back to an HTTP histogram the alert silently
    becomes useless, which is what this test is for.
    """
    high_latency = next(r for r in RULES if r["alert"] == "HighLatency")
    assert "limbot_llm_call_duration_seconds" in high_latency["expr"]
    assert "limbot_http_request_duration_seconds" not in high_latency["expr"]
    assert "4.0" in high_latency["expr"], "the 4.0s threshold was requested; do not drop it"


def test_a_higher_traffic_alert_inhibits_the_ones_it_makes_meaningless() -> None:
    """TargetDown silences the rest, because a dead target makes every other rule blind."""
    config = yaml.safe_load(ALERTMANAGER_TEMPLATE.read_text(encoding="utf-8"))
    inhibits = config.get("inhibit_rules", [])
    target_down = [r for r in inhibits if "TargetDown" in str(r.get("source_matchers"))]
    assert target_down, (
        "TargetDown must inhibit the other alerts: when the target is not scraped there is no "
        "data, so the remaining rules would all sit quiet and make a full outage look calm"
    )
    assert any(r["target_matchers"] for r in target_down), (
        "the TargetDown inhibition needs target_matchers to do anything"
    )


def test_every_receiver_referenced_by_a_route_exists() -> None:
    """A route pointing at a receiver that was never defined drops alerts on the floor."""
    config = yaml.safe_load(ALERTMANAGER_TEMPLATE.read_text(encoding="utf-8"))
    defined = {receiver["name"] for receiver in config.get("receivers", [])}
    referenced = {config["route"]["receiver"]}
    for route in config["route"].get("routes", []):
        referenced.add(route["receiver"])
    missing = referenced - defined
    assert not missing, f"routes point at undefined receivers: {sorted(missing)}"


# ---------------------------------------------------------------------------------------------
# The template contract.
#
# render-config.sh is the only thing standing between a typo in the template and a container that
# boots, loads every alert, and notifies nobody. These tests pin the seam: what the template asks
# for, what the script promises to supply, and what docker-compose passes in.
# ---------------------------------------------------------------------------------------------


def test_template_placeholders_are_all_braced() -> None:
    """A bare $NAME is invisible to envsubst's default variable list and to the awk fallback.

    envsubst without a shell-format argument only substitutes the braced form, and the awk
    renderer matches on `\\$\\{...\\}` for the same reason. So a `$SMTP_FROM` typed without braces
    would survive both renderers verbatim, and the rendered file would contain the literal text
    "$SMTP_FROM" as a URL. Nothing would fail: the container would start and deliver nowhere.
    """
    template = ALERTMANAGER_TEMPLATE.read_text(encoding="utf-8")
    # Strip comment lines, since the header documents these variables in prose.
    body = "\n".join(line for line in template.splitlines() if not line.lstrip().startswith("#"))
    bare = re.findall(r"\$(?!\{)([A-Za-z_][A-Za-z0-9_]*)", body)
    assert not bare, (
        f"unbraced $VAR references in the template: {sorted(set(bare))}. Only ${{VAR}} is "
        f"substituted by either renderer, so these would ship as literal text."
    )


def test_every_template_placeholder_is_supplied_by_the_render_script() -> None:
    """A placeholder the script never sets renders as an empty string, silently.

    An empty destination is the specific failure this whole arrangement exists to prevent:
    Alertmanager accepts it, starts, and discards every notification.
    """
    template = ALERTMANAGER_TEMPLATE.read_text(encoding="utf-8")
    script = RENDER_SCRIPT.read_text(encoding="utf-8")
    used = set(_PLACEHOLDER_RE.findall(template))
    assert used, "no placeholders found in the template; has templating been reverted?"

    # Every var the script exports with a default counts as supplied, including the auth pair
    # whose default is deliberately the empty string (an open relay needs no credentials).
    supplied = set(re.findall(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=", script, re.MULTILINE))
    missing = used - supplied
    assert not missing, (
        f"the template uses {sorted(missing)} but render-config.sh never exports them, so they "
        f"render as empty strings. Add them to the script's export block with a default."
    )


def test_the_render_script_substitutes_only_known_destination_vars() -> None:
    """The script's export list and the documented variable set must not drift apart.

    An export the template never reads is a dead setting that looks configured in .env.example;
    a template placeholder with no export is the empty-destination bug above. Comparing both
    directions catches either.
    """
    script = RENDER_SCRIPT.read_text(encoding="utf-8")
    exported = set(re.findall(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=", script, re.MULTILINE))
    assert exported == set(DESTINATION_VARS), (
        f"render-config.sh exports {sorted(exported)} but the documented destination set is "
        f"{sorted(DESTINATION_VARS)}. Update .env.example and this test together, or the "
        f"variable a user sets will be silently ignored."
    )


def test_compose_passes_every_destination_var_to_the_container() -> None:
    """A variable added to .env.example but not to compose's environment block never arrives.

    compose does not forward arbitrary variables into a container; only what `environment:` lists
    does. So forgetting one line here is indistinguishable from not setting it, and the container
    falls back to the placeholder and warns. Worth failing on.
    """
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    env = compose["services"]["alertmanager"]["environment"]
    missing = sorted(DESTINATION_VARS - set(env))
    assert not missing, (
        f"docker-compose.yml does not pass {missing} to the alertmanager container, so setting "
        f"them in .env would have no effect."
    )


def test_compose_renders_to_a_writable_path_outside_the_read_only_mount() -> None:
    """Rendering into the bind mount would fail on every start with a permission error.

    ./monitoring/alertmanager is mounted read-only, so the rendered output has to land somewhere
    writable. If it regressed to the mount, the container would crash-loop rather than alert.
    """
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    service = compose["services"]["alertmanager"]
    config_path = service["environment"]["ALERTMANAGER_CONFIG"]
    assert not config_path.startswith("/etc/alertmanager"), (
        f"ALERTMANAGER_CONFIG is {config_path}, inside the read-only mount. It must be a "
        f"writable path such as /tmp/alertmanager.yml."
    )

    mount = next(v for v in service["volumes"] if "/etc/alertmanager" in v)
    assert ":ro" in mount, "the template mount should stay read-only"

    command = service["command"]
    assert command == ["/etc/alertmanager/render-config.sh"], (
        f"alertmanager must be started via render-config.sh, not directly: {command}"
    )


def test_the_rendered_config_from_defaults_cannot_deliver_anywhere() -> None:
    """The default deployment must fail closed, never open.

    Renders the template exactly as an unwired deployment would, with no destination set, and
    requires every resulting destination to be provably undeliverable. Anything routable here is a
    committed address that would page a stranger on day one.
    """
    rendered = _render(env={})
    config = yaml.safe_load(rendered)
    destinations = _collect_destinations(config)

    assert destinations, "no notification destinations are configured at all"
    for destination in destinations:
        assert destination.strip(), "a destination rendered empty"
    routable = [d for d in destinations if not _is_provably_undeliverable(d)]
    assert not routable, (
        f"the shipped defaults route to {routable}. They must use an RFC 2606 reserved name "
        f"(example.invalid) or the IANA discard port 127.0.0.1:9, so an unwired deployment "
        f"starts and discards alerts instead of paging someone."
    )


def test_an_operator_can_wire_a_real_destination_through_the_environment() -> None:
    """Proves the substitution is genuinely wired, not just inert by accident.

    An implementation that ignored its environment variables entirely would pass the
    fail-closed test above, because the placeholders would never be replaced and nothing would
    ever be delivered. So render with real values and require them to appear.
    """
    env = {
        "ALERT_WEBHOOK_URL": "https://hooks.example.test/services/T00/B00/token",
        "SMTP_FROM": "limbot-alerts@corp.example.test",
        "SMTP_TO": "oncall@corp.example.test",
        "SMTP_SMARTHOST": "smtp.corp.example.test:587",
    }
    rendered = _render(env=env)
    config = yaml.safe_load(rendered)

    assert config["global"]["smtp_from"] == "limbot-alerts@corp.example.test"
    urls = [
        hook["url"]
        for receiver in config["receivers"]
        for hook in receiver.get("webhook_configs", [])
    ]
    assert "https://hooks.example.test/services/T00/B00/token" in urls

    # Scoped to config lines: the template's header documents the ${...} syntax in prose, and a
    # bare assertion on the whole file would trip over that documentation.
    body = "\n".join(line for line in rendered.splitlines() if not line.lstrip().startswith("#"))
    assert "${" not in body, "a placeholder survived rendering"


def test_a_password_with_shell_metacharacters_survives_substitution() -> None:
    """Values are inserted verbatim; nothing re-evaluates them.

    A password containing $, &, # or a backtick is normal in a generated secret. If the renderer
    expanded or interpreted the value on the way in, the stored password would not match the one
    the mail server expects and auth would fail in a way that looks like a wrong password.
    """
    nasty = 'p@ss$w&rd#1`x`y"z'
    rendered = _render(env={"SMTP_AUTH_PASSWORD": nasty})
    config = yaml.safe_load(rendered)
    assert config["global"]["smtp_auth_password"] == nasty


def test_go_templates_survive_substitution() -> None:
    """Alertmanager's own `{{ }}` templates must not be mistaken for placeholders.

    Both renderers key on `${...}`, and the config's `{{ .Status }}` expressions use a different
    delimiter, so this should hold. Asserted because the two syntaxes share a file and a future
    renderer that keys on `{{` as well would silently blank out every subject line.
    """
    rendered = _render(env={"SMTP_TO": "oncall@corp.example.test"})
    assert "{{ .Status | toUpper }}" in rendered
    assert "{{ .GroupLabels.alertname }}" in rendered


def test_no_literal_destination_is_committed_in_the_template() -> None:
    """Destinations belong in the environment, so the template must contain none.

    This is the whole reason for the template. A hardcoded webhook URL or mail address in a
    tracked file is a leak that outlives every attempt to rotate it, so the template is only
    allowed to hold ${VAR} references.
    """
    template = ALERTMANAGER_TEMPLATE.read_text(encoding="utf-8")
    body = "\n".join(line for line in template.splitlines() if not line.lstrip().startswith("#"))
    config = yaml.safe_load(body)

    # Substitute nothing: any destination that is still non-empty here is a hardcoded one.
    for value in _collect_destinations(config):
        assert value.startswith("${") or value == "", (
            f"the template hardcodes the destination {value!r}. Put it in .env and reference it "
            f"as ${{VAR}} instead."
        )


def _collect_destinations(config: dict[str, Any]) -> list[str]:
    """Every address Alertmanager would try to notify, in the template or a rendered config."""
    destinations = [config["global"].get("smtp_from", "")]
    destinations += [
        hook["url"]
        for receiver in config.get("receivers", [])
        for hook in receiver.get("webhook_configs", [])
    ]
    destinations += [
        target
        for receiver in config.get("receivers", [])
        for mail in receiver.get("email_configs", [])
        for target in ([mail["to"]] if isinstance(mail["to"], str) else list(mail["to"]))
    ]
    return destinations


def _render(env: dict[str, str]) -> str:
    """Render the template the way render-config.sh does, for the tests above.

    The shell script is exercised directly in tests/test_render_config_script.py, which skips
    where no POSIX shell exists. This in-Python renderer exists so the config contract is checked
    on every platform: it mirrors the script's defaulting and YAML-escaping, after which both
    renderers insert the text verbatim.
    """
    script = RENDER_SCRIPT.read_text(encoding="utf-8")
    # Matches both the `VAR="${VAR:-default}"` defaults and an `export VAR=` of the same name, so
    # this survives the script's current two-stage assign-then-export shape.
    defaults = dict(
        re.findall(
            r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=["\']?\$\{\1:-([^}]*)\}["\']?$',
            script,
            re.MULTILINE,
        )
    )

    resolved = {name: _yaml_escape(value) for name, value in defaults.items()}
    resolved.update({name: _yaml_escape(value) for name, value in env.items()})

    template = ALERTMANAGER_TEMPLATE.read_text(encoding="utf-8")
    return _PLACEHOLDER_RE.sub(lambda m: resolved.get(m.group(1), ""), template)


def _yaml_escape(value: str) -> str:
    """Escape a value for a YAML double-quoted scalar, matching yaml_escape in the script."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _is_provably_undeliverable(destination: str) -> bool:
    """True when a destination cannot route anywhere, whatever DNS or routing says."""
    reserved_hosts = ("example.invalid", "example.test", "example.com", "127.0.0.1", "localhost")
    return any(host in destination for host in reserved_hosts)
