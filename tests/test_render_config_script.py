"""Exercise render-config.sh as a real process.

The config contract in test_alert_rules.py is checked against an in-Python renderer that mirrors
the script. That mirror can drift from the script and stay green, which is exactly the failure
mode worth ruling out, so these tests run the actual shell script and compare.

Skipped where no POSIX shell is available. On Windows it uses Git Bash if it is installed, since
CI runs on Linux and the container image is Linux, so the Linux path is what matters and the
Windows run is a convenience.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

MONITORING = Path(__file__).resolve().parents[1] / "monitoring"
TEMPLATE = MONITORING / "alertmanager" / "alertmanager.yml.template"
SCRIPT = MONITORING / "alertmanager" / "render-config.sh"


def _find_sh() -> str | None:
    """A POSIX shell, preferring one that also has awk and sed."""
    found = shutil.which("sh")
    if found:
        return found
    for candidate in (
        r"C:\Program Files\Git\bin\sh.exe",
        r"C:\Program Files (x86)\Git\bin\sh.exe",
    ):
        if Path(candidate).is_file():
            return candidate
    return None


SHELL = _find_sh()
requires_shell = pytest.mark.skipif(SHELL is None, reason="no POSIX shell available")

# Stands in for /bin/alertmanager and records its argv, one argument per line. Redirects fd 1 to
# $ARGV_OUT when set, so a caller can read argv back without a shell redirect embedded in
# ALERTMANAGER_BIN, which would break on the spaces in the repository path. /dev/stdout is not
# used because it does not exist under Git Bash.
_STUB = """#!/bin/sh
if [ -n "${ARGV_OUT:-}" ]; then
    exec > "$ARGV_OUT"
fi
for arg in "$@"; do echo "$arg"; done
"""


def run_script(
    tmp_path: Path,
    env: dict[str, str],
    renderer: str = "awk",
    args: list[str] | None = None,
    template: Path = TEMPLATE,
) -> subprocess.CompletedProcess[str]:
    """Run render-config.sh with a stubbed binary so it never starts a real Alertmanager."""
    stub = tmp_path / "stub-alertmanager"
    stub.write_text(_STUB, encoding="utf-8")
    stub.chmod(0o755)

    return subprocess.run(
        [SHELL, str(SCRIPT), *(args or [])],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ALERTMANAGER_BIN": str(stub),
            "ALERTMANAGER_TEMPLATE": str(template),
            "ALERTMANAGER_CONFIG": str(tmp_path / "rendered.yml"),
            "ALERTMANAGER_RENDERER": renderer,
            **env,
        },
        timeout=30,
    )


def stubbed_argv(tmp_path: Path, args: list[str]) -> list[str]:
    """Run the script with a stub that records argv, and return the arguments it saw."""
    stub = tmp_path / "stub-alertmanager"
    stub.write_text(_STUB, encoding="utf-8")
    stub.chmod(0o755)
    argv_path = tmp_path / "argv.txt"

    result = subprocess.run(
        [SHELL, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ALERTMANAGER_BIN": str(stub),
            "ALERTMANAGER_TEMPLATE": str(TEMPLATE),
            "ALERTMANAGER_CONFIG": str(tmp_path / "rendered.yml"),
            "ALERTMANAGER_RENDERER": "awk",
            "ARGV_OUT": str(argv_path),
        },
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert argv_path.is_file(), "the stub was never reached"
    return argv_path.read_text(encoding="utf-8").splitlines()


def render(tmp_path: Path, env: dict[str, str], renderer: str = "awk") -> str:
    """Render and return the config, asserting the script succeeded."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    result = run_script(tmp_path, env, renderer=renderer)
    assert result.returncode == 0, f"script failed: {result.stderr}"
    rendered = tmp_path / "rendered.yml"
    assert rendered.is_file(), "nothing was rendered"
    return rendered.read_text(encoding="utf-8")


def _has_envsubst() -> bool:
    """Whether this shell can run envsubst.

    The busybox container cannot, so the awk path is the one that matters in production and must
    always be tested; the envsubst path is a developer-machine convenience that is skipped rather
    than failed when gettext is not installed.
    """
    if shutil.which("envsubst"):
        return True
    probe = subprocess.run(
        [SHELL, "-c", "command -v envsubst"], capture_output=True, text=True, timeout=30
    )
    return probe.returncode == 0


pytestmark = requires_shell

REAL_ENV = {
    "ALERT_WEBHOOK_URL": "https://hooks.example.test/services/T00/B00/token",
    "SMTP_SMARTHOST": "smtp.example.test:587",
    "SMTP_FROM": "limbot-alerts@corp.example.test",
    "SMTP_TO": "oncall@corp.example.test",
    "SMTP_HELLO": "limbot",
    "SMTP_AUTH_USERNAME": "limbot",
    "SMTP_AUTH_PASSWORD": "s3cret",
}


@requires_shell
@pytest.mark.parametrize("renderer", ["awk", "envsubst"])
def test_renders_with_both_renderers(renderer: str, tmp_path: Path) -> None:
    """Both renderers must work; envsubst is preferred when present, awk is the container path."""
    if renderer == "envsubst" and not _has_envsubst():
        pytest.skip("envsubst is not available to this shell")

    output = render(tmp_path, REAL_ENV, renderer=renderer)
    assert "https://hooks.example.test/services/T00/B00/token" in output
    assert "limbot-alerts@corp.example.test" in output


@requires_shell
def test_the_two_renderers_agree(tmp_path: Path) -> None:
    """awk and envsubst must produce the same config.

    The script prefers envsubst on a developer machine and silently falls back to awk in the
    busybox container, so a divergence would mean the config that gets tested locally is not the
    config that ships. Compared line by line rather than byte for byte because envsubst built for
    Windows emits CRLF; inside the Linux container both emit LF.
    """
    if not _has_envsubst():
        pytest.skip("envsubst is not available to this shell")

    from_awk = render(tmp_path / "awk", REAL_ENV, renderer="awk")
    from_envsubst = render(tmp_path / "env", REAL_ENV, renderer="envsubst")
    assert from_awk.splitlines() == from_envsubst.splitlines(), (
        "the renderers disagree; the config tested locally is not the config that ships"
    )


@requires_shell
def test_defaults_render_reserved_placeholders_and_warn(tmp_path: Path) -> None:
    """An unwired deployment must boot, warn, and deliver nowhere.

    Asserting on the warning matters as much as on the values: an operator who never pages anyone
    will assume the rules were quiet rather than undeliverable, so this log line is the only
    signal that alerts are going nowhere.
    """
    result = run_script(tmp_path, {})
    assert result.returncode == 0, result.stderr
    output = (tmp_path / "rendered.yml").read_text(encoding="utf-8")
    assert "http://127.0.0.1:9/replace-me" in output
    assert "example.invalid" in output
    assert "WARNING" in result.stderr, (
        "the placeholder warning is the only clue that alerts are going nowhere"
    )


@requires_shell
def test_no_warning_when_destinations_are_real(tmp_path: Path) -> None:
    """A permanently-firing warning trains people to ignore the one that matters.

    The first version grepped the whole rendered file, which also matched the template's own
    comments naming the reserved addresses, so this fired forever even once real destinations
    were configured.
    """
    result = run_script(tmp_path, REAL_ENV)
    assert result.returncode == 0, result.stderr
    assert "WARNING" not in result.stderr, "a placeholder warning fired with real destinations set"


@requires_shell
def test_a_leading_separator_is_not_forwarded_to_alertmanager(tmp_path: Path) -> None:
    """Go's flag package stops parsing at a bare "--" and silently drops everything after it.

    Forwarding it would discard --storage.path and --data.retention.time, losing the silence
    retention that is the only thing stopping a planned deploy from paging.
    """
    argv = stubbed_argv(tmp_path, ["--", "--cluster.peer=alertmanager-1"])
    assert "--" not in argv, "a bare -- reached the binary and would stop flag parsing"
    assert "--cluster.peer=alertmanager-1" in argv, "arguments after the separator were dropped"
    assert any(line.startswith("--config.file=") for line in argv), (
        "the script's own flags did not reach the binary"
    )


@requires_shell
def test_alertmanager_flags_are_passed_in_default_then_override_order(tmp_path: Path) -> None:
    """Go takes the last occurrence of a repeated flag, so a caller's override must come last."""
    argv = stubbed_argv(tmp_path, ["--", "--data.retention.time=720h"])
    retentions = [line for line in argv if line.startswith("--data.retention.time=")]
    assert retentions, "the retention flag never reached the binary"
    assert retentions[-1] == "--data.retention.time=720h", (
        f"the caller's value must be last to win, got {retentions}"
    )


@requires_shell
@pytest.mark.parametrize("renderer", ["perl", "AUTO", "awk -v x=1"])
def test_an_unknown_renderer_fails_before_start(renderer: str, tmp_path: Path) -> None:
    """A typo in ALERTMANAGER_RENDERER must stop the container, not silently pick a renderer.

    Falling back to the default here would mean a misconfigured value looks honoured, and the
    operator has no way to tell which renderer actually ran.
    """
    result = run_script(tmp_path, {}, renderer=renderer)
    assert result.returncode != 0, f"the script accepted renderer={renderer!r}"
    assert "must be auto, envsubst or awk" in result.stderr
    assert not (tmp_path / "rendered.yml").exists(), (
        "a config was written despite an unusable renderer setting"
    )


@requires_shell
def test_an_empty_renderer_setting_means_auto_not_an_error(tmp_path: Path) -> None:
    """Compose renders an unset optional variable as an empty string, so empty must mean auto.

    docker-compose.yml writes `${ALERTMANAGER_RENDERER:-auto}`, but a value can still arrive
    empty from a .env line like `ALERTMANAGER_RENDERER=`. Treating that as an error would make a
    container fail to start for a setting the operator never touched, and since the whole point of
    the default is that it is optional, that would be the wrong trade.
    """
    result = run_script(tmp_path, {}, renderer="")
    assert result.returncode == 0, result.stderr
    assert "using auto" in result.stdout


@requires_shell
def test_a_missing_template_fails_loudly_instead_of_starting_blind(tmp_path: Path) -> None:
    """Without the template the container must not start.

    A container that is visibly down is recoverable; one that runs and alerts nobody is not,
    because nothing anywhere reports the failure.
    """
    result = run_script(tmp_path, {}, template=tmp_path / "absent.yml.template")
    assert result.returncode != 0
    assert "cannot read template" in result.stderr
    assert not (tmp_path / "rendered.yml").exists(), (
        "a config was produced despite the missing template"
    )


@requires_shell
def test_an_unwritable_output_directory_fails_loudly(tmp_path: Path) -> None:
    """Rendering into the read-only bind mount would crash-loop on every start."""
    stub = tmp_path / "stub-alertmanager"
    stub.write_text(_STUB, encoding="utf-8")
    stub.chmod(0o755)
    bad = tmp_path / "no" / "such" / "dir" / "rendered.yml"

    result = subprocess.run(
        [SHELL, str(SCRIPT)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ALERTMANAGER_BIN": str(stub),
            "ALERTMANAGER_TEMPLATE": str(TEMPLATE),
            "ALERTMANAGER_CONFIG": str(bad),
            "ALERTMANAGER_RENDERER": "awk",
        },
        timeout=30,
    )
    assert result.returncode != 0
    assert "is not writable" in result.stderr
    assert not bad.exists()


@requires_shell
def test_the_script_is_valid_posix_shell() -> None:
    """`sh -n` parses without executing, catching syntax only another shell would accept.

    The image is busybox and a developer's shell is bash, so a bashism would pass every local run
    and fail only inside the container.
    """
    shell = shutil.which("dash") or SHELL
    result = subprocess.run([shell, "-n", str(SCRIPT)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"{shell} rejects render-config.sh: {result.stderr}"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
