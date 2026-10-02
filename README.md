# Limbot

WhatsApp Cloud API bot platform. A student-facing assistant that answers over a
multi-provider AI pipeline, optionally grounded in course material it retrieves from Qdrant,
and backed by read-only access to a student database when a question needs someone's real
records.

## What is in place

| Area | Detail |
| --- | --- |
| FastAPI app | `app/main.py` factory, lifespan-owned runtime, raw ASGI metrics middleware, global error handler |
| Webhooks | `GET` subscription handshake, `POST` deliveries with `X-Hub-Signature-256` verification, immediate 200 |
| Async work | `app/core/tasks.py` — detached tasks, bounded concurrency, graceful drain on shutdown |
| Outbound HTTP | One `httpx.AsyncClient` per process, connection pooling, timeouts, retry policy that respects idempotency |
| LLM pipeline | `app/llm/` — Groq, Gemini and an OpenAI-compatible tier, per-tier circuit breakers, tool calling, per-request fallback |
| Tools | Read-only student records (timetable, deadlines, exams, grades, submissions) via Postgres, injected context |
| RAG | `app/rag/` — chunking, `fastembed` embeddings, Qdrant retrieval with a score threshold, `/debug/knowledge` ingest |
| Conversation memory | `app/conversation/` — recent turns per student, fed back into the prompt |
| Observability | `prometheus_client` registry, `/metrics`, circuit / LLM / tool / RAG / eval series, provisioned Grafana dashboard |
| Evaluation | `app/eval/` — 15 synthetic cases graded in CI on every prompt change, provider distribution reported |
| CI/CD | GitHub Actions — ruff/mypy/pytest, GHCR image, synthetic evals, Fly.io deploy |
| Tests | Signature verification, webhook acknowledgement, task bounds, retry policy, circuit breakers, providers, tools, RAG, ingest, readiness |

## Layout

```
app/
  main.py                 app factory, lifespan wiring, middleware
  config.py               pydantic-settings, multi-tier validation, fail-fast production checks
  runtime.py              AiRuntime: builds pipeline, tools, retrieval, ingestor, database
  metrics.py              Prometheus registry and all series
  logging_config.py       single stdout handler, JSON or text
  core/
    http.py               shared client, retry policy, throttling metrics
    tasks.py              bounded background task manager
    security.py           webhook signature verification
    whatsapp.py           Cloud API client
    circuit.py            per-provider circuit breaker
    embeddings.py         fastembed wrapper (lazy import)
    qdrant.py             vector store handle, health probe, upsert/search
  llm/                    types, base client, openai_compat, gemini, pipeline/fallback
  rag/                    chunker, retrieval pipeline, knowledge ingestor
  tools/                  JSON-schema tool calling, registry, student tools
  db/                     pooled asyncpg connection, read-only student repository
  conversation/           per-student recent turn memory
  models/whatsapp.py      webhook payload schemas
  services/
    message_handler.py    inbound message processing and routing
    answer.py             orchestrates retrieval + tools + LLM into a reply
    dedupe.py             TTL message id cache
  api/routes/             health, webhooks, metrics, debug, knowledge
  middleware/metrics.py   latency, request counters, access log
db/                       schema.sql + seed.sql applied by the postgres container
infra/
  prometheus/prometheus.yml
  grafana/provisioning/   datasource + dashboard providers
  grafana/dashboards/limbot-overview.json
tests/
```

## Run the stack

```bash
cp .env.example .env
# fill in WHATSAPP_APP_SECRET, plus at least one provider key (GROQ_API_KEY or GEMINI_API_KEY), then:
docker compose up --build
```

A local LLM tier is available but off by default:

```bash
docker compose --profile ollama up --build
# pull a model inside the container, then set TIER3_ENABLED=true in .env
docker compose exec ollama ollama pull llama3.2
```

| Service | URL | Notes |
| --- | --- | --- |
| web | <http://localhost:8000> | `/docs` unless `ENVIRONMENT=production` |
| health / readiness | `/healthz`, `/readyz` | `/readyz` reports provider + store state |
| metrics | <http://localhost:8000/metrics> | scraped every 5s |
| Prometheus | <http://localhost:9090> | targets `web:8000` and `qdrant:6333` |
| Alertmanager | not published | internal only; see `docker compose exec alertmanager wget -q -O- http://127.0.0.1:9093/` |
| Grafana | <http://localhost:3000> | dashboard *Limbot Overview*, user `admin` |
| Qdrant | <http://localhost:6333/dashboard> | storage on a named volume |
| Postgres | `localhost:5432` | read-only to the app; schema seeded on first boot |

Published ports bind to `127.0.0.1` by default. Set `BIND_ADDRESS=0.0.0.0` to expose them on
the LAN, and change `GRAFANA_ADMIN_PASSWORD` and `POSTGRES_PASSWORD` before doing that.

Meta needs a public HTTPS URL. Point a tunnel at port 8000 (`cloudflared tunnel --url
http://localhost:8000`, `ngrok http 8000`, and so on) and use
`https://<subdomain>/webhooks/whatsapp` as the callback URL in the dashboard.

### Alerting

Thirteen rules live in `monitoring/prometheus/alert_rules.yml` and are grouped as
availability, LLM, latency, security and capacity. Prometheus evaluates them every 15s and
sends the results to Alertmanager, which routes by severity: `critical` pages over webhook and
email, `warning` and `info` send an email digest only.

**Destinations are configured through the environment, not by editing a file.** Alertmanager does
not expand environment variables, so its config is a template
(`monitoring/alertmanager/alertmanager.yml.template`) that `render-config.sh` renders at
container start and then hands to Alertmanager. Set `ALERT_WEBHOOK_URL`, `SMTP_SMARTHOST`,
`SMTP_FROM`, `SMTP_TO` and optionally `SMTP_HELLO`, `SMTP_AUTH_USERNAME` and `SMTP_AUTH_PASSWORD`
in `.env`; nothing secret is ever committed. Do not edit a rendered file or add one to the repo -
the template holds no destinations at all, and a test fails if it ever does.

Every one of those variables may be left unset. Unset values render to `example.invalid`
addresses and the IANA discard port `127.0.0.1:9`, which RFC 2606 and the IANA registries reserve
so they cannot deliver anywhere: an unwired deployment starts, evaluates alerts and discards them
rather than paging a stranger. `render-config.sh` logs a `WARNING` on every start while those
placeholders are still in place, which is the only clue you get that alerts are going nowhere -
check it with `docker compose logs alertmanager`.

The script prefers `envsubst` and falls back to an `awk` renderer, because the Prometheus images
are busybox-based and busybox does not ship `envsubst`. Both produce the same config, the test
suite renders with both and compares, and CI renders inside the real image before running
`amtool`. Set `ALERTMANAGER_RENDERER=envsubst` or `awk` to pin one.

The rules are guarded by `tests/test_alert_rules.py`, which fails the build if a rule names a
metric or label the application does not emit, or if an alert has no severity, summary or
runbook. `tests/test_render_config_script.py` covers the renderer itself. CI additionally runs
the authoritative validators in containers, rendering the template first:

```bash
promtool check rules monitoring/prometheus/alert_rules.yml
amtool check-config <rendered alertmanager.yml>
promtool check config prometheus.yml
```

Two things worth knowing before tuning thresholds. `HighLatency` watches
`limbot_llm_call_duration_seconds`, not `limbot_http_request_duration_seconds`: the webhook
route acknowledges Meta and returns before any AI work starts, so HTTP latency is always
small and would never cross a 4s threshold. And `WebhookVerificationFailures` cannot fire when
`WHATSAPP_SIGNATURE_REQUIRED=false`, which is the local default — silence means verification
is off, not that nobody is probing.

### Local development without Docker

```bash
uv sync                      # resolves and writes uv.lock
uv run uvicorn app.main:app --reload
uv run pytest
uv run ruff check .
uv run mypy app
```

The app needs Postgres and Qdrant for the tools and RAG paths, but it boots without them and
disables the affected features:

```bash
docker compose up -d qdrant postgres
```

## How answers are built

Inbound text is read, marked read, and handed to `AnswerService.answer`. It:

1. pulls the last few turns of the conversation with that student,
2. retrieves up to `RAG_TOP_K` chunks above `RAG_SCORE_THRESHOLD` from Qdrant,
3. calls the LLM through the provider chain (Groq → Gemini → tier3, each behind a circuit
   breaker) allowing up to `AI_MAX_TOOL_ROUNDS` of tool calls,
4. appends the reply to the conversation and sends it back via the Cloud API.

Tools only ever read student data: the pool opens with `default_transaction_read_only`, so
nothing a model invents can mutate a row. Identities come from the verified `wa_id`, never
from what the model says.

## Evaluation

`app/eval/` holds a fifteen-case synthetic catalogue (`app/eval/cases.py`): identity and
safety cases that always run, plus capability-scoped cases for the student tools, retrieval,
and conversation memory that are skipped when that backend is not configured — a bare-key CI
run measures the prompt behaviour, a fully provisioned host measures everything.

```bash
GROQ_API_KEY=... python -m app.eval --min-pass-rate 0.8
GEMINI_API_KEY=... python -m app.eval --min-pass-rate 0.8 --group retrieval --json
```

Every case is graded deterministically (required phrases, forbidden phrases, word limits, and
which tools were or were not called), so the suite cannot be argued with or tuned by luck.
Results are recorded per case and per provider in `limbot_evals_total`, and the pass-rate gate
is what makes CI exit non-zero on a prompt regression. The CLI also posts a summary to
`$GITHUB_STEP_SUMMARY` when running inside a workflow.

## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request:

1. **quality** — ruff (lint + format), mypy, pytest;
2. **filter** — whether this change touches the prompt or the answer path;
3. **evals** — the synthetic suite above, **required whenever the prompt changed**, so a
   regression blocks the merge;
4. **build** — the image is built and pushed to `ghcr.io/<owner>/<repo>` (sha-tagged, `latest`
   on main);
5. **deploy** (main only) — `flyctl deploy` to Fly.io.

Configure the repository for this:

- Push the repo to GitHub first (`git init` is already done; commit and create a remote).
- Repository secrets: `EVAL_GROQ_API_KEY` and/or `EVAL_GEMINI_API_KEY` for the eval job.
- Create the Fly app and set its secrets (see `fly.toml`), then add `FLY_API_TOKEN`
  (a Fly machine token with deploy access) as a repository secret.
- Once `uv.lock` is committed, the quality job can switch its `uv sync` calls to
  `uv sync --locked` for reproducible builds.

## Configuration

Every setting is an environment variable, and `.env` is read automatically. Nothing is
required to boot except an app secret unless you enable the AI pipeline.

### Providers and AI

| Variable | Default | Purpose |
| --- | --- | --- |
| `AI_ENABLED` | `true` | Master switch; `/readyz` is not-ready when on and no provider is usable |
| `GROQ_API_KEY` | — | Enables tier 1 (OpenAI-compatible) |
| `GEMINI_API_KEY` | — | Enables tier 2 (native Gemini API) |
| `TIER3_ENABLED` | `false` | Enables tier 3, the OpenAI-compatible fallback (e.g. ollama) |
| `TIER3_BASE_URL`, `TIER3_MODEL` | `http://ollama:11434/v1`, `llama3.2` | Tier 3 endpoint |
| `AI_MAX_TOOL_ROUNDS` | `3` | Tool-call rounds allowed per answer |
| `AI_MAX_TOOL_CALLS_PER_RESPONSE` | `4` | Tool calls accepted in a single round |
| `AI_TOOL_CALL_TIMEOUT_SECONDS` | `10` | Per-tool budget |
| `AI_CIRCUIT_FAILURE_THRESHOLD` | `3` | Transient failures before a provider opens |
| `AI_CIRCUIT_RECOVERY_SECONDS` | `30` | Half-open probe window |

### Retrieval and embeddings

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_ENABLED` | `true` | Retrieval + ingest; off disables both |
| `QDRANT_URL`, `QDRANT_API_KEY` | `http://qdrant:6333`, — | Vector store |
| `QDRANT_COLLECTION` | `limbot_knowledge` | Collection the bot reads and writes |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | `fastembed` model |
| `EMBEDDING_DIMENSIONS` | `384` | Must match the model |
| `RAG_TOP_K` | `4` | Chunks fed to the model |
| `RAG_SCORE_THRESHOLD` | `0.35` | Below this a chunk is ignored and the bot answers without context |
| `KNOWLEDGE_CHUNK_CHARS`, `KNOWLEDGE_CHUNK_OVERLAP` | `900`, `150` | Ingestion chunking |

Load material with `DEBUG_ENDPOINTS=true`:

```bash
curl -X POST http://localhost:8000/debug/knowledge \
  -H 'content-type: application/json' \
  -d '{"title":"Week 3","source":"la.pdf","text":"Eigenvalues satisfy A v = lambda v ..."}'
```

The response includes a `document_id` hash: re-posting the same material replaces its chunks
instead of duplicating them.

### Student data

| Variable | Default | Purpose |
| --- | --- | --- |
| `POSTGRES_DSN` | — | Enables the student tools (read-only connection) |
| `POSTGRES_POOL_MIN/MAX_SIZE` | `1`/`5` | Pool bounds |
| `CONVERSATION_*` | — | Turn count, TTL and conversation-capacity for memory |

### WhatsApp, HTTP, tasks, observability

As in Phase 1: `WHATSAPP_*` (secret, token, access token, phone number id), `HTTP_*` retry and
pool tuning, `BACKGROUND_*` concurrency, `MESSAGE_DEDUPE_*`, `QDRANT_REQUIRED`,
`DEBUG_ENDPOINTS`, `BIND_ADDRESS`. The full list is in `.env.example`.

## Verifying the webhook by hand

```bash
BODY='{"object":"whatsapp_business_account","entry":[{"id":"1","changes":[{"field":"messages","value":{"messaging_product":"whatsapp","metadata":{"phone_number_id":"1"},"contacts":[{"profile":{"name":"Ada"},"wa_id":"15550001111"}],"messages":[{"from":"15550001111","id":"wamid.TEST1","timestamp":"1700000000","type":"text","text":{"body":"hello"}}]}}]}]}'

SECRET=$(grep WHATSAPP_APP_SECRET .env | cut -d= -f2-)
SIG="sha256=$(printf '%s' "$BODY" | openssl dgst -sha256 -hmac "$SECRET" -r | cut -d' ' -f1)"

curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8000/webhooks/whatsapp \
  -H "content-type: application/json" -H "x-hub-signature-256: $SIG" -d "$BODY"
```

Expect `200`, a `limbot_webhook_events_total{result="accepted"}` increment, and an
`inbound message` line on stdout. A tampered body or secret returns `401`.

With `DEBUG_ENDPOINTS=true` and `WHATSAPP_ECHO_MODE=true` the bot replies with your text,
exercising the outbound path end to end.

## Design notes

**Webhooks answer before they work.** Meta redelivers anything that is not answered quickly,
so `POST /webhooks/whatsapp` verifies the signature, parses the envelope and hands the payload
to `TaskManager` without awaiting it. Rejections are deliberate: `401` for a bad signature,
`422` for an unparseable body, and `503` while draining so the delivery is retried after the
container is back.

**Readiness reflects what actually works.** `/readyz` is not ready while draining; while
`AI_ENABLED` with no usable closed provider; and while echo mode is on with AI off. Qdrant and
Postgres are reported but only fail readiness under their own `*_REQUIRED` flags, so a dev
stack stays green without them.

**Providers degrade, they do not default out.** Every provider sits behind a circuit breaker
that opens after `AI_CIRCUIT_FAILURE_THRESHOLD` consecutive transient failures. An open tier is
skipped in favour of the next; non-transient errors (bad keys, model limits) surface without
fallback. Answering with "I could not get a response" is reserved for when every tier has
failed.

**Tools cannot write.** The database connection is forced read-only at the protocol level, so a
wrong model turn cannot mutate a row. Timeouts, argument schemas with
`additionalProperties: false`, and a four-call round cap keep a runaway loop from costing money.

**One worker.** The dedupe cache, circuit breaker state, the task pool and the metrics
registry all live in the process. Scaling out means more containers, not `--workers`.

**Nothing sensitive is logged.** Request logs carry method, route template, status and
duration. Payloads, API keys and full answers are never logged, and secrets stay in
`SecretStr` fields.

## Next

Media download handling, per-conversation custom state, and stronger RLS in Postgres for a
multi-tenant deployment.