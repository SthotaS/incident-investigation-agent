# Incident Investigation Agent

A Python MCP client and server for the production incident investigation capstone. All incident
records are synthetic. The original six-tool demo is kept, and the RAG module adds five historical
incidents, real text embeddings, a saved FAISS vector index, and an API model that chooses the next
investigation check from the evidence it has seen.

## Local setup

After cloning, the `.venv` folder will be missing. It is excluded from Git because each
project copy needs its own Python environment. Create it inside the new clone.

Open Terminal in the cloned project folder, where `pyproject.toml` is located. Use Python 3.12.

1. Create `.venv` and install the application and its dependencies:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

2. Prepare RAG by downloading the embedding model and building the FAISS index:

```bash
.venv/bin/incident-index
```

3. Run the application and enter your API key at the hidden prompt:

```bash
.venv/bin/incident-client --mode with-rag --planner openai --ask-key
```

These commands use the new folder's `.venv` directly; activation is not required. The folder
is hidden because its name starts with a dot. Use `ls -a` to see it in Terminal.

The downloaded model, index, and saved human decisions are also excluded from Git. Previously
saved decisions remain in your original project folder. A new clone starts with the five
synthetic historical cases unless you separately restore your decision database.

Run `incident-index` again after changing the historical data. Investigations read the saved
index and cached model; they do not download or rebuild them.

### Alternative setup with uv

If you use [uv](https://docs.astral.sh/uv/getting-started/installation/), it can create `.venv`
and install dependencies from the lock file:

```bash
uv sync --locked --extra dev
uv run incident-index
uv run incident-client --mode with-rag --planner openai --ask-key
```

The pip route uses the version ranges in `pyproject.toml`; the uv route uses the tested lock file.

## Save the human decision and action

```bash
.venv/bin/incident-knowledge save
```

Enter the engineer's decision, action actually taken, result, and name. Review the record, then
explicitly approve saving it. The complete record is stored in SQLite at
`data/knowledge/incidents.sqlite3`; the FAISS search index is then updated using the cached model.
No API key is needed to save or read knowledge. The agent cannot save records by itself.
Saved records are immutable: repeating the same save is allowed; a different record under the
same incident ID is rejected.

```bash
.venv/bin/incident-knowledge list
.venv/bin/incident-knowledge show --incident-id INC-10234
```

See [the human decision guide](docs/human-decisions.md) for interactive/file-based saves, duplicate
handling, index recovery, and the difference between save approval and cause confirmation.

## Demo without and with RAG

Both runs use the same investigation engine, selected model, investigation rules, and
current-data tools. The boolean only controls retrieval of past incidents:

```python
InvestigationRequest(use_rag=False)
InvestigationRequest(use_rag=True)
```

```bash
.venv/bin/incident-client --no-use-rag --planner openai --ask-key
.venv/bin/incident-client --use-rag --planner openai --ask-key
```

The existing `--mode without-rag` and `--mode with-rag` commands remain supported.
Conflicting mode and boolean options are rejected. With no mode or flag, the original
legacy demo remains the default.

When retrieval is disabled, progress shows the skipped check after logs:

```text
[Skipped] Similar incidents: RAG disabled; knowledge database not checked.
[3/8] Trace: found; ...
```

Only actual tool attempts (including retries) receive call numbers. A skipped check
neither uses a call nor counts as an empty search or a step without new evidence.
JSON reports expose `use_rag` and `skipped_checks` separately from the tool-call `audit`.
A status-only run or a run stopped before reaching retrieval has no skipped check.

After completing Local setup above, open Terminal in the project folder and run each command:

```bash
.venv/bin/incident-client --mode without-rag --planner openai --ask-key
.venv/bin/incident-client --mode with-rag --planner openai --ask-key
```

Enter your OpenAI API key at each hidden prompt. It is used only for that process and is not
saved. The client starts and closes the MCP server automatically. API requests use your API
account. You do not need a GPU or a locally installed language model.

Take a screenshot of each terminal result for your document. Include the tool order, the reason
for the next check, the failure point, and the remaining questions. If a short reason is cut off,
add `--verbose` to see it in full. These commands display results without creating report files.

Both runs investigate INC-10234 / TX784521 with fresh memory and the same model and limits:

- **Without RAG:** uses only the current transaction, logs, trace, monitoring, and deployments.
- **With RAG:** also calls `searchSimilarIncidents()` after reading the transaction and logs.
  The model receives the returned past cases as context when choosing the next check.

A model may choose the same next check in both runs. Use what your run actually shows; retrieval
does not guarantee a different choice. To run both modes in one terminal command:

```bash
.venv/bin/incident-compare --planner openai --ask-key
```

This comparison displays the actual actions and whether a historical source supported a changed
next check. It displays results in Terminal and creates no report files. If either run pauses, the command exits
with code 2 and shows the evidence collected so far.

### Run without an API key

```bash
.venv/bin/incident-client --mode without-rag --planner demo
.venv/bin/incident-client --mode with-rag --planner demo
```

This uses **real embeddings, FAISS retrieval, and MCP calls**, with Python rules choosing tools.
It is a repeatable check of retrieval and the workflow. It is not a live language-model run.
For the API-based RAG submission, use the OpenAI commands above and screenshots of their results.

## If the AI step pauses

The transaction, logs, and RAG search can finish before the first OpenAI request. If that
request fails, the investigation pauses with its current evidence. Eight calls is the maximum;
it does not mean the agent must keep going after a failed AI decision.

The `AI issue` line now gives a safe error category instead of one general message:

| Message | What to check |
| --- | --- |
| HTTP 401: API key rejected | Check the API key and selected API project. |
| HTTP 403: access denied | Check project permissions and model access. |
| HTTP 404: model or endpoint not found | Check the model, access, and any custom API base URL. |
| HTTP 429: credit balance empty or credit/quota limit | Check API billing and available credit. |
| HTTP 429: project or organization spend limit | Review the spending limit for that project or organization. |
| HTTP 429: organization usage limit | Request a higher approved API usage limit. |
| HTTP 429: rate limit | Wait briefly before starting another run. |
| HTTP 400 or 422: request rejected | Check the model, output schema, and request settings. |
| Timeout or connection problem | Check internet, VPN, or proxy settings. |
| HTTP 5xx: API service error | Wait briefly before starting another run. |
| Invalid AI decision | Check the output format and client version. |

For a limit error, select the same organization and project used by your API key. Check
[API billing](https://platform.openai.com/settings/organization/billing/overview) and
[API limits](https://platform.openai.com/settings/organization/limits). Adding credit only helps
when the credit balance is the problem; repeated retries do not fix billing or spending limits.

The app uses specific error codes first and a known broader error type when needed. It shows
fixed messages and safe status numbers. It never prints the API key or raw
provider response. It keeps API retries disabled and does not silently switch to the demo planner.
The settlement HTTP 503 in the synthetic logs is separate from an OpenAI request failure.
See the [official OpenAI error guide](https://developers.openai.com/api/docs/guides/error-codes).

## What the RAG example shows

The synthetic current transaction has stayed `SETTLEMENT_SENT` for 26 hours. Logs show settlement
HTTP 503 responses. These observed facts form the search query; the agent does not use a hidden
answer or the ID of a desired past case.

The vector search finds **INC-09876**, an earlier settlement-service failure. Its record suggests
checking settlement errors and application health. This can move monitoring ahead of the trace.
The model must cite observed records in its action summary. The trace is still needed before the
agent reports the settlement-service call as an evidence-supported failure point.

The checked rule-based run produced:

```text
Without RAG: transaction -> logs -> trace -> monitoring -> deployments
With RAG:    transaction -> logs -> vector search -> monitoring -> trace -> deployments
```

See [the RAG submission guide](docs/rag-module.md) for data, models, limits, and what was verified.

## How the agent narrows possible causes

The next check now has an explicit priority based on the current evidence. For example,
settlement HTTP 503s plus a failed trace make service health and other transactions useful to
check next. A completed deployment review with no related change weakens the release explanation;
an empty or unfinished check does not. A matching past case can guide a check, but current evidence
has priority. Already checked tools are reused as evidence rather than called again.

Monitoring now returns a time window, when errors began, and a sample of failed and successful
transactions. The synthetic example has four failures close in time, including TX784521, and two
successes before the errors began. This supports checking a shared service problem; the underlying
cause still needs an engineer's review.

The terminal shows a short summary. With `--verbose`, it shows the same complete possible causes,
source references, evidence for and against them, missing evidence, and next checks as the JSON
shown with `--json`. Check priority is not a probability.
See [the architecture guide](docs/architecture.md#choosing-the-next-check) for the sample and rules.

## Models and storage

| Part | Used for |
| --- | --- |
| `gpt-4.1-mini-2025-04-14` through the OpenAI API | Generate the next check and a short reason when several checks remain. |
| `BAAI/bge-small-en-v1.5` through FastEmbed | Turn historical text and search queries into 384-number vectors locally. |
| FAISS `IndexFlatIP` | Search normalized vectors using cosine similarity. |
| Python evidence rules | Count calls, enforce stops, validate decisions, and produce the final findings. |

The embedding model and index live under `data/rag/`. They can be rebuilt and are excluded from
source control. Cached models, the Python environment, and package downloads use most of the disk
space. Each index rebuild keeps a complete generation so readers can finish safely; old generations
are not removed automatically.

## Folder structure

```text
incident-investigation-agent/
├── client/src/incident_client/
│   ├── llm.py                 # API model decisions and reference validation
│   ├── policy.py              # Allowed checks and evidence-based priorities
│   ├── hypotheses.py          # Possible causes, evidence, gaps, and next checks
│   ├── reporting.py           # Clear console cause summaries
│   ├── investigation.py       # Shared limits, memory updates, evidence-based findings
│   ├── compare.py             # Compare without/with RAG in Terminal
│   └── mcp_client.py          # MCP stdio connection; API key stays in the client
├── server/src/incident_server/
│   ├── mcp_server.py          # Seven read-only investigation tools
│   ├── rag.py                 # Build/load/search the FAISS index
│   └── fixtures/
│       ├── incidents.json
│       └── historical_incidents.json  # Exactly five made-up historical incidents
├── shared/src/incident_common/       # Validated tool and retrieval data
├── tests/                           # Unit, MCP, and optional real-embedding checks
├── docs/
│   ├── architecture.md             # Agent flow, next-check rules, and safety limits
│   ├── rag-module.md               # RAG design and terminal demo
│   ├── human-decisions.md          # Save and read approved decisions
│   └── examples/                   # Synthetic decision input and blank template
├── data/rag/                        # Local generated data; excluded from source control
├── data/knowledge/                  # Durable approved human decisions in SQLite; back up this folder
├── pyproject.toml
└── uv.lock
```

## Investigation rules

- At most **8 tool attempts per cycle**, including vector searches, failures, and retries.
- A failed tool has at most **2 retries**, only while the shared budget and pause rules allow it.
- **2 consecutive calls without new meaningful evidence** pause the investigation. A changed score
  or record ID does not make the same facts new. Empty results are different from tool failures.
- Current logs **and** a current settlement trace must support the failure point. Past cases and
  similarity scores cannot replace either source or confirm the present root cause.
- API decisions are checked against the available tools and already-seen source references.
  Failed or invalid API decisions pause with the evidence collected so far. There is no silent
  switch to the demo planner. API requests have no automatic retries and are counted separately.
- A status-only question can finish after one call. A paused run never resets its own budget.
- An engineer confirms root cause and separately approves saving new knowledge. Recovery is
  manual. The investigation tools do not change production systems or save knowledge. The separate
  human-operated command saves approved decisions; it excludes the current incident from its own search.

## Terminal output

The terminal now shows short progress lines and a short final summary by default. It keeps the
pause reason, possible failure point, past case IDs, missing checks, and next step visible.
A failed AI request still pauses the investigation; shortening the display does not fix an API issue.

```bash
# Short output (default).
.venv/bin/incident-client --mode with-rag --planner openai --ask-key

# Full action reasons, evidence, source links, and possible causes.
.venv/bin/incident-client --mode with-rag --planner openai --ask-key --verbose

# Full structured report with no live progress output.
.venv/bin/incident-client --mode with-rag --planner openai --ask-key --json
```

`incident-compare` also accepts `--verbose` for full live progress. Investigation and comparison
commands display their results in Terminal; they do not create report files.

## Other commands

```bash
# Original six-call, rule-based demo: no API key and no index required.
.venv/bin/incident-client

# Run one API investigation with retrieval.
.venv/bin/incident-client --mode with-rag --ask-key

# Use an API key already exported in this Terminal.
.venv/bin/incident-compare

# Measure RAG quality on six synthetic cases; no API key or saved reports.
.venv/bin/incident-compare --evaluate

# Discover all seven tools, including searchSimilarIncidents.
.venv/bin/incident-client --list-tools

# One-call status check.
.venv/bin/incident-client --transaction-id TX000001 --scope status

# JSON only on stdout; live action messages normally go to stderr.
.venv/bin/incident-client --mode with-rag --planner demo --json
```

## Configuration

`.env.example` lists optional settings. The app reads environment variables and does not load an
`.env` file automatically. `OPENAI_MODEL` can override the default with a model supporting the
same Responses structured-output options. Model availability depends on your API account.
`INCIDENT_RAG_DIR` changes the local model/index directory; use the same path for build and run.
`MCP_TOOL_TIMEOUT_SECONDS` defaults to 10 (maximum 120). `OPENAI_TIMEOUT_SECONDS` defaults to 30
(maximum 120). `MAX_API_INPUT_BYTES` defaults to 131072, and `MAX_API_CALLS` defaults to 8. These
control request size and usage; they do not impose a dollar spending cap. The report includes actual
API token usage when returned by the provider. Missing usage remains unknown.

## Reliability

The engine validates planner choices and tool evidence independently. Evidence sizes, source IDs,
timestamps, and monitoring samples have explicit limits. Source timestamps must include a time
zone, and monitoring samples contain at most 100 observations. Historical records cannot supply
current incident evidence.

Approved knowledge records stay immutable. The save command checks the 1,000-case capacity,
including the five seed cases, inside the SQLite write transaction. An identical save is safe to
repeat; different content under the same incident ID is rejected. Each saved decision may have
up to 32 source references of at most 512 characters.

SQLite saves remain durable when an index rebuild fails. FAISS rebuilds publish complete index
generations by atomically replacing a single `current.json` pointer. A process lock prevents
competing builds. A failed build preserves the previous complete generation, and existing
two-file indexes remain readable until an explicit rebuild.

A decision saved during or just after a rebuild can require another `incident-index --offline` run;
search rejects stale indexes. The stores are separate, so an index failure does not erase a decision.

To rebuild using the already downloaded embedding model:

```bash
.venv/bin/incident-index --offline
```

Back up `data/knowledge/` separately. The index can be rebuilt from the seed cases and saved
decisions. Old index generations are not automatically deleted while readers may still use them.

Exit codes are `0` for finished, `2` for paused, `1` for an operational error, and `130` for an
interruption.

## Checks

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m pytest -q

# After preparing the index: include real embeddings and MCP retrieval checks.
RUN_RAG_INTEGRATION=1 .venv/bin/python -m pytest -q
```

Normal tests do not need internet or an API key. Model decision tests use a clearly marked fake
API response. The real-embedding checks require the prepared local model and index.
The latest local checks passed 449 tests, including real MCP and embedding checks; Ruff and the
wheel build also passed. No paid API calls were used. The GitHub Checks workflow is configured
but has not been run on GitHub. See `verification.json` for the recorded results.

See [architecture](docs/architecture.md) and the [RAG submission guide](docs/rag-module.md).
This is a working synthetic capstone application. Real production use still needs secure live
data connectors and credentials, authenticated users and approval roles, monitoring, and tested
backup and recovery procedures.
