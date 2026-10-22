# RAG module: Incident Investigation Agent

This module extends the existing Python MCP client and server. All data is made up for learning.
The current incident is excluded from its own historical search. After an explicit human save,
its decision and action can be retrieved by a future incident with a different ID.

## Five synthetic historical incidents

| Incident | Symptoms | Useful next check |
| --- | --- | --- |
| INC-09876 | Billing settlement stays pending; settlement calls return HTTP 503; retries fail | Monitoring: check settlement errors and application health |
| INC-09741 | Card authorization fails with a local exception after an application release | Deployments: inspect the relevant recent change |
| INC-09632 | Database row locks delay payment balance updates | Traces: find where the request waits |
| INC-09508 | Refund calls return HTTP 401 after an access token expires | Traces: inspect the failed provider call |
| INC-09402 | Payment finishes, but receipt emails wait in a growing queue | Monitoring: inspect notification workers and queue age |

Each case includes its own ID, symptoms, service, past cause, check suggestion, date, and
`synthetic://` reference. A past cause is confirmed only within that invented past case.
It does not confirm the current cause. The original current-incident fixtures are unchanged.

## Embeddings and index

`incident-index` loads the five documents and uses FastEmbed's `BAAI/bge-small-en-v1.5` model to
create 384-number embeddings on the CPU. The corpus now also includes approved records from
SQLite. Each compact incident is one document. The embedded
text includes the title, symptoms, and service. The past cause and suggested next check are
returned as context after retrieval; they are not part of the embedded search text.

Vectors are normalized and stored in a FAISS `IndexFlatIP` index. The query uses the same model
and normalization. Inner product then measures cosine similarity. Higher scores mean closer text,
not a higher chance that the same cause applies now.

The demo accepts scores of at least **0.70** and returns at most `top_k` matches (default 2).
This is a simple setting checked on this tiny synthetic collection, not a threshold validated for
real incidents. Five separate paraphrase queries found their intended case first; two unrelated
queries returned no cases. Real deployment would need a larger collection and evaluation set.

Index metadata includes the model, vector size, corpus hash, index checksum, and document order.
A missing, changed, or damaged index is a tool failure. A completed search with no match is an
empty result. Searches use cached model files; build/download happens only in the setup command.

## The two history tools

`search_incidents(transaction_id)` is the original fixed fixture lookup. It reads the old record
already linked to a transaction ID. It does not generate query embeddings or compare documents.

`searchSimilarIncidents(transaction_id, query, top_k)` is the new vector-search MCP tool. The
symptom text drives the search; the transaction ID only identifies the current investigation.
The server returns matched cases, references, and similarity scores.

The original demo still uses the fixed lookup. The without-RAG comparison uses neither history
tool. The with-RAG comparison uses the vector-search tool, so both runs do not receive the same
historical clue through different routes.

## The API agent

The client uses OpenAI Responses structured output with the pinned model
`gpt-4.1-mini-2025-04-14`. The model generates a next-tool choice and a short reason, with references
to records already observed. It receives the actual retrieved case when RAG is enabled.

Transaction, logs, and the RAG search are setup steps. The model chooses among remaining current
checks when more than one is available. A sole remaining check needs no model request. Python
checks tool names, source references, call limits, and final findings. Both modes use the same
model instructions. No instruction tells the model to manufacture a different result.

An invalid decision, refusal, timeout, or API error pauses with the evidence collected so far.
API retries are disabled. A run cannot silently switch to the rule-based planner. The API key is
read from the environment or a hidden `--ask-key` prompt and stays in the client.

## Observed comparison

The verified run used real embeddings, the saved FAISS index, and actual MCP calls. It selected
the explicitly named **demo** planner, which uses Python rules. It did not make a live API request.

| Step | Without RAG | With RAG |
| --- | --- | --- |
| 1 | Read transaction: SETTLEMENT_SENT for 26 hours | Same |
| 2 | Read logs: settlement HTTP 503 | Same |
| 3 | Check distributed trace | Search history: INC-09876 ranked first, similarity about 0.93 |
| 4 | Check monitoring | Check monitoring, citing the retrieved INC-09876 |
| 5 | Check deployments | Check distributed trace |
| 6 | Finished | Check deployments, then finish |

The retrieved case suggests checking settlement health earlier. Current monitoring then shows
settlement errors while the billing application is healthy. The current trace still confirms
that this transaction received no successful settlement response after processing retries.
Both runs report the settlement-service call as a supported failure point. The underlying root
cause remains open for engineer review. The purpose of RAG here is a better-informed next check;
it does not create proof or guarantee fewer calls.

The comparison command shows whether the next check changed and whether its action cites a
retrieved historical source. It also shows an unchanged choice honestly. It does not save report
files. The live API run may choose differently from this verified demo run.

## Run the API version for submission

In the project directory, run each mode and take a screenshot of the terminal result:

```bash
.venv/bin/incident-client --mode without-rag --planner openai --ask-key
.venv/bin/incident-client --mode with-rag --planner openai --ask-key
```

Enter the key at each hidden prompt. No report folder or saved report is required. The key is
never saved. Include the tools used, the next-check reason, and the final summary in your document.
Add `--verbose` if you need full reasons and source references.

To compare both modes in one command, run `incident-compare --planner openai --ask-key`.
It displays the results without creating files. A paused run exits with code 2 and shows the
partial evidence. Report-file export code has been removed; use terminal screenshots for your document.

For a fresh extracted source package, first run:

```bash
uv sync --locked --extra dev
uv run incident-index
uv run incident-client --mode without-rag --planner openai --ask-key
uv run incident-client --mode with-rag --planner openai --ask-key
```

## Rules kept from the earlier design

- Eight tool attempts total per cycle, including vector searches, failures, and retries.
- At most two retries per failed tool, subject to the shared budget and earlier pause rules.
- Pause after two consecutive tool calls with no new meaningful evidence.
- A score change or a new record ID alone is not new incident evidence.
- Current logs and a current trace must support the current failure point.
- Empty results and missing evidence do not prove that a service is healthy or a cause is false.
- Stop early when a status-only request is answered.
- The engineer confirms root cause and separately approves any new knowledge save.
- All recovery remains manual. The agent never saves its current findings automatically. A separate human-approved save may add
that incident later, while self-matches remain excluded.

API requests are reported separately from investigation tool attempts. The eight-call tool limit
is enforced before another model decision, so reaching it cannot trigger another API decision.

## Validation and limits

The automated checks cover index persistence, score filtering, all five retrieval examples,
unrelated queries, actual MCP calls, branching, retry accounting, insufficient current evidence,
conflicting current evidence, duplicate facts, API failure reports, and honest comparison results.
API tests use fake responses, including a fake HTTP response parsed by the real installed SDK.
No paid API request was made because no key was available in this session.

The normal test suite needs no model download or API key. Real embedding checks require the
prepared local index and `RUN_RAG_INTEGRATION=1`. See `verification.json` for the final check result.

## Sources used for the implementation

- [FastEmbed getting started](https://qdrant.github.io/fastembed/Getting%20Started/)
- [FAISS cosine similarity](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)
- [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)
- [OpenAI GPT-4.1 mini model](https://developers.openai.com/api/docs/models/gpt-4.1-mini)

## Saving human decisions and actions

The new SQLite store keeps the complete approved record. Use `incident-knowledge save` to enter
a decision, actual action, result, and approver. It commits the record before updating FAISS.
See [human-decisions.md](human-decisions.md) for the command and database details.
The five original cases remain examples; saved human records extend the collection.
