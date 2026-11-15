# Shared investigation evaluations

The paired RAG evaluator and the Reviewer use the same eight synthetic incidents. The
[server fixtures](../server/src/incident_server/fixtures/evaluation_incidents.json) hold the raw records.
The [client answer key](../client/src/incident_client/fixtures/rag_evaluation.json) specifies expected
findings, unresolved hypotheses, and claims to withhold. The planner and historical index receive
the current records and historical search data through their ordinary entry points. The answer key
stays in the evaluator.

| Case | Current evidence | Supported finding | Open question | Claim to withhold |
| --- | --- | --- | --- | --- |
| Useful match | Settlement 503 log, exhausted trace, clustered failures | The settlement call failed. Check the shared service. | Why the calls failed | A complete outage or confirmed cause |
| Related change | Local exception, isolated transaction sample, related deployment at 09:59 UTC | The settlement call failed after the deployment. | Whether the change contributed | Deployment causation based on timing |
| Misleading history | Target failure and nearby successes, despite a similar past outage | The sample favors checking the individual transaction. | What differed about it | The old outage diagnosis applies today |
| No match | Archive-checksum records concern a different workload | The settlement workflow cannot establish a payment failure. | Why the archive check failed | A settlement cause or a useful historical match |
| Missing trace | Current 503 logs and nearby failures, with an empty trace result | The logs show settlement errors. The exact failure point remains unresolved. | Where the request stopped | History and logs establish the failure point |
| Paraphrased symptoms | Settlement 503, exhausted trace, clustered failures, different wording | The settlement call failed. Check current service health. | Why multiple requests failed | Similar wording confirms a cause |
| Late deployment | First 503 at 10:00:02 UTC and deployment at 10:05 UTC | The deployment followed the observed failure. | What initiated the failures | The later deployment initiated the earlier failure |
| Contradictory statuses | Pending and completed records share an observation time | The records conflict and require engineer review. | Which status is authoritative | A definitive status or failure point before reconciliation |

The automated checks score structured findings and cause labels. The narrative answer key also
captures what an engineer should look for when reviewing the evidence. See the
[stop rules and priorities](architecture.md#investigation-loop-and-memory) and the
[Reviewer design](architecture.md#evidence-reviewer).

## Reproduce

```bash
.venv/bin/incident-compare --evaluate
.venv/bin/python -m incident_client.evaluation
```

The paired run uses the Python demo planner, the real MCP server, and a temporary FAISS index
built from the five fixed historical seeds. It reports finding quality, all tool attempts, and
elapsed time. It scores historical usefulness against labels assigned before retrieval.

## Results

The existing focused baseline passed 94 tests. The expanded paired evaluation matched the
expected findings in 8/8 cases per mode with 18/18 cause labels and zero unsafe claims. Retrieval
changed the mean tool count from five to six. One local run averaged 59.83 ms without retrieval
and 128.34 ms with retrieval, including first-use warmup. These figures describe the synthetic
demo with zero model requests.

Six of thirteen returned historical records were useful under the rubric. All six cases with an
expected match retrieved at least one. The archive case returned no history. The contradictory
case returned irrelevant history and still escalated on current evidence. The three anchoring
checks preserved the expected findings when history could have misled the investigation.

The Reviewer passed ten checks, including the eight shared incidents and two citation probes.

| Defect | Decision path | Result |
| --- | --- | --- |
| Omitted failure-point citations | Revise → accept | Restored the current log and trace references. |
| Invented reference | Revise → escalate | Paused and withheld the unverified finding. |
| Conflicting transaction statuses | Escalate | Paused immediately with no revision. |
| Deployment after the first 503 | Revise → accept | Moved the late deployment to opposing evidence and kept the supported failure point. |
