# G1 Research Governance v2.0 — entrypoint

Governance v2.0.0 — 2026-11-27 — documentation only. Historical research state pinned to a2a1056075f4a41e9f05115b30e2a50144b34154.

**Mandatory per conversation:** read [CORE_RULES.md](CORE_RULES.md), [CURRENT_STATE.md](CURRENT_STATE.md), [ACTIVE_EXPERIMENT.md](ACTIVE_EXPERIMENT.md), then only applicable exact gates, runtime protocol and evidence ranges. Declare governance version, current Git HEAD, active state and exact evidence consulted. Do not load the 2.34 MB historical monolith by default. If uncertainty or a conflict remains, retrieve original evidence or HOLD.

| Module | Read when |
|---|---|
| [Core](CORE_RULES.md) | Always |
| [Current State](CURRENT_STATE.md) | Always |
| [Active Experiment](ACTIVE_EXPERIMENT.md) | Always |
| [Scientific Gates](SCIENTIFIC_GATES.md) | Scientific review/design |
| [Runtime Protocol](RUNTIME_PROTOCOL.md) | Static/runtime/authorization |
| [Evidence Index](EVIDENCE_INDEX.md) | Retrieve exact original evidence by task/version |
| [Lessons Learned](LESSONS_LEARNED.md) | Planning and avoiding prior failures |
| [Task Ledger](TASK_LEDGER.md) | Task completion |
| [Changelog](CHANGELOG.md) | Governance changes |
| [Immutable archive](archive/v1.2.294/G1_RESEARCH_RULES.md) | Exact historical retrieval only |

## Compatibility and migration invariants
Root G1_RESEARCH_RULES.md is deliberately retained at original path and bytes; it is the legacy parser/launcher ABI, not mandatory human reading. Original historical Rule 0 full-read instruction is superseded by v2.0, along with old AGENTS reading/maintenance routing only. Exact original blob 7d5961158420ad2053ed00cc1342aa2a54a12ce9; archived copy references the same Git blob, not a re-render. Root stays available for future separately authorized marker lifecycle; archived copy never changes. No active GO at base; historical approvals are not active authorization. All original sections remain accessible via EVIDENCE_INDEX, including every historical detail and embedded launcher. Preserve existing frozen source and SHA contracts; no runtime dispatch/retry/promotion. Historical launcher HEAD/diff pins are not reissued by migration. Future authorization is a distinct Owner action using exact new base, with root-only changes where required.

## Review and rollback
Base a2a1056075f4a41e9f05115b30e2a50144b34154; original tree 7b76eaf95337b792226462bd32e2374b8fd95e90. Migration changes AGENTS.md routing and adds only Markdown modules plus exact archive. Validate pre/post tree: all non-documentation and legacy-root blobs unchanged; archive same blob; no active marker added and five launcher pairs unchanged by identity. No GPU, local fixtures, external checkpoints or artifact rehash performed. Local dependencies not independently verified. If compatibility is ever uncertain, HOLD and inspect exact reader/launcher contract. Rollback via reviewed revert of migration commit, never force/reset. No robot workflow advanced.

## Next conversation template
```
Read AGENTS.md and docs/research_rules/{CORE_RULES,CURRENT_STATE,ACTIVE_EXPERIMENT}.md at exact experiment branch HEAD.
Declare governance version, HEAD, current authorization and task scope.
Retrieve only applicable exact evidence via EVIDENCE_INDEX; read SCIENTIFIC_GATES and RUNTIME_PROTOCOL when relevant.
Follow EDIT-FIRST; report evidence, limits, GO/NO-GO and exact next step. No runtime without distinct Owner GO. Branch A ACTIVE, B HOLD.
```


**Latest governance revision: v2.0.1 (2026-11-28).** Runtime Protocol now requires complete exclusive outer terminal capture with tee and separate pipeline exit receipts; no change to scientific thresholds, root ABI or executable source. Historical v2.0.0 headings remain provenance. Current R16 disposition is recorded at the end of CURRENT_STATE and ACTIVE_EXPERIMENT.
