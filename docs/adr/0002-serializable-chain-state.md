# 0002. Serializable per-chain state

Status: Accepted

## Context

A full run covers thousands of chains and takes many hours on one GPU. Runs crash, machines reboot,
and a human may want to reopen a finished chain in the GUI days later. Recomputing a chain to inspect
it is wasteful, and losing progress on a crash is worse.

## Decision

Carry one serializable `ChainState` per chain and write it to `state.json`. It holds the chain
identity and status, the anchor frame, the prompts, the QC summary, the frames flagged for review,
and the crop window for tier-2 chains. The status is one of `pending`, `running`, `done`, `flagged`,
or `failed`.

The batch driver writes a `running` breadcrumb before a chain starts and the terminal status when it
finishes. Resume reads these: a chain already `done` or `flagged` is skipped, and an interrupted
chain left at `running` is retried.

## Consequences

- A run resumes after a crash without recomputing finished chains.
- The GUI reopens any chain from `state.json` plus its saved masks, with no GPU work needed to browse.
- The state is the contract between the drivers. The batch writes it; the GUI and the read-only
  viewer read it. See [state-and-storage.md](../reference/state-and-storage.md) for the schema.
- State must stay serializable. Fields added to `ChainState` have to be JSON-friendly and declared,
  or they are lost on resume.
