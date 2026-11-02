# 0018. Neuron ids are frozen, not derived

Status: accepted, 2026-08-18.

## Context

A mask's identity was the name of the directory holding it. Where a numeric id was needed,
`experiments/dense_overlay.py` derived one at render time with
`{n: i + 1 for i, n in enumerate(neurons)}`, so a neuron's id was its position in whatever subset
that render happened to cover. Rendering a different set of neurons moved the numbers.

That is survivable for a throwaway figure. It is not survivable for anything that outlives the run
that produced it: picking specific neurons to display, a Blender material per neuron, or a VAST
segment number all need an id that means the same thing every time.

## Decision

Neuron ids live in `data/neuron_registry.csv`, tracked in git and append-only. The mapping is built
once from the distinct `cell_name` values in `data/chains.json`, sorted by name so the first build
is reproducible. After that the order stops mattering: a new neuron appends with the next free id,
and no id is ever reassigned.

Id 0 is reserved for background, so registry values drop straight into a `uint16` labelmap.
`sam2_utils/registry.py` rejects a 0 id, a duplicate id, and a duplicate name at load, and an
unregistered lookup raises rather than inventing a number.

## Consequences

Ids are permanent, which is the property everything downstream needs, and it is also the cost: once
an id ships in an export, it cannot be changed without invalidating that export. A guard test pins
the mapping so a careless rebuild fails loudly instead of silently renumbering.

Adding a neuron becomes an explicit act (`scripts/build_neuron_registry.py --append`) rather than
something that happens implicitly at render time. That is deliberate. Implicit assignment is exactly
what produced ids that could not be trusted.

The registry is a flat CSV rather than a database, consistent with
[ADR 0003](0003-filesystem-only-no-database.md).
