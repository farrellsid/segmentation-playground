# export_bundle.py --no-frames Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--no-frames` flag to `export_bundle.py` so a bundle headed to a reviewer
who already has her own raw EM copy can skip shipping frames entirely, relying on
`gui.py`'s existing `_ensure_local_frames` fallback to regenerate them locally on open.

**Architecture:** One new keyword-only parameter on `export_bundle()`, one new CLI flag on
`main()`. Reuses `bundle.rewrite_state_frames_dir`'s existing `relative` parameter (pass
`None` instead of the default `"frames"`) and `bundle.validate_bundle`'s existing
`require_frames` parameter (pass `False`). No new functions in `sam2_utils/bundle.py`.

**Tech Stack:** Python, this repo's existing `export_bundle.py`/`sam2_utils/bundle.py`.

## Global Constraints

- Default behavior (no flag) is byte-identical to today: frames are generated and
  shipped exactly as before. This is purely additive.
- `--no-frames` together with `--force-frames` is a contradiction ("never generate" vs
  "always regenerate"); `main()` refuses this combination with a clear `SystemExit`
  rather than silently picking a winner.
- When `no_frames=True`, the per-chain loop must never call `_frames_complete`,
  `regenerate_frames`, or `_replace_frames_dir`, and must never create an
  `out_dir / "frames"` directory at all.
- `state.json`'s `frames_dir` becomes `None` (via
  `bundle.rewrite_state_frames_dir(rec["state"], relative=None)`), not a relative path,
  when `no_frames=True`.
- The bundle's own `bundle.validate_bundle(dest)` call must pass
  `require_frames=not no_frames`.

---

### Task 1: `--no-frames` flag, end to end

**Files:**
- Modify: `export_bundle.py` (`export_bundle()`, `main()`)
- Test: `tests/test_export_bundle.py`

**Interfaces:**
- Produces: `export_bundle(..., no_frames: bool = False)` (extended signature, every
  existing call site/test keeps working since it defaults to `False`). CLI:
  `--no-frames` (`store_true`, default `False`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_export_bundle.py`, after `test_export_copies_masks_and_frames`
(after line 116) and before `test_neuron_filter_limits_the_bundle`:

```python
def test_no_frames_skips_frame_generation_entirely(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    dest = tmp_path / "bundle"

    def boom(*a, **k):
        raise AssertionError("regenerate_frames must not be called when no_frames=True")
    monkeypatch.setattr(export_bundle, "regenerate_frames", boom)

    export_bundle.export_bundle(root, dest, source_tree="master", no_frames=True)

    assert not (dest / "AIAL" / "chain_00" / "frames").exists()


def test_no_frames_leaves_frames_dir_none_in_state(tmp_path):
    root = _tree(tmp_path)
    dest = tmp_path / "bundle"
    export_bundle.export_bundle(root, dest, source_tree="master", no_frames=True)
    state = json.loads((dest / "AIAL" / "chain_00" / "state.json").read_text(encoding="utf-8"))
    assert state["frames_dir"] is None


def test_no_frames_bundle_validates_clean_without_require_frames(tmp_path):
    root = _tree(tmp_path)
    dest = tmp_path / "bundle"
    export_bundle.export_bundle(root, dest, source_tree="master", no_frames=True)
    assert bundle.validate_bundle(dest, require_frames=False) == []


def test_no_frames_still_copies_masks_and_qc(tmp_path):
    root = _tree(tmp_path)
    dest = tmp_path / "bundle"
    export_bundle.export_bundle(root, dest, source_tree="master", no_frames=True)
    assert (dest / "AIAL" / "chain_00" / "masks" / "mask_1402.png").read_bytes() == b"px"
    assert (dest / "AIAL" / "chain_00" / "qc.csv").exists()


def test_no_frames_and_force_frames_together_is_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "export_bundle.py", "--output-root", "x", "--dest", "y",
        "--no-frames", "--force-frames"])
    with pytest.raises(SystemExit, match="no-frames.*force-frames|force-frames.*no-frames"):
        export_bundle.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3 -m pytest tests/test_export_bundle.py -v -k no_frames`
Expected: FAIL — the first four with `TypeError: export_bundle() got an unexpected
keyword argument 'no_frames'`, the last with `SystemExit` not raised (today's `main()`
accepts `--no-frames`/`--force-frames` as unrecognized-argument errors from argparse
itself, not the specific combination check this step adds) or an `argparse` error with a
different message than the one the test's `match` expects.

- [ ] **Step 3: Implement `no_frames` in `export_bundle()`**

Change the signature (currently `def export_bundle(output_root: Path, dest: Path, *,
neurons: Optional[List[str]] = None, source_tree: Optional[str] = None, backend: str =
"sam2", reprop_variant: Optional[str] = None, frames_root: Optional[Path] = None,
overlay_roots: Optional[List[Path]] = None, force: bool = False, force_frames: bool =
False) -> dict:`) to add one parameter:

```python
def export_bundle(output_root: Path, dest: Path, *, neurons: Optional[List[str]] = None,
                  source_tree: Optional[str] = None, backend: str = "sam2",
                  reprop_variant: Optional[str] = None, frames_root: Optional[Path] = None,
                  overlay_roots: Optional[List[Path]] = None,
                  force: bool = False, force_frames: bool = False,
                  no_frames: bool = False) -> dict:
```

Add to the docstring's `Parameters` section, right after `force_frames`'s entry:

```python
    no_frames : bool, optional
        Skip generating/shipping frames entirely. Only for a destination reviewer who
        already has her own raw EM and the recrop setup from
        docs/how-to/set-up-sam2-on-a-review-machine.md: gui.py's _ensure_local_frames
        regenerates frames locally from a chain's frames_dir=None, the same mechanism
        import_bundle.py already relies on for a returned recrop. This is a
        trust-the-caller flag like `force`/`force_frames` above; export_bundle.py has
        no way to verify the destination machine actually has raw EM, and using this
        for a reviewer without it ships a bundle that silently fails to open any chain.
```

Replace the per-chain frame block (currently):

```python
        frames_out = out_dir / "frames"
        if _frames_complete(frames_out, rec["state"]) and not force_frames:
            print(f"[export] {rec['chain_dir']}: frames already present, skipping")
        else:
            # Frames are ALWAYS regenerated, never copied from the recorded
            # frames_dir. That directory cannot be trusted, for two independent
            # reasons found on real data. It is usually a dead Compute Canada
            # Narval /localscratch path, deleted when the job ended. And when it
            # does still exist, a `gui.py --anchor-only --context-frames N` review
            # session will have rewritten it to hold only the anchor plus or minus
            # N: the first real export found every one of 18 AIYL chains left with
            # exactly 5 frames, for chains needing anywhere from 1 to 43.
            #
            # A bare directory of numbered jpgs carries nothing that proves which
            # z each frame is, so a count cannot tell a complete set from a
            # narrowed one that happens to be the same size. Regenerating is the
            # only option correct by construction, and a reviewer checking masks
            # against the wrong EM slices is worse than one waiting a bit longer.
            print(f"[export] {rec['chain_dir']}: regenerating frames from raw EM")
            src = regenerate_frames(rec["state"], neuron=rec["cell_name"],
                                    chain_idx=rec["chain_idx"], output_root=output_root,
                                    frames_root=frames_root)
            _replace_frames_dir(src, frames_out)

        state_out = bundle.rewrite_state_frames_dir(rec["state"])
        (out_dir / "state.json").write_text(json.dumps(state_out, indent=2), encoding="utf-8")
```

with:

```python
        if no_frames:
            print(f"[export] {rec['chain_dir']}: --no-frames, not shipping frames "
                 f"(reviewer regenerates locally from her own raw EM)")
            state_out = bundle.rewrite_state_frames_dir(rec["state"], relative=None)
        else:
            frames_out = out_dir / "frames"
            if _frames_complete(frames_out, rec["state"]) and not force_frames:
                print(f"[export] {rec['chain_dir']}: frames already present, skipping")
            else:
                # Frames are ALWAYS regenerated, never copied from the recorded
                # frames_dir. That directory cannot be trusted, for two independent
                # reasons found on real data. It is usually a dead Compute Canada
                # Narval /localscratch path, deleted when the job ended. And when it
                # does still exist, a `gui.py --anchor-only --context-frames N` review
                # session will have rewritten it to hold only the anchor plus or minus
                # N: the first real export found every one of 18 AIYL chains left with
                # exactly 5 frames, for chains needing anywhere from 1 to 43.
                #
                # A bare directory of numbered jpgs carries nothing that proves which
                # z each frame is, so a count cannot tell a complete set from a
                # narrowed one that happens to be the same size. Regenerating is the
                # only option correct by construction, and a reviewer checking masks
                # against the wrong EM slices is worse than one waiting a bit longer.
                print(f"[export] {rec['chain_dir']}: regenerating frames from raw EM")
                src = regenerate_frames(rec["state"], neuron=rec["cell_name"],
                                        chain_idx=rec["chain_idx"], output_root=output_root,
                                        frames_root=frames_root)
                _replace_frames_dir(src, frames_out)
            state_out = bundle.rewrite_state_frames_dir(rec["state"])
        (out_dir / "state.json").write_text(json.dumps(state_out, indent=2), encoding="utf-8")
```

Replace the bundle-wide validation call (currently):

```python
    problems = bundle.validate_bundle(dest)
```

with:

```python
    problems = bundle.validate_bundle(dest, require_frames=not no_frames)
```

- [ ] **Step 4: Implement `--no-frames` in `main()`**

Add the new flag, right after `--force-frames`'s `ap.add_argument` call:

```python
    ap.add_argument("--no-frames", action="store_true",
                    help="skip shipping frames; only for a destination reviewer who "
                         "already has her own raw EM and the recrop setup from "
                         "set-up-sam2-on-a-review-machine.md (gui.py regenerates frames "
                         "locally on open). Do not use this for a reviewer without raw "
                         "EM, the bundle will not open.")
```

Add the combination check right after `args = ap.parse_args()`:

```python
    if args.no_frames and args.force_frames:
        ap.error("--no-frames and --force-frames contradict each other "
                "(never generate vs. always regenerate)")
```

Add `no_frames=args.no_frames` to the `export_bundle(...)` call's keyword arguments.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `py -3 -m pytest tests/test_export_bundle.py -v`
Expected: all PASS, including the 5 new tests and every pre-existing test in the file.

- [ ] **Step 6: Update `docs/how-to/review-over-git.md`**

Add a new section right after "## Building the repository" (before "## The reviewer's
setup"):

```markdown
## A simpler path when the reviewer already has her own raw EM

If the reviewer has gone through
[set-up-sam2-on-a-review-machine.md](set-up-sam2-on-a-review-machine.md) (torch,
checkpoint, and her own raw EM copy, for recrop), she does not need frames shipped at
all: `gui.py`'s chain-loading code already regenerates a chain's frames locally from the
raw EM whenever its recorded `frames_dir` is unset, the same fallback `import_bundle.py`
already relies on for a returned recrop.

```bash
py -3 export_bundle.py --output-root <tree> --dest ~/mask-review/<GROUP>_for_lucinda \
    --neurons <NEURONL> <NEURONR> --no-frames
```

No separate "copy without frames" step is needed (there are no frames to strip), and no
`place_frames.py` step on her side either (there is nothing to place). Her first
`gui.py` open of each chain regenerates it from her own raw EM. Only use this for a
reviewer confirmed to have raw EM access; the rest of this page's frames-shipped path is
unchanged for everyone else.
```

- [ ] **Step 7: Commit**

```bash
git add export_bundle.py tests/test_export_bundle.py docs/how-to/review-over-git.md
git commit -m "feat: export_bundle.py --no-frames for raw-EM-capable reviewers"
```
