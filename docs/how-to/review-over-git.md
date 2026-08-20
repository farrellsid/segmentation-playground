# Run a review over git

For a reviewer on another machine whose corrections need to come back continuously,
rather than once when a drive is handed over. The alternative, a zipped bundle carried
on a drive, is described in [review-on-a-mac.md](review-on-a-mac.md); this builds on it
rather than replacing it.

## The split

A bundle divides very unevenly, which is what makes this work:

| | Size | Changes | Travels by |
|---|---|---|---|
| Masks and metadata | ~9 MB | constantly | git |
| EM frames | ~1.9 GB | never | drive, once |

Frames are static once a bundle is built and can be regenerated from the raw EM, so
committing them would add nearly two gigabytes of history to carry something that never
changes. Keeping them out leaves a repository that clones in seconds and whose history
is a readable record of the corrections.

## Building the repository

Export bundles as usual, then copy them without their frames:

```python
import shutil, pathlib
src = pathlib.Path(r"F:\Lucinda_Review\bundles")
dst = pathlib.Path(r"D:\mask-review")
for b in sorted(p for p in src.iterdir() if p.is_dir()):
    shutil.copytree(b, dst / b.name,
                    ignore=lambda d, names: ["frames"] if "frames" in names else [])
```

The repository needs two files of its own. A `.gitignore` holding `frames/` and `*.jpg`,
so a stray frame can never be committed. And a `.gitattributes` holding `* text=auto
eol=lf` with `*.png binary`, because the repository is written from Windows and edited on
a Mac: without it every `qc.csv` reads as entirely rewritten the first time the other
side touches it, and git would try to merge two PNGs as text, which yields a corrupt file
rather than an error.

Do not create the repository on the drive itself. exFAT records no ownership, so git
refuses to operate there without an explicit `safe.directory` exception.

## The reviewer's setup

Three things have to meet: the repository, the review code, and the frames.

```bash
git clone --depth 1 --branch repo-reorg     https://github.com/farrellsid/segmentation-playground.git ~/segmentation-playground
git clone https://github.com/farrellsid/mask-review.git ~/mask-review

python3 -m venv ~/review-env
source ~/review-env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r ~/segmentation-playground/requirements-review.txt

python3 place_frames.py --clone ~/mask-review/AIY_for_lucinda \
    --from /Volumes/Expansion/Lucinda_Review/bundles/AIY_for_lucinda
```

`place_frames.py` matches chain directories between the two and reports any chain the
drive turns out not to have, rather than leaving a hole to be discovered mid-review. It
skips chains that already have frames, so an interrupted copy resumes. It prints
`clone validates clean and is ready to review` when the bundle is whole.

Install into a virtual environment rather than with `pip install --user`. Recent macOS
Pythons refuse a user install outright with `externally-managed-environment` (PEP 668),
and an environment that can be deleted and rebuilt in one step is worth more than the one
line it costs. Every later command needs `source ~/review-env/bin/activate` first, or
`python3` will not find napari.

She then reviews out of the clone, not out of the drive. That is what makes her work
committable.

## Getting the work back

She commits and pushes whenever she finishes a stretch. On this side:

```bash
git -C ~/mask-review pull
py -3 import_bundle.py --bundle ~/mask-review/AIY_for_lucinda \
    --output-root "F:\ZhenLab\Data\output_masks\manual_verify_AIYL_AIYR" --dry-run
```

Read the dry run, then repeat without `--dry-run`.

Import never reads a frame. It moves masks, `qc.csv`, and the reviewer's rows in
`_review.csv` and `_labels.csv`, so it works against a frames-less clone directly and no
EM has to exist on the importing machine at all.

Expect the first import to report far more chains than she has reviewed. A bundle built
with `--overlay` carries the re-propagated masks, while the base tree still holds the
pre-re-propagation ones, so that first merge carries the re-propagation results across
as well. That is the intended outcome: it leaves the base tree as the single master
holding both.

## Two constraints that are not optional

**One editor per mask.** Git cannot merge two versions of a PNG. New chains may be added
from this side, but nobody else edits a mask the reviewer may be working on, or one of
the two versions has to be discarded.

**Frames stay out of git.** `.gitignore` enforces it. A repository that starts
accumulating frames loses the property that makes this arrangement worth having.

## Validating

`validate_bundle` takes a side on whether frames must be present:

```python
bundle.validate_bundle(root)                        # a delivered bundle: frames required
bundle.validate_bundle(root, require_frames=False)  # a clone: frames arrive separately
```

The launcher requires them, since it is the picker a chain is about to be opened from.
`import_bundle` does not, since it never reads one.
