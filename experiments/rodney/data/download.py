"""Fetch the tinyshakespeare corpus.

The corpus is a single plain-text file (~1.1 MB) of concatenated Shakespeare
plays, formatted as speaker headings followed by lines of verse/prose. We keep
it as raw bytes on disk and do all tokenization at load time -- there is no
preprocessing step, because a byte-level tokenizer has nothing to preprocess:
the file *is* the token stream.

Run:  python data/download.py
"""

import hashlib
import pathlib
import sys

import requests

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Known-good size of the canonical file. We check it so that a truncated
# download or a silently-changed upstream file fails loudly here, rather than
# showing up 3000 training steps later as a mysteriously wrong loss. The sha256
# is printed (not asserted) so you can record it in your report as the exact
# corpus these results came from.
EXPECTED_BYTES = 1_115_394

DEST = pathlib.Path(__file__).parent / "tinyshakespeare.txt"


def main() -> int:
    if DEST.exists():
        raw = DEST.read_bytes()
        print(f"already present: {DEST}  ({len(raw):,} bytes)")
    else:
        print(f"downloading {URL}")
        resp = requests.get(URL, timeout=30)
        resp.raise_for_status()
        raw = resp.content
        DEST.write_bytes(raw)
        print(f"wrote {DEST}  ({len(raw):,} bytes)")

    sha = hashlib.sha256(raw).hexdigest()
    print(f"sha256: {sha}")

    if len(raw) != EXPECTED_BYTES:
        print(
            f"WARNING: expected {EXPECTED_BYTES:,} bytes, got {len(raw):,}. "
            "The upstream file may have changed; results will not match the "
            "numbers reported in this project.",
            file=sys.stderr,
        )
        return 1

    # A byte-level model needs no vocabulary file: the vocabulary is the 256
    # possible byte values, fixed in advance. But it is worth knowing how many
    # of those 256 actually occur, because that tells us how much of the output
    # softmax is dead weight (see model.py / evaluate.py).
    distinct = len(set(raw))
    is_ascii = max(raw) < 128
    print(f"distinct byte values present: {distinct} / 256   (pure ASCII: {is_ascii})")
    print(f"first 120 bytes:\n{raw[:120].decode('utf-8')!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
