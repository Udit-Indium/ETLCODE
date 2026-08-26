from typing import List


def source_window(lines: List[str], line_number: int, window: int) -> str:
    """Return `window` lines around `line_number` (1-indexed), annotated with >>> markers."""
    start = max(0, line_number - window - 1)
    end = min(len(lines), line_number + window)
    out = []
    for i, ln in enumerate(lines[start:end], start=start + 1):
        marker = ">>>" if i == line_number else "   "
        out.append(f"{marker} {i:4d} | {ln.rstrip()}")
    return "\n".join(out)
