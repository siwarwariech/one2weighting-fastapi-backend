# app/utils/encoding.py
from builtins import UnicodeDecodeError, str
import csv, os, pandas as pd, chardet
from pathlib import Path

def _guess_delimiter(first_line: str) -> str:
    for delim in (",", ";", "\t", "|"):
        if first_line.count(delim) >= 1:
            return delim
    return ","            # fallback

def smart_csv(path: str | os.PathLike, **kwargs):
    p = Path(path)

    # -------- encodage --------
    for enc in ("utf-8", "latin-1"):
        try:
            first_line = p.open("r", encoding=enc).readline()
            break
        except UnicodeDecodeError:
            continue
    else:
        enc = chardet.detect(p.read_bytes()[:32_000])["encoding"] or "latin-1"
        first_line = p.open("r", encoding=enc).readline()

    # -------- séparateur --------
    delim = _guess_delimiter(first_line)

    return pd.read_csv(p, encoding=enc, sep=delim, **kwargs)
