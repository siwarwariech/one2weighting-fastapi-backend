# app/routes/targets.py
from builtins import dict, float, int, len, str
from pathlib import Path
from typing import List, Dict
from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
from app.utils.encoding import smart_csv       # utf-8 ➜ latin-1 fallback

UPLOAD_DIR = Path("uploads")

router = APIRouter(prefix="/api/projects/{proj_id}")

# ---------- helper -----------------------------------------------------------
def load_df(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return smart_csv(path, nrows=nrows)
    return pd.read_excel(path, nrows=nrows)

# ---------- 1) survey distribution -------------------------------------------
@router.get("/survey-dist", response_model=Dict[str, Dict[str, float]])
def survey_distribution(proj_id: int):
    """Return, for every column, the percentage distribution of the survey."""
    csv = UPLOAD_DIR / f"{proj_id}_survey.csv"
    xls = UPLOAD_DIR / f"{proj_id}_survey.xlsx"
    if not csv.exists() and not xls.exists():
        raise HTTPException(404, "Survey not uploaded")

    df  = load_df(csv if csv.exists() else xls)
    tot = len(df)

    out: dict[str, dict[str, float]] = {}
    for col in df.columns:
        ser = df[col].dropna()
        out[col] = (
            ser.value_counts(normalize=True)    # proportion
               .mul(100)
               .round(2)
               .to_dict()
        )
    return out


# ---------- 2) upload target file --------------------------------------------
@router.post("/upload-target/{var_name}")
async def upload_target(proj_id: int, var_name: str, file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in {".csv", ".xlsx"}:
        raise HTTPException(400, "File must be .csv or .xlsx")

    dest = UPLOAD_DIR / f"{proj_id}_target_{var_name}{ext}"
    dest.write_bytes(await file.read())
    return {"ok": True}
