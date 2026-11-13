"""Helpers to export a fitted scikit-learn pipeline to ONNX and describe its inputs
for the web demo (web/meta.json)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType


def onnx_input_types(X: pd.DataFrame):
    """One ONNX input per column: float for numbers, string for categories."""
    types = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            types.append((col, FloatTensorType([None, 1])))
        else:
            types.append((col, StringTensorType([None, 1])))
    return types


def export_onnx(pipeline, X: pd.DataFrame, path, classifier=True):
    options = {id(pipeline): {"zipmap": False}} if classifier else None
    onx = convert_sklearn(pipeline, initial_types=onnx_input_types(X),
                          options=options, target_opset=17)
    Path(path).write_bytes(onx.SerializeToString())
    return onx


def check_onnx(path, pipeline, X: pd.DataFrame, classifier=True, n=200):
    """Make sure the ONNX model gives the same answers as scikit-learn."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    sample = X.head(n)
    feeds = {}
    for col in sample.columns:
        v = sample[col].to_numpy().reshape(-1, 1)
        feeds[col] = v.astype(np.float32) if pd.api.types.is_numeric_dtype(sample[col]) else v.astype(str)
    outs = sess.run(None, feeds)
    if classifier:
        ref = pipeline.predict_proba(sample)
        got = outs[1]
    else:
        ref = pipeline.predict(sample).reshape(-1)
        got = outs[0].reshape(-1)
    err = float(np.max(np.abs(ref - got)))
    assert err < 5e-3, f"ONNX mismatch: max abs diff {err}"
    return err


def feature_schema(X: pd.DataFrame, labels=None, helps=None, option_labels=None):
    """Describe each input column for the web form."""
    labels, helps, option_labels = labels or {}, helps or {}, option_labels or {}
    out = []
    for col in X.columns:
        s = X[col]
        f = {"name": col, "label": labels.get(col, col.replace("_", " ").capitalize())}
        if col in helps:
            f["help"] = helps[col]
        if pd.api.types.is_numeric_dtype(s):
            is_int = bool(np.all(np.mod(s.dropna(), 1) == 0))
            f.update(type="number",
                     min=float(s.min()), max=float(s.max()),
                     step=1 if is_int else "any",
                     default=float(s.median()) if not is_int else int(s.median()))
        else:
            opts = sorted(s.dropna().astype(str).unique().tolist())
            f.update(type="category", options=opts, default=s.mode().iloc[0])
            if col in option_labels:
                f["labels"] = option_labels[col]
        out.append(f)
    return out


def examples(X: pd.DataFrame, n=12, seed=0):
    rows = X.sample(n, random_state=seed)
    return json.loads(rows.to_json(orient="records"))


def write_meta(path, meta: dict):
    Path(path).write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
