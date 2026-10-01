from __future__ import annotations

import ast


def expand_scp_codes(metadata, statements):
    """Expand PTB-XL SCP dictionaries without turning contemporary labels into future outcomes."""
    import pandas as pd

    rows = []
    for row in metadata[["ecg_id", "scp_codes"]].itertuples(index=False):
        codes = ast.literal_eval(row.scp_codes) if isinstance(row.scp_codes, str) else row.scp_codes
        for code, likelihood in codes.items():
            detail = statements.loc[code].to_dict() if code in statements.index else {}
            rows.append(
                {"ecg_id": row.ecg_id, "scp_code": code, "likelihood": likelihood, **detail}
            )
    return pd.DataFrame(rows)


def rhythm_flags(metadata, statements):
    expanded = expand_scp_codes(metadata, statements)
    rhythm = expanded.loc[expanded.get("rhythm", 0).fillna(0).astype(bool)].copy()
    names = rhythm.get("description", rhythm["scp_code"]).astype(str).str.lower()
    rhythm["is_sinus_rhythm"] = names.str.contains("sinus")
    rhythm["is_atrial_fibrillation"] = names.str.contains("atrial fibrillation|atrial flutter")
    return rhythm
