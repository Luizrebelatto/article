from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecg_ai.config import load_config, resolve_project_path
from ecg_ai.dataset import read_manifest
from ecg_ai.io import ECGValidationError, load_wfdb_ecg, validate_and_reorder
from ecg_ai.predict import DISCLAIMER, predict_ecg
from ecg_ai.visualization import plot_12_leads

st.set_page_config(page_title="Future AF research PoC", layout="wide")
st.title("Predição experimental de FA futura a partir de ECG sinusal")
st.warning(DISCLAIMER)

with st.expander("Avaliação do modelo", expanded=False):
    metrics_path = ROOT / "outputs" / "figures" / "test_metrics.json"
    roc_path = ROOT / "outputs" / "figures" / "roc_future_af.png"
    confusion_path = ROOT / "outputs" / "figures" / "confusion_matrix.png"
    if metrics_path.exists():
        st.json(json.loads(metrics_path.read_text(encoding="utf-8")))
        if roc_path.exists():
            st.image(str(roc_path), caption="Curva ROC no teste")
        if confusion_path.exists():
            st.image(str(confusion_path), caption="Matriz de confusão no teste")
    else:
        st.info("Avaliação indisponível: ainda não há modelo futuro-FA validado e previsões reais do teste.")

config = load_config()
model_dir = resolve_project_path(config["artifacts"]["model_dir"])
os.environ["ECG_AI_MODEL_DIR"] = str(model_dir)
manifest_path = resolve_project_path(config["data"]["manifest_path"])


def show_record(path: str, key: str, truth=None, exam_info=None):
    try:
        record = validate_and_reorder(load_wfdb_ecg(path), minimum_seconds=10.0)
        st.pyplot(plot_12_leads(record), clear_figure=True)
        with st.expander("Informações do exame"):
            st.json(exam_info or {"ecg_id": record.ecg_id, **record.metadata})
        if st.button("Executar predição", key=f"predict-{key}", type="primary"):
            try:
                st.session_state[f"result-{key}"] = predict_ecg(path)
            except FileNotFoundError as exc:
                st.error(str(exc))
        result = st.session_state.get(f"result-{key}")
        if result:
            col1, col2, col3 = st.columns(3)
            col1.metric("Escore não calibrado", f"{result['risk_score']:.4f}")
            col2.metric("Limiar experimental", f"{result['experimental_threshold']:.4f}")
            col3.metric("Classificação", result["classification"])
            st.json(result)
            if truth is not None:
                st.info(f"Desfecho documentado (revelado após a predição): {int(truth)}")
    except (ECGValidationError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        st.error(f"ECG rejeitado: {exc}")


dataset_tab, upload_tab = st.tabs(["Conjunto de teste", "WFDB externo"])
with dataset_tab:
    if not manifest_path.exists():
        st.info("Manifesto de coorte ausente. Construa uma coorte temporalmente validada antes da demonstração.")
    else:
        manifest = read_manifest(manifest_path)
        test = manifest.loc[manifest.split.eq("test")].reset_index(drop=True)
        selected = st.selectbox("ECG", test.ecg_id.tolist())
        row = test.loc[test.ecg_id.eq(selected)].iloc[0]
        # Truth is passed only to the display branch after prediction; never to predict_ecg.
        show_record(
            str(row.ecg_path),
            f"dataset-{selected}",
            truth=row.future_af,
            exam_info={
                "ecg_id": str(row.ecg_id),
                "subject_id": str(row.subject_id),
                "exam_datetime": str(row.exam_datetime),
                "horizon_days": int(row.horizon_days),
            },
        )

with upload_tab:
    st.caption("Envie juntos o cabeçalho .hea e todos os arquivos binários referenciados por ele.")
    files = st.file_uploader("Arquivos WFDB", accept_multiple_files=True, type=["hea", "dat", "mat"])
    if files:
        names = [Path(item.name).name for item in files]
        headers = [name for name in names if name.endswith(".hea")]
        if len(headers) != 1:
            st.error("Envie exatamente um arquivo .hea por vez.")
        else:
            temp = Path(tempfile.mkdtemp(prefix="ecg_ai_upload_"))
            for item in files:
                (temp / Path(item.name).name).write_bytes(item.getvalue())
            show_record(str(temp / Path(headers[0]).stem), f"upload-{headers[0]}")
