from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT_SETUP = """from pathlib import Path
import sys
ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / 'src'))
print('Project root:', ROOT)"""


def notebook(title: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(f"# {title}")] + cells
    nb.metadata.kernelspec = {"display_name": "ECG AI", "language": "python", "name": "ecg-ai"}
    nb.metadata.language_info = {"name": "python", "version": "3.12"}
    return nb


def code(source: str):
    return nbf.v4.new_code_cell(source)


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source)


NOTEBOOKS = {
    "01_download.ipynb": notebook(
        "01 — Download integral do PTB-XL 1.0.3 (100 Hz)",
        [
            code(ROOT_SETUP),
            markdown("O sync é retomável e exclui explicitamente `records500/*`. A validação decodifica todos os sinais."),
            code("""import shutil
from ecg_ai.download import DownloadReport, _utc_now, download_ptbxl_100hz, validate_ptbxl, write_report
usage = shutil.disk_usage(ROOT)
print({'free_gib': round(usage.free / 2**30, 2), 'aws_cli': shutil.which('aws')})
destination = ROOT / 'data/raw/ptb-xl'
report = DownloadReport(destination=str(destination), started_at=_utc_now(), status='downloading')
report.download_exit_code = download_ptbxl_100hz(destination, ROOT / 'outputs/reports/download.log')
report = validate_ptbxl(destination, read_signals=True, report=report)
write_report(report, ROOT / 'outputs/reports/download_report.json')
report"""),
        ],
    ),
    "02_read_database.ipynb": notebook(
        "02 — Leitura integral dos metadados",
        [
            code(ROOT_SETUP),
            code("""import pandas as pd
from ecg_ai.analysis import process_metadata
from ecg_ai.data_loader import load_ptbxl_metadata
raw = ROOT / 'data/raw/ptb-xl'
summary = process_metadata(raw, ROOT / 'data/processed', ROOT / 'outputs/reports')
metadata, statements = load_ptbxl_metadata(raw)
display(summary)
display(metadata.head())
display(pd.DataFrame({'dtype': metadata.dtypes.astype(str), 'missing': metadata.isna().sum(), 'unique': metadata.nunique(dropna=True)}))
print('Todas as colunas:', metadata.columns.tolist())
display(statements.head())"""),
        ],
    ),
    "03_exploratory_analysis.ipynb": notebook(
        "03 — Análise exploratória",
        [
            code(ROOT_SETUP),
            code("""from ecg_ai.analysis import write_eda
from ecg_ai.data_loader import load_ptbxl_metadata
metadata, statements = load_ptbxl_metadata(ROOT / 'data/raw/ptb-xl')
summary = write_eda(metadata, statements, ROOT / 'outputs/figures/eda', ROOT / 'outputs/reports')
summary"""),
            markdown("Os rótulos PTB-XL descrevem o ECG contemporâneo. AFIB versus SR não é apresentado como previsão futura."),
        ],
    ),
    "04_ecg_visualization.ipynb": notebook(
        "04 — Visualização das 12 derivações",
        [
            code(ROOT_SETUP),
            code("""import ast
import matplotlib.pyplot as plt
from ecg_ai.contracts import ECGRecord, LEADS_12
from ecg_ai.data_loader import load_ecg, load_ptbxl_metadata
from ecg_ai.visualization import plot_12_leads, export_ecg_png
metadata, _ = load_ptbxl_metadata(ROOT / 'data/raw/ptb-xl')
def has_code(value, code): return code in ast.literal_eval(value)
examples = {}
for label, scp in [('sinus', 'SR'), ('atrial_fibrillation', 'AFIB')]:
    match = metadata.loc[metadata.scp_codes.map(lambda value: has_code(value, scp))]
    if not match.empty: examples[label] = int(match.iloc[0].ecg_id)
examples"""),
            code("""for label, ecg_id in examples.items():
    signal, meta = load_ecg(ecg_id, 100, ROOT / 'data/raw/ptb-xl')
    record = ECGRecord(str(ecg_id), signal, 100, LEADS_12, tuple(meta['units']), meta, ROOT / meta['filename_lr'])
    fig = plot_12_leads(record, title=f\"{label} — ECG {ecg_id} — {meta['scp_codes']}\")
    display(fig)
    plt.close(fig)
    export_ecg_png(record, ROOT / f'outputs/figures/ecg/{label}_{ecg_id}.png')"""),
        ],
    ),
    "05_data_cleaning.ipynb": notebook(
        "05 — Limpeza e qualidade dos sinais",
        [
            code(ROOT_SETUP),
            markdown("Anomalias de amplitude/ruído são sinalizadas; somente falhas estruturais são excluídas automaticamente."),
            code("""from ecg_ai.analysis import assess_quality
from ecg_ai.data_loader import load_ptbxl_metadata
metadata, _ = load_ptbxl_metadata(ROOT / 'data/raw/ptb-xl')
quality = assess_quality(metadata, ROOT / 'data/raw/ptb-xl', ROOT / 'outputs/reports/data_quality.csv')
display(quality.status.value_counts())
display(quality.loc[quality.status.ne('valid')].head(20))"""),
        ],
    ),
    "06_preprocessing.ipynb": notebook(
        "06 — Pré-processamento e normalização de treino",
        [
            code(ROOT_SETUP),
            code("""import json
from dataclasses import asdict
from ecg_ai.config import load_config
from ecg_ai.contracts import SignalSpec
from ecg_ai.data_loader import load_ptbxl_metadata
from ecg_ai.preprocessing import fit_training_normalization
metadata, _ = load_ptbxl_metadata(ROOT / 'data/raw/ptb-xl')
train = metadata.query('strat_fold <= 8').copy()
train['ecg_path'] = train.filename_lr.map(lambda value: str(ROOT / 'data/raw/ptb-xl' / value))
spec = fit_training_normalization(train, SignalSpec.from_dict(load_config(ROOT / 'configs/default.json')['signal']))
target = ROOT / 'data/processed/manifests/preprocessing.json'
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(asdict(spec), indent=2))
asdict(spec)"""),
            markdown("Os parâmetros são estimados somente nos folds oficiais 1–8 e reutilizados em validação, teste e inferência."),
        ],
    ),
    "07_database_integration.ipynb": notebook(
        "07 — Viabilidade de integração MIMIC",
        [
            code(ROOT_SETUP),
            code("""from ecg_ai.feasibility import write_feasibility_report
report = write_feasibility_report(ROOT, ROOT / 'outputs/reports/longitudinal_feasibility.json')
report"""),
        ],
    ),
    "08_patient_timeline.ipynb": notebook(
        "08 — Linha temporal dos pacientes",
        [
            code(ROOT_SETUP),
            code("""import json
report = json.loads((ROOT / 'outputs/reports/longitudinal_feasibility.json').read_text())
if not report['predictive_training_allowed']:
    print('BLOQUEADO:', report['reason'])
else:
    print('Dados presentes; revisão de schema e proveniência ainda obrigatória.')"""),
            markdown("Datas de documentação, alta e início da doença não são consideradas equivalentes."),
        ],
    ),
    "09_cohort_creation.ipynb": notebook(
        "09 — Construção da coorte longitudinal",
        [
            code(ROOT_SETUP),
            code("""import json
from ecg_ai.feasibility import longitudinal_feasibility
status = longitudinal_feasibility(ROOT)
target = ROOT / 'outputs/reports/cohort_status.json'
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(status, indent=2))
print('Treinamento preditivo permitido:', status['predictive_training_allowed'])
print(status['reason'])"""),
            markdown("Sem os dados MIMIC autorizados, pacientes não são convertidos em negativos e nenhum rótulo futuro é fabricado."),
        ],
    ),
    "10_feature_engineering.ipynb": notebook(
        "10 — Características auxiliares",
        [
            code(ROOT_SETUP),
            code("""import pandas as pd
metadata = pd.read_parquet(ROOT / 'data/processed/metadata.parquet')
quality = pd.read_csv(ROOT / 'outputs/reports/data_quality.csv')
features = metadata[['ecg_id', 'patient_id', 'age', 'sex', 'height', 'weight']].merge(
    quality[['ecg_id', 'max_abs', 'minimum_lead_std', 'noise_diff_ratio']], on='ecg_id', how='left', validate='one_to_one')
features.to_parquet(ROOT / 'data/processed/features.parquet', index=False)
display(features.describe(include='all'))"""),
            markdown("As métricas de qualidade vêm do sinal. Variáveis posteriores ao ECG não são incluídas. Intervalos clínicos exigem um algoritmo validado e não são inventados a partir de 10 s."),
        ],
    ),
    "11_dataset_validation.ipynb": notebook(
        "11 — Validação do dataset",
        [
            code(ROOT_SETUP),
            code("""import json
from ecg_ai.data_loader import load_ptbxl_metadata
from ecg_ai.export import validate_ptbxl_splits
metadata, _ = load_ptbxl_metadata(ROOT / 'data/raw/ptb-xl')
validation = validate_ptbxl_splits(metadata)
target = ROOT / 'outputs/reports/dataset_validation.json'
target.write_text(json.dumps(validation, indent=2))
validation"""),
            markdown("Essa validação cobre o experimento auxiliar PTB-XL. Ela não cria ou valida desfechos de FA futura."),
        ],
    ),
    "12_export_dataset.ipynb": notebook(
        "12 — Exportação dos dados processados",
        [
            code(ROOT_SETUP),
            code("""from ecg_ai.data_loader import load_ptbxl_metadata
from ecg_ai.export import export_ptbxl_auxiliary
metadata, _ = load_ptbxl_metadata(ROOT / 'data/raw/ptb-xl')
manifest = export_ptbxl_auxiliary(metadata, ROOT / 'data/raw/ptb-xl', ROOT / 'data/processed')
manifest"""),
            markdown("`train.parquet`, `validation.parquet` e `test.parquet` preditivos não são exportados sem coorte longitudinal validada."),
        ],
    ),
    "13_model_training.ipynb": notebook(
        "13 — Treinamento da CNN",
        [
            code(ROOT_SETUP),
            code("""import json
status = json.loads((ROOT / 'outputs/reports/cohort_status.json').read_text())
if status['predictive_training_allowed']:
    from ecg_ai.train import train
    result = train(ROOT / 'configs/default.json')
else:
    result = {'status': 'blocked', 'reason': status['reason'], 'checkpoint_created': False}
    (ROOT / 'outputs/reports/model_training_status.json').write_text(json.dumps(result, indent=2))
result"""),
        ],
    ),
    "14_model_evaluation.ipynb": notebook(
        "14 — Curva ROC e avaliação",
        [
            code(ROOT_SETUP),
            code("""import json
predictions = ROOT / 'outputs/predictions/test_predictions.parquet'
if predictions.exists():
    import pandas as pd
    from ecg_ai.evaluate import evaluate_predictions
    metrics = evaluate_predictions(pd.read_parquet(predictions), ROOT / 'outputs/figures')
else:
    metrics = {'status': 'blocked', 'reason': 'No valid future-AF model/test predictions are available.'}
    (ROOT / 'outputs/reports/model_evaluation_status.json').write_text(json.dumps(metrics, indent=2))
metrics"""),
        ],
    ),
    "15_individual_prediction.ipynb": notebook(
        "15 — Predição individual",
        [
            code(ROOT_SETUP),
            code("""model = ROOT / 'artifacts/model/model.pt'
if not model.exists():
    print('BLOQUEADO: nenhum modelo de FA futura treinado e validado; nenhuma saída aleatória será mostrada.')
else:
    print('Use predict_ecg em um registro do manifesto de teste validado.')"""),
            markdown("Quando houver resultados reais, TP/TN/FP/FN serão mostrados somente para categorias existentes."),
        ],
    ),
}


def main() -> None:
    destination = Path(__file__).resolve().parents[1] / "notebooks"
    destination.mkdir(parents=True, exist_ok=True)
    for name, nb in NOTEBOOKS.items():
        nbf.write(nb, destination / name)
        print(destination / name)


if __name__ == "__main__":
    main()
