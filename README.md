# PoC de predição precoce de fibrilação atrial

Esta é uma implementação de pesquisa, ponta a ponta, para estimar **risco de fibrilação
atrial (FA) futura** a partir de um ECG de 12 derivações em ritmo sinusal. O software não
é um dispositivo médico e não possui validação clínica.

## Estado científico desta cópia

O repositório contém a engenharia de dados, CNN 1D, inferência individual, avaliação,
exportação PNG, notebook e interface. Ele **não contém dados clínicos credenciados nem um
checkpoint treinado**. Portanto, esta cópia não apresenta resultados, AUC ou probabilidades
inventadas. Treinamento e predição ficam bloqueados até que exista uma coorte com datas e
seguimento defensáveis.

Para o caso de uso de 12 derivações, uma fonte apropriada no PhysioNet é o
[MIMIC-IV-ECG v1.0](https://physionet.org/content/mimic-iv-ecg/1.0/), associado a eventos
clínicos datados do MIMIC-IV. Ambos exigem acesso autorizado e a
construção do rótulo precisa ser revisada sob o protocolo e a governança locais. Bases como
[PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) podem validar o processamento de sinais,
mas seus diagnósticos no exame não devem ser
reapresentados como desfechos futuros.

O próprio PhysioNet alerta que o relógio do equipamento de ECG pode estar dessincronizado e
que parte dos exames ocorreu fora das visitas cobertas pelo banco clínico. Essas limitações
precisam entrar na auditoria temporal da coorte; o vínculo por `subject_id` sozinho não prova
seguimento completo.

## Contrato dos dados

O construtor recebe dois CSV/Parquet preparados a partir das tabelas autorizadas.

`studies`:

- `ecg_id`, `subject_id`, `ecg_path` (stem WFDB ou `.hea`)
- `exam_datetime`
- `is_sinus_rhythm` — anotação disponível no momento do ECG
- `last_observed_at` — fim defensável do acompanhamento

`events`:

- `subject_id`, `event_datetime`, `event_type`
- FA documentada deve usar `event_type = AF`

O algoritmo inclui somente ECGs sinusais, procura FA estritamente depois do índice e dentro
do horizonte, censura negativos com seguimento insuficiente, usa um índice por paciente e
divide por paciente. A origem clínica das colunas deve ser documentada no protocolo do
estudo; o código não adivinha FA por códigos sem data.

## Instalação

Use Python 3.11–3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

No Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

Como alternativa reproduzível, `uv sync --extra app --extra dev` usa o `uv.lock` versionado.

## Download e integridade do PTB-XL

O download padrão inclui todos os metadados e `records100/` da versão 1.0.3. `records500/`
é explicitamente excluído:

```bash
python scripts/download_dataset.py
```

O comando usa o bucket público oficial, retoma arquivos existentes, confere os SHA-256
publicados, abre todos os cabeçalhos e decodifica todos os sinais WFDB. O relatório fica em
`outputs/reports/download_report.json`; falhas de rede são preservadas em `download.log`.
Para repetir somente a validação:

```bash
python scripts/download_dataset.py --validate-only
```

Os dados brutos permanecem em `data/raw/ptb-xl/`. Nenhuma limpeza altera esses arquivos.

## Ordem dos notebooks

Execute `01_download.ipynb` até `12_export_dataset.ipynb` em ordem. Os notebooks 13–15
só treinam, avaliam e demonstram inferência quando a coorte longitudinal estiver validada.
Todos localizam a raiz automaticamente quando iniciados dentro de `notebooks/`.

1. Download e validação integral.
2. Leitura de todas as colunas e catálogo de arquivos.
3. EDA.
4. Visualização e PNGs derivados.
5. Qualidade e limpeza auditável.
6. Pré-processamento; normalização ajustada apenas no treino.
7–9. Viabilidade MIMIC, linha temporal e coorte.
10–12. Características, validação e exportação.
13–15. Treino, avaliação e inferência, condicionados a rótulos futuros válidos.

Os folds oficiais do PTB-XL são usados apenas para experimentos auxiliares: 1–8 treino,
9 validação e 10 teste. PTB-XL e MIMIC-IV nunca são unidos por paciente.

## Execução reproduzível

```bash
ecg-ai-build-cohort \
  --studies data/private/studies.parquet \
  --events data/private/events.parquet \
  --horizon-days 365 \
  --minimum-followup-days 365 \
  --output data/manifests/cohort.parquet

ecg-ai-train --config configs/default.json
ecg-ai-batch-predict --config configs/default.json
ecg-ai-evaluate --config configs/default.json
streamlit run app/streamlit_app.py
```

Os artefatos esperados após uma execução real são:

- `artifacts/model/model.pt` e `metadata.json`
- `outputs/predictions/test_predictions.parquet`
- `outputs/figures/roc_future_af.png`
- distribuição dos escores, curva de calibração e métricas JSON

O limiar é escolhido **apenas na validação**. A avaliação final usa somente o teste. A saída
sigmoid permanece descrita como `uncalibrated_score`; uma curva de confiabilidade e Brier
score isolados não bastam para promovê-la automaticamente a probabilidade clínica.

## Predição individual

```python
from ecg_ai.predict import predict_ecg

result = predict_ecg("/caminho/para/registro_wfdb")
```

O caminho pode apontar para o stem ou para `.hea`/`.dat`. A função valida 12 derivações,
frequência, duração, valores finitos e aplica exatamente o pré-processamento salvo no bundle.
Identificadores, datas, rótulos e dados posteriores nunca entram no tensor.

## Imagens

O PNG é apenas uma visualização de alta resolução:

```bash
python scripts/export_ecg_png.py /caminho/registro outputs/figures/ecg.png
```

Não há CNN 2D nesta PoC. Imagens não são encaminhadas à CNN 1D e não são tratadas como
equivalentes ao sinal WFDB.

## Limpeza, elegibilidade e proveniência

`outputs/reports/data_quality.csv` registra, por ECG, problemas estruturais e sinalizações de
amplitude ou ruído. Forma, frequência, duração, derivações, finitude, sinais constantes e
decodificação são verificadas. Sinais apenas atípicos são sinalizados, não eliminados sem
justificativa. `data/processed/manifests/` documenta fonte, versão, folds e finalidade.

Os sinais não são armazenados em um DataFrame gigante: `ecg_index.parquet` referencia os
WFDB e o carregador materializa um ECG por vez. Todos os metadados originais são preservados
em `metadata.parquet`; variáveis posteriores não se tornam entradas do modelo.

## Detecção contemporânea versus previsão futura

Classificar `AFIB` versus `SR` no PTB-XL responde se FA está anotada **no ECG atual**. O
objetivo científico deste projeto é diferente: usar um ECG sinusal como índice e prever FA
documentada posteriormente. Somente a segunda tarefa, com data índice, exclusão de FA prévia,
evento posterior e acompanhamento suficiente, pode alimentar a CNN de futuro-FA.

Ausência de diagnóstico não transforma um paciente em negativo. Casos com seguimento
incompleto ou temporalidade ambígua são indeterminados e excluídos da tarefa supervisionada.

## Arquitetura e avaliação

A CNN 1D recebe `[batch, 12, amostras]`, usa blocos convolucionais residuais, BatchNorm,
ReLU, pooling global, `BCEWithLogitsLoss`, AdamW, early stopping e checkpoint por AUC de
validação. O pré-processamento salvo no bundle é reutilizado sem alteração na inferência.

A avaliação calcula ROC-AUC, PR-AUC, sensibilidade, especificidade, precisão, recall, F1,
matriz de confusão, Brier score, curva de confiabilidade e IC de 95% da AUC por bootstrap no
nível do paciente. O teste nunca seleciona modelo ou limiar.

## Testes e controles

```bash
pytest
python -m compileall -q src app scripts
```

Os testes cobrem download, WFDB, derivações, modelo, bundle, avaliação e regras temporais da coorte. Antes de
qualquer alegação científica, ainda são necessários revisão de rótulos, análise de viés,
validação externa, intervalos de confiança, protocolo congelado e avaliação clínica.

## Referências

- Wagner et al. *PTB-XL, a large publicly available electrocardiography dataset*.
  Scientific Data, 2020. PhysioNet v1.0.3: https://physionet.org/content/ptb-xl/1.0.3/
- Gow et al. *MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset*.
  PhysioNet v1.0: https://physionet.org/content/mimic-iv-ecg/1.0/
- Johnson et al. *MIMIC-IV*, PhysioNet v3.1: https://physionet.org/content/mimiciv/3.1/
- Goldberger et al. *PhysioBank, PhysioToolkit, and PhysioNet*. Circulation, 2000.
