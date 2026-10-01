# Relatório final da execução — PTB-XL 1.0.3

## IMPLEMENTADO

- Download retomável via bucket público oficial do PhysioNet, com exclusão explícita de `records500/*`.
- Validação por SHA-256 publicado, leitura integral de cabeçalhos WFDB e decodificação dos sinais.
- Carregamento sob demanda por `ecg_id` e `patient_id`.
- Preservação integral dos metadados originais e catálogo de arquivos.
- EDA com contagens, pacientes, idade, sexo, faltantes, códigos SCP e folds.
- Visualização das 12 derivações e exportação PNG de alta resolução.
- Limpeza auditável com critérios de exclusão e sinalização.
- Pré-processamento configurável com normalização ajustada somente nos folds de treino.
- Exportação Parquet de metadados, índice, pacientes, características auxiliares e manifesto.
- Notebooks Jupyter 01–15 e interface Streamlit.
- CNN 1D, inferência individual, avaliação estatística e testes técnicos implementados.

## EXECUTADO E VALIDADO

- Fonte: `s3://physionet-open/ptb-xl/1.0.3/`.
- Arquivos no diretório bruto: 43.606.
- ECGs no metadado: 21.799.
- ECGs disponíveis e decodificados: 21.799.
- Pares `.hea/.dat`: 21.799 / 21.799.
- Checksums esperados e validados: 43.605 / 43.605.
- Arquivos ausentes, corrompidos e caminhos inválidos: 0 / 0 / 0.
- Arquivos `records500`: 0.
- Pacientes: 18.869.
- Qualidade: 21.797 válidos, 1 excluído (`ecg_id=12722`, derivação constante), 1 sinalizado (`ecg_id=20045`, amplitude extrema).
- Folds PTB-XL: nenhum paciente compartilhado entre folds; verificação crítica aprovada.
- Testes automatizados: 12 aprovados.
- Notebooks 01–15 executados; relatórios, Parquet, gráficos e PNGs foram gerados.

## PENDENTE OU BLOQUEADO

- PTB-XL fornece anotações do ECG contemporâneo, não desfechos futuros de FA.
- Os dados autorizados MIMIC-IV-ECG/MIMIC-IV não estão presentes localmente; a integração foi documentada como bloqueada.
- Não foi criada coorte longitudinal defensável, checkpoint, predição, AUC, ROC ou métrica científica futura-FA.
- `artifacts/model/model.pt`, `outputs/predictions/test_predictions.parquet` e métricas de futuro-FA só devem ser criados após rótulos temporais autorizados, seguimento suficiente e revisão do protocolo.
- Os resultados PTB-XL deste relatório são de engenharia/qualidade e não constituem validação clínica nem previsão de FA futura.
