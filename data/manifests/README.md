# Manifestos

`cohort.parquet` é gerado pelo comando `ecg-ai-build-cohort` e não é versionado quando
contém identificadores clínicos. O schema obrigatório é:

`ecg_id`, `subject_id`, `ecg_path`, `exam_datetime`, `is_sinus_rhythm`, `future_af`,
`horizon_days`, `followup_days`, `split`.

O caminho aponta para o stem de um registro WFDB local autorizado. Nunca coloque PHI,
credenciais ou dados brutos no Git.
