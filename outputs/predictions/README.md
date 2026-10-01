# Predições

`test_predictions.parquet` será criado somente após inferência no conjunto `test` e terá:
`ecg_id`, `subject_id`, `y_true`, `y_score`, `y_pred`, `threshold`, `horizon_days` e
`checkpoint_id`. Nenhum arquivo sintético é fornecido.
