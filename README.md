# HEAR_CED

Hear evaluation for [CED models](https://github.com/RicherMans/CED).

## Validate


First install `hear-validator`

```bash
pip3 install hearvalidator
```

```bash
hear-validator hear_ced.ced_tiny
hear-validator hear_ced.ced_mini
hear-validator hear_ced.ced_small
hear-validator hear_ced.ced_base
```


## Evaluate


Install `heareval` and extract the embeddings.

```bash
pip3 install heareval
python3 -m heareval.embeddings.runner hear_ced.ced_mini  --tasks-dir <path to tasks>
python3 -m heareval.predictions.runner embeddings/hear_ced.ced_mini/*
```
