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

## Results

| Model                                      | Beehive States Avg | Beijing Opera Percussion | CREMA-D | DCASE16 | ESC-50 | FSD50K | GTZAN Genre | GTZAN Music Speech | Gunshot Triangulation | LibriCount | MAESTRO 5hr | Mridangam Stroke | Mridangam Tonic | NSynth Pitch 50hr | NSynth Pitch 5hr | Speech Commands 5hr | Speech Commands Full | Vocal Imitations | VoxLingua107 Top10 |
|--------------------------------------------|--------------------|--------------------------|---------|---------|--------|--------|-------------|-------------------|----------------------|------------|-------------|------------------|------------------|-------------------|-----------------|--------------------|---------------------|-----------------|--------------------|
| ced-tiny | 38.345             | 94.90                    | 62.52   | 88.02   | 95.80  | 62.73  | 89.20       | 93.01             | 91.67                | 61.26      | 4.81        | 96.13            | 90.74            | 69.19             | 44.00           | 70.53              | 77.10               | 19.18           | 33.64              |
| ced-mini     | 59.17              | 96.18                    | 65.26   | 90.66   | 95.35  | 63.88  | 90.30       | 94.49             | 86.01                | 64.02      | 8.29        | 96.56            | 93.32            | 75.20             | 55.60           | 77.38              | 81.96               | 20.37           | 34.67              |
| ced-small     | 51.70              | 96.60                    | 66.64   | 91.63   | 95.95  | 64.33  | 89.50       | 91.22             | 93.45                | 65.59      | 10.96       | 96.82            | 93.94            | 79.95             | 60.20           | 80.92              | 85.19               | 21.92           | 36.53              |
| ced-base      | 48.35              | 96.60                    | 69.10   | 92.19   | 96.65  | 65.48  | 88.60       | 94.36             | 89.29                | 67.85      | 14.76       | 97.43            | 96.55            | 82.81             | 68.20           | 86.93              | 89.67               | 22.69           | 38.57              |
