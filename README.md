# How each person feels, or which video they watched?

HABS Neuro-AI Hackathon 2026 · team Totally Spies

Emotion decoders trained on video-viewing data can look accurate while only
recognising *which video* was on screen. Our question:

> **Do brain and body signals show how each person feels, or just which video they watched?**

Every video in the dataset was watched by many people who reported different
feelings about it. That lets us separate the part of a rating shared by everyone
(the video) from the part that belongs to the person, and ask which part
physiology tracks.

Results, figures and the full write-up will be added with our final submission.

## Code

| File | Role |
| --- | --- |
| [process.py](process.py) | GPU job: extracts per-trial features from every labelled video trial |
| [features.py](features.py) | the per-trial features, one function per modality |
| [analysis.py](analysis.py) | the experiments, statistics and figures |
| [simulate.py](simulate.py) | fake dataset with a known answer, to check the analysis |

## Reproduce

```bash
pip install -r requirements.txt
pip install "habs-hack[storage] @ git+https://github.com/HABS-BRAIN/habs-hack@main"

habs run --data ./sample --out ./out      # test the extraction on the sample
# full dataset: Actions -> HABS -> Run workflow (runs process.py on /data)
python analysis.py results                # on the downloaded job outputs

python simulate.py sim && python analysis.py sim   # same analysis on data with a known answer
```

The original hackathon instructions (GPU platform, accounts) are in
[HABS-BRAIN/habs-hackathon-template](https://github.com/HABS-BRAIN/habs-hackathon-template).
