# notebooks/

Analysis notebooks (keep them thin — real logic lives in `src/auralguard`):

- `01_eda.ipynb`        — corpus stats, class balance, duration/lang distributions
- `02_baseline.ipynb`   — reproduce B3/B4/B5, sanity-check in-domain EER
- `03_results.ipynb`    — load `experiments/*/results.json`, build the paper figures
- `04_explainability.ipynb` — spectro-temporal attributions / case studies

Run `pip install -e .` first so `import auralguard` works from any notebook.
