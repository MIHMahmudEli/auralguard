.PHONY: help setup test lint manifests train eval baseline repro clean

help:
	@echo "setup      install package + dev deps"
	@echo "test       run unit tests"
	@echo "lint       ruff + black --check"
	@echo "manifests  build all dataset manifests"
	@echo "train      train AuralGuard (experiment=auralguard)"
	@echo "baseline   train the strong baseline (b5_wavlm_ocs)"
	@echo "eval       evaluate best checkpoint on the full protocol"
	@echo "repro      baselines + proposed + evals (long)"

setup:
	pip install -e .[train,serve,dev]

test:
	pytest -q

lint:
	ruff check src tests && black --check src tests

download:
	python scripts/download_all.py --all --asvspoof-url "${ASVSPOOF_URL}"

download-aug:
	python scripts/download_all.py --aug

manifests:
	python scripts/build_manifests.py --all

train:
	python scripts/train.py experiment=auralguard

baseline:
	python scripts/train.py experiment=b5_wavlm_ocs

eval:
	python scripts/evaluate.py --ckpt experiments/auralguard/checkpoints/best.ckpt

repro: baseline train eval

clean:
	rm -rf experiments/*/checkpoints .pytest_cache __pycache__
