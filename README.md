## Geometric Data Analysis Project — Kernel Methods Through the Roof

Small experimental codebase to reproduce core ideas from  
**Kernel Methods Through the Roof: Handling Billions of Points Efficiently** ([NeurIPS 2020 paper](https://proceedings.neurips.cc/paper/2020/file/a59afb1b7d82ec353921a55c579ee26d-Paper.pdf)).

Project developed for the MVA course  
**[Geometric Data Analysis](https://www.jeanfeydy.com/Teaching/index.html)**.


## Getting Started: Installation & Requirements

### Essential Requirement: CUDA/GPU
This project requires a working **CUDA** setup and a compatible GPU due to its dependency on the **[FALKON](https://falkonml.github.io/falkon/)** library. Installation and import will fail if GPU support is not available.

The code was tested with:

* `torch==2.4`
* `CUDA 11.8 (cu118)`

Please refer to the FALKON documentation for compatible drivers and CUDA versions.


### Installation (using `uv`)

This codebase is designed to be used with **[uv](https://docs.astral.sh/uv/getting-started/installation/)**. If you do not use `uv`, you must set up the environment manually (not detailled here). The steps to run our code are : 

1.  Clone the repository.
2.  Install in editable mode:
    ```bash
    uv run pip install -e .
    ```
3.  Run any script from the `scripts/` directory:
    ```bash
    uv run  scripts/<script_name>.py
    ```


## Repository Structure

```text
.
├── data/
│   ├── download.sh        – Dataset download helper (YearPredictionMSD / HIGGS)
│   ├── higgs.py           – Convert raw HIGGS CSV → clean parquet
│   └── make_mini_higgs.py – Create balanced mini-HIGGS parquet subset
│
├── scripts/
│   ├── benchmark.py – CLI benchmark runner (dataset × solver)
│   ├── benchmark_logm.py – Benchmark over logarithmic ranges of m (for scaling curves)
│   ├── math_asymptotes.py – Theoretical asymptotic slope visualisations
│   └── time_benchmarking.py – Runtime scaling benchmarks of different solvers
│
├── src/
│   └── kmtr/
│       ├── datasets_and_metrics.py – Loaders for datasets (MDS, HIGGS, mini_Higgs) and metrics
│       └── kernel_solvers.py – Implementations of KRR variants
│
├── notebooks/ – Exploratory and report notebooks
├── outputs/ – CSV benchmark results
├── figures/ – Generated figures
├── report/  – Our report 
└── pyproject.toml – Project configuration
```

### Usage

### Main Benchmark
You can benchmark any algorithm on any dataset using the CLI:
Example :
```bash
uv run scripts/benchmark.py \
  --dataset HIGGS \
  --model FalkonGPU \
  --m 12000 \
  --sigma 7.0 \
  --lam 2e-6
```
This script trains the selected solver, evaluates it using the dataset-specific
metric, prints timing and error to stdout, and saves a CSV result file to:
```php-template
outputs/<dataset>/<model>_<sigma>_<lam>_<m>.csv
```

### Benchmark over range of points
The script benchmark_logm.py performs the same benchmark but sweeps m over
a logarithmic grid for a fixed dataset and solver, allowing scaling curves to be generated.

### Synthetic Benchmarks

The following scripts reproduce the toy experiments used for Figures 4–6 of the report:
- approximation_benchmark.py — Nyström approximation accuracy vs n,m
- math_asymptotes.py — theoretical asymptotic slope plots
- time_benchmarking.py — runtime scaling of solvers

Each script saves corresponding figures to the figures/ directory.

### Notebooks
Notebooks are mostly self-contained and demonstrate usage of the implemented
solvers and datasets. Some notebooks optionally reuse CSV results generated
by the benchmark scripts.

## Data

The project uses two main datasets from the UCI repository:

- **YearPredictionMSD** (regression)
- **HIGGS** (binary classification)

Download and preprocessing scripts are located in `data/`.

### Download datasets

To download the raw datasets:

```bash
cd data
./download.sh yearpred   # only YearPredictionMSD
./download.sh higgs      # only HIGGS
./download.sh all        # both datasets

### HIGGS Data

The raw HIGGS dataset is expected as a CSV file at:
- `data/HIGGS.csv`
You can convert it to a clean parquet file used by the code with:
```bash
python data/higgs.py
```
This creates ```data/higgs.parquet```

For faster experiments you can create a smaller balanced subset of HIGGS : 
```uv data/mini_hiigs.py size```

## Ouputs
Some ready-made data are included in the outputs folder.