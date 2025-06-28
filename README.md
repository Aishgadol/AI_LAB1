# Genetic Algorithm Playground

This project contains multiple implementations of genetic algorithms for string matching, bin packing and experimental ARC problems. The code demonstrates advanced GA techniques such as probabilistic tournament selection, linear fitness scaling and aging-based survival. Results are visualized through Matplotlib and stored in the `experiment_results` folder.

## Purpose & Objectives
- Achieve exact string matching (`ga_target`) using varying crossover and mutation strategies.
- Evaluate selection schemes (RWS, SUS, deterministic and probabilistic tournaments) under linear scaling.
- Measure population diversity via Shannon entropy and Levenshtein distance while tracking runtime.
- Produce plots comparing best/mean/worst fitness, boxplots per generation and entropy trends.

## Architecture & Modules
- **sol.py** – Core GA engine with configurable parameters (see lines 10‑30)【F:sol.py†L8-L30】. The `run_ga` API exposes customization of crossover, selection and aging (lines 699‑744)【F:sol.py†L699-L744】.
- **new_sol.py** – Variant of `sol.py` retaining identical options for experimentation.
- **mixed_sol_engine.py** / **sol_with_binpacking.py** – Extended engines supporting both string and bin‑packing problems.
- **runner5.py** and **ga_helloworld_base.py** – Minimal GA examples showcasing basic crossover and fitness modes (lines 1‑15)【F:runner5.py†L1-L15】.
- **experiment_runner.py** – Automates parameter sweeps using `sol.run_ga` (lines 8‑33)【F:experiment_runner.py†L8-L33】 and saves comparison graphs under `experiment_results/`.
- **plotter.py** – Generates a 3D surface of gene distances after running the GA (lines 1‑40)【F:plotter.py†L1-L40】.
- **ai_lab1_eden_sol/** – Legacy solutions for coursework including executables and scripts for bin packing and ARC tasks. The accompanying `requirments.txt` describes basic dependencies.
- **experiment_results/** – Precomputed PNG plots illustrating fitness evolution under different settings.

### Data Flow
1. **Ingestion** – `init_population()` randomly initializes genes (string or permutation form).  
2. **Preprocessing** – `calc_fitness()` computes ASCII, LCS or combined fitness; bin‑packing mode evaluates wasted space.  
3. **Core Logic** – `mate()` selects parents via configurable strategy, applies crossover and mutation, and optionally replaces aged individuals.  
4. **Output** – After each generation statistics are collected via `compute_fitness_statistics()` (lines 445‑480)【F:sol.py†L445-L480】 and optionally plotted with `plot_fitness_evolution` and related functions.

## Installation & Environment Setup
1. Install Python ≥3.8.  
2. Install package requirements:
```bash
pip install -r requirements.txt
```
3. No GPU is required. Ensure system packages for matplotlib (e.g., `libfreetype6`, `libpng`) are present on Linux.

## Usage Examples
### Run default GA on a string target
```bash
python sol.py
```
This executes the main loop until the target string is reached or iteration/time limits are hit, printing generation statistics.

### Programmatic invocation with custom parameters
```python
from sol import run_ga
results = run_ga(crossover_method="two_point", fitness_mode="combined",
                 mutation_rate=0.55, population_size=500)
print(results["converged_generation"], results["termination_reason"])
```
`results` contains fitness histories and diversity metrics as returned by `run_ga` (see function definition). Command‑line scripts in `ai_lab1_eden_sol` accept arguments for heuristic type, mutation, crossover and selection.

## Outputs & Artifacts
- Fitness logs and diversity metrics are printed to the console.  
- PNG plots are saved to `experiment_results/` when running `experiment_runner.py`.  
- Additional figures may be produced by `plotter.py` in the current directory.

## Development & Contribution Workflow
- Run tests with `pytest -q` (currently none are defined).  
- Follow PEP8 formatting and submit pull requests with concise commits.  
- Use `git status` to ensure a clean working tree before committing.

## Project Status & Roadmap
This codebase is **alpha** quality. Future plans include full ARC integration, improved bin‑packing heuristics and unit tests.

## License & Attribution
No license file is provided. It is recommended to release the code under the MIT License for maximum reuse.


Automated PR #1

Automated PR #2

Automated PR #3