# QNN + Zero-Noise Extrapolation (PennyLane)

A compact **2-qubit quantum neural network (QNN)** demonstration of **zero-noise extrapolation (ZNE)** with odd-integer **local gate folding**: **1×, 3×, 5×, 7×, 9×**. It compares standard depolarizing noise with simplified, calibration-inspired profiles from historical IBM fake-backend snapshots.

The script plots noisy expectation values, linear/quadratic polynomial fits, and the ideal expectation value. It saves both PNG/PDF plots and CSV results.

> **Scope:** This is a fixed-parameter variational quantum circuit (QNN inference example), **not a trained classifier**. The output is the expectation value of Pauli-Z on qubit 0, **not classification accuracy**. IBM profiles are simplified approximations, **not full backend-level noise simulations** or live hardware calibrations.

## Installation and virtual environment

Requires **Python 3.10** with **PennyLane 0.42.3** (also compatible with newer Python versions supported by this release). Use a local virtual environment so PennyLane and Qiskit dependencies do not affect other Python projects.

**1. Get the repository and enter its directory**

```bash
git clone https://github.com/Ratun11/qnn-zne-pennylane.git
cd qnn-zne-pennylane
```


**2. Create a virtual environment**

On macOS / Linux:

```bash
python3 -m venv .venv
```

On Windows:

```powershell
py -m venv .venv
```

**3. Activate the virtual environment**

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt (`cmd.exe`):

```bat
.venv\Scripts\activate.bat
```

Your terminal should now show `(.venv)` at the beginning of its prompt.

**4. Install required packages**

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**5. Verify and run**

```bash
python -c "import pennylane as qml; print('PennyLane', qml.__version__)"
python qnn_zne_pennylane.py
```

To run only the standard depolarizing-noise example:

```bash
python qnn_zne_pennylane.py --models standard
```

**6. Deactivate when finished**

```bash
deactivate
```

To work on the project again, return to its directory and repeat **step 3** to reactivate the existing `.venv`. You do not need to recreate it or reinstall packages every time.

**Troubleshooting:** On Ubuntu/Debian, if creating the environment fails due to a missing `venv` module, install `python3-venv` with your package manager. If PowerShell blocks script activation, you can run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that PowerShell session and activate again, or use Command Prompt. `.venv/` is excluded by `.gitignore` and must **not** be committed to GitHub.

## Run options

```bash
# Fastest example; no IBM backend required
python qnn_zne_pennylane.py --models standard

# Compare two IBM fake backend snapshots
python qnn_zne_pennylane.py --models Brisbane Torino

# Change odd scale factors (at least 3 required)
python qnn_zne_pennylane.py --models standard --factors 1 3 5 7 9 11

# Finite sampling and repetitions (shows ±1 sample SD error bars)
python qnn_zne_pennylane.py --models standard --shots 4096 --repeats 20

# Use a different output folder
python qnn_zne_pennylane.py --models standard --out my_results

# All available flags
python qnn_zne_pennylane.py --help
```

**Defaults:** observable `p1` (probability of reading qubit 0 as `1`); models `standard Brisbane Sherbrooke Torino`; odd factors `1 3 5 7 9`; analytic expectation values (`shots=None`); one repetition. The first run may take longer because multiple density-matrix circuits are evaluated.

## How the noise amplification works

For an original gate `U`, local unitary folding replaces it with:

| Factor | Ideal gate sequence |
|---|---|
| 1× | `U` |
| 3× | `U U† U` |
| 5× | `U U† U U† U` |
| 7× | `U U† U U† U U† U` |
| 9× | `U U† U U† U U† U U† U` |

In general **factor = 2n + 1** for `n=0,1,...`. Gates cancel ideally, leaving the original unitary, but **noise is inserted after every physical gate** using PennyLane `default.mixed`. CNOT is its own inverse; inverse RY simply negates its rotation angle. All input-encoding gates and variational gates are folded.

This is **local gate folding** (not global circuit folding, and not a direct multiplication of error probabilities). Readout noise is applied **only once**, so it is not scaled with the gate count. Therefore extrapolation to factor 0 is not guaranteed to remove measurement noise: factor 0 is a model-based extrapolation, not a physical circuit execution.

## Noise models

| Option | Source | Implementation |
|---|---|---|
| `standard` | Manually chosen `p1=0.004`, `p2=0.020`, `pm=0.015` | Depolarizing channels after RY and CNOT proxy channels; one readout flip |
| `Brisbane` | `FakeBrisbane` snapshot | Median single- and two-qubit gate error and readout error |
| `Sherbrooke` | `FakeSherbrooke` snapshot | Same summarization |
| `Torino` | `FakeTorino` snapshot | Same summarization |

For the IBM-inspired options, the script extracts median `sx` (or alternative) error, median `ecr/cz/cx` error, and median measurement error from `backend.target`. The extracted values depend on the installed `qiskit-ibm-runtime` release. The single-qubit rate is used in `qml.DepolarizingChannel` after each RY. **A noisy CNOT is approximated by one single-qubit depolarizing channel on each participating wire**, not an exact calibrated two-qubit noise channel. The readout error is approximated with `qml.BitFlip` before measuring Z on wire 0.

**Important limitations:** IBM's hardware uses specific native gates and connected qubit pairs; median rates discard this granularity. The reported backend error rates are treated as channel parameters for illustration, even though calibration error metrics and depolarizing probabilities are not exactly identical. The simulation does not include T1/T2 relaxation, crosstalk, qubit mapping, native-gate compilation, or current/live device calibrations. Consequently **the labels do not imply accurate comparative benchmarking of IBM hardware**.

See [IBM fake-backend documentation](https://qiskit.qotlabs.org/docs/api/qiskit-ibm-runtime/fake-provider).

## Extrapolation and graphs

The code evaluates the same QNN at each fold factor and fits, using **all available factor values**:

- **Linear ZNE:** `numpy.polyfit(factors, expectations, 1)`
- **Quadratic ZNE:** `numpy.polyfit(factors, expectations, 2)`

The intercept at `factor=0` is the ZNE estimate. A separate noiseless run provides an ideal reference. The figure contains **two columns per noise profile**: linear fit (left), quadratic fit (right). Blue circles are noisy results, red dashed curves are fits, red stars are extrapolated intercepts, and the black dotted horizontal line is the ideal value. The x-axis is the physical gate-folding scale **λ = 2n+1**, not the error probability itself.

Quadratic ZNE is **not always better** than linear ZNE. High factors can be hard to fit, especially with shot noise and unscaled measurement noise. With `--shots` and `--repeats`, error bars are the **sample standard deviation between repeated runs**, not confidence intervals for the fitted intercept.


### Choosing the plotted observable and interpreting the slope

The default observable is **P(qubit 0 = 1)**, computed from `P(1) = (1 - <Z0>)/2`. For the fixed parameters in this demonstration, the ideal `<Z0>` is positive (~0.505), and depolarizing noise generally makes `<Z0>` **decrease** toward zero. Therefore `P(1)` generally **increases** toward 0.5 as folding scale λ increases. This is a legitimate measurement probability, not a flipped plot axis. The slope is circuit/observable-dependent; ZNE does **not** require an increasing curve. The original expectation-value plot can be recovered via:

```bash
python qnn_zne_pennylane.py --models standard --observable z
```

To show the probability (default) with the standard noise profile:

```bash
python qnn_zne_pennylane.py --models standard --observable p1 --lambdas 1 3 5 7 9
```

The ideal point at λ=0 is not a physical circuit. Only λ>=1 odd integers are executed; polynomial extrapolation estimates the λ→0 intercept. Readout noise is not amplified and can bias the extrapolated estimate.

## Outputs

Files are written under `results/` by default:

```text
results/
├── zne_comparison.png   # Plot suitable for quick viewing
├── zne_comparison.pdf   # Vector figure for papers
├── zne_results.csv      # Per-factor, per-repeat expectations and rates
└── zne_summary.csv      # Ideal/raw/ZNE estimates, errors and rates
```

The `results/` directory is ignored by Git to keep experimental outputs out of version control. Remove `results/` from `.gitignore` if you want to commit selected figures/results.

## Project structure

```text
qnn-zne-pennylane/
├── qnn_zne_pennylane.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Publish to GitHub

1. Create an **empty** GitHub repository called `qnn-zne-pennylane` (don't initialize with another README).
2. Open a terminal in this project directory.
3. Run (substitute your actual GitHub username):

```bash
git init
git add .
git commit -m "Add PennyLane QNN ZNE experiments with odd gate folding"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/qnn-zne-pennylane.git
git push -u origin main
```

GitHub may prompt you to authenticate with a browser, credential manager, or personal access token; GitHub account passwords are not accepted as HTTPS Git credentials.

## References

- [PennyLane: `fold_global` and ZNE circuit folding](https://docs.pennylane.ai/en/stable/code/api/pennylane.fold_global.html)
- [PennyLane: `mitigate_with_zne`](https://docs.pennylane.ai/en/stable/code/api/pennylane.noise.mitigate_with_zne.html)
- [IBM Quantum: Fake backends](https://qiskit.qotlabs.org/docs/api/qiskit-ibm-runtime/fake-provider)

## Reproducibility note

This project uses fixed input features and fixed variational weights. Its default expectation values are deterministic in analytic mode. With finite shots, measurements fluctuate; no random seed is fixed for that optional mode. Record your library versions (`pip freeze`) and IBM fake-backend calibration snapshot if reporting hardware-inspired comparisons in a paper.
