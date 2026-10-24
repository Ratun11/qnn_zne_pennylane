"""Two-qubit fixed-parameter QNN with odd-factor local gate folding ZNE.

IBM presets use median calibration error rates from historical fake backend
snapshots as simplified depolarizing probabilities, NOT complete device noise.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

INPUT = (0.74, -0.31)
PARAMS = ((0.6, -0.4), (0.5, 0.9))
STANDARD = {"p1": 0.004, "p2": 0.020, "pm": 0.015}


def median_error(target, operation):
    if operation not in target.operation_names:
        return None
    values = [float(p.error) for p in target[operation].values()
              if p is not None and p.error is not None]
    return float(np.median(values)) if values else None


def ibm_profile(name):
    from qiskit_ibm_runtime import fake_provider
    cls = getattr(fake_provider, "Fake" + name, None)
    if cls is None:
        raise ValueError(f"Fake{name} unavailable in installed qiskit-ibm-runtime")
    target = cls().target
    p1 = next((v for op in ("sx", "x", "u3")
               if (v := median_error(target, op)) is not None), None)
    p2 = next((v for op in ("ecr", "cz", "cx")
               if (v := median_error(target, op)) is not None), None)
    pm = median_error(target, "measure")
    if None in (p1, p2, pm):
        raise ValueError(f"Fake{name} lacks required calibration error data")
    return {"p1": p1, "p2": p2, "pm": pm}


def qnn_expectation(profile, factor, shots=None, observable="p1"):
    """Odd-factor local folding: U (U† U)^n, with noise after each physical gate.

    We fold each unitary gate independently, not the readout process.
    Single-qubit depolarization follows every RY; approximate 2-qubit
    depolarization is a channel on each CNOT wire (a proxy, not exact).
    """
    n = (factor - 1) // 2
    dev = qml.device("default.mixed", wires=2, shots=shots)

    def ry(theta, wire):
        for k in range(2 * n + 1):
            qml.RY(theta if k % 2 == 0 else -theta, wires=wire)
            if profile["p1"] > 0:
                qml.DepolarizingChannel(profile["p1"], wires=wire)

    def cx():
        # CNOT is self-inverse: repeated odd times is the original CNOT.
        for _ in range(2 * n + 1):
            qml.CNOT(wires=[0, 1])
            if profile["p2"] > 0:
                qml.DepolarizingChannel(profile["p2"], wires=0)
                qml.DepolarizingChannel(profile["p2"], wires=1)

    @qml.qnode(dev)
    def circuit():
        ry(INPUT[0], 0)
        ry(INPUT[1], 1)
        for layer in PARAMS:
            ry(layer[0], 0)
            ry(layer[1], 1)
            cx()
        # Readout is applied once and is not amplified by gate folding.
        if profile["pm"] > 0:
            qml.BitFlip(profile["pm"], wires=0)
        return qml.expval(qml.PauliZ(0))

    z = float(circuit())
    return z if observable == "z" else (1.0 - z) / 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["standard", "Brisbane", "Sherbrooke", "Torino"])
    ap.add_argument("--factors", "--lambdas", type=int, nargs="+", default=[1, 3, 5, 7, 9],
                    help="Odd gate-folding scale values lambda = 2n+1")
    ap.add_argument("--observable", choices=["p1", "z"], default="p1",
                    help="p1: P(qubit 0 = 1), usually rising here; z: <Z0>, usually falling here")
    ap.add_argument("--shots", type=int, default=None,
                    help="Optional finite measurement shots; default: analytic expectations")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Repeats per factor, averaged before fitting")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    factors = np.asarray(args.factors, dtype=int)
    if len(set(factors)) != len(factors) or any(v < 1 or v % 2 != 1 for v in factors):
        ap.error("Factors must be distinct, positive odd integers, e.g. 1 3 5 7 9")
    if len(factors) < 3:
        ap.error("At least three noise factors are needed for quadratic fitting")
    if args.repeats < 1 or (args.shots is not None and args.shots < 1):
        ap.error("--repeats and --shots must be positive")
    factors.sort()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    profiles = {}
    for name in args.models:
        if name in profiles:
            ap.error(f"Repeated model {name}")
        try:
            profiles[name] = STANDARD.copy() if name.lower() == "standard" else ibm_profile(name)
        except (ImportError, ValueError) as exc:
            ap.error(str(exc))

    ideal = qnn_expectation({"p1": 0.0, "p2": 0.0, "pm": 0.0}, 1, observable=args.observable)
    fig, axes = plt.subplots(len(profiles), 2, figsize=(11, 3.5 * len(profiles)), squeeze=False)
    data_rows = []
    summary_rows = []
    grid = np.linspace(0, max(factors) + .3, 300)

    for row_idx, (name, profile) in enumerate(profiles.items()):
        values = np.asarray([[qnn_expectation(profile, int(f), args.shots, args.observable)
                              for _ in range(args.repeats)] for f in factors])
        means = values.mean(axis=1)
        stds = values.std(axis=1, ddof=1) if args.repeats > 1 else np.zeros(len(factors))
        # All measured factors are used for either fit; change here for subset fitting.
        fits = [("Linear ZNE", np.polyfit(factors, means, 1)),
                ("Quadratic ZNE", np.polyfit(factors, means, 2))]
        lin_est, quad_est = [float(np.polyval(coeff, 0)) for _, coeff in fits]
        summary_rows.append({"model": name, "observable": args.observable, "ideal": ideal, "unmitigated_1x": float(means[np.where(factors == 1)[0][0]]) if 1 in factors else "",
                             "linear_zne": lin_est, "quadratic_zne": quad_est,
                             "linear_abs_error": abs(lin_est - ideal), "quadratic_abs_error": abs(quad_est - ideal), **profile})
        print(f"{name}: ideal={ideal:.6f} linear={lin_est:.6f} quadratic={quad_est:.6f} rates={profile}")

        for f, rep_vals, mean, std in zip(factors, values, means, stds):
            for run_idx, val in enumerate(rep_vals, 1):
                data_rows.append({"model": name, "observable": args.observable, "lambda": int(f), "repeat": run_idx,
                                  "expectation": float(val), "mean": float(mean), "std": float(std), **profile})
        for col, (title, coeff) in enumerate(fits):
            ax = axes[row_idx, col]
            ax.errorbar(factors, means, yerr=stds if args.repeats > 1 else None,
                        fmt="o", color="tab:blue", capsize=3, label="Noisy QNN")
            ax.plot(grid, np.polyval(coeff, grid), "--", color="tab:red", label="Polynomial fit")
            ax.axhline(ideal, ls=":", color="black", label="Ideal QNN")
            ax.scatter(0, np.polyval(coeff, 0), color="tab:red", marker="*", s=100,
                       label="ZNE at λ = 0", zorder=5)
            ax.set(title=f"{name} — {title}", xlabel="Gate-folding scale λ (λ = 2n + 1)",
                   ylabel="P(qubit 0 = 1)" if args.observable == "p1" else "Expectation ⟨Z₀⟩", xlim=(-0.3, max(factors) + .4))
            ax.set_xticks([0, *factors])
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "zne_comparison.png", dpi=220)
    fig.savefig(out / "zne_comparison.pdf")
    plt.close(fig)
    for filename, rows in (("zne_results.csv", data_rows), ("zne_summary.csv", summary_rows)):
        with (out / filename).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved outputs to {out.resolve()}")


if __name__ == "__main__":
    main()
