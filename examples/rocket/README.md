# Rocket FastSLS gradient-window ablation

This folder contains a controlled two-run minimum-time experiment. Both runs
use FastSLS and the same rocket dynamics, cost, constraints, initialization, and
altitude-dependent disturbance profile. The only changed solver setting is:

| Experiment | `SLSConfig.gradient_window` |
| --- | ---: |
| `no_gradient` | 0 |
| `gradient_aware` | 10 |

Run both experiments from the `gpu_sls` repository root:

```bash
python examples/rocket/run_gradient_window_experiments.py
```

The launcher explicitly imports `gpu_sls` from this worktree's `src`
directory, runs each case in a fresh Python process, and writes:

- `rocket_gradient_window_experiments/no_gradient/`: window-0 NPZ, plot, and log
- `rocket_gradient_window_experiments/gradient_aware/`: window-10 NPZ, plot, and log
- `rocket_gradient_window_experiments/summary.csv`: timing, disturbance, and feasibility metrics
- `rocket_gradient_window_experiments/gradient_window_comparison.png`: trajectory/tube overlay on the disturbance field and a minimum-time bar chart

The launcher rejects a comparison if either trajectory exceeds the configured
dynamics or constraint tolerances. It also rejects the intended demonstration
if window 10 does not both spend more of its path below the disturbance
transition and achieve a lower minimum time.

To rebuild the summary and plot from existing NPZ files:

```bash
python examples/rocket/run_gradient_window_experiments.py --skip-runs
```

An individual run can be generated with:

```bash
PYTHONPATH=src python examples/rocket/rocket.py \
  --gradient-window 0 \
  --output-dir /tmp/rocket_window_0
```
