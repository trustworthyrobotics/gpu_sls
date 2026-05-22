import jax.numpy as jnp
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def get_trajectory_tubes(Phi_x):
    return jnp.linalg.norm(Phi_x, ord=2, axis=-1).sum(axis=1)

def plot_tube_graph(
    disturbed,
    lower,
    upper,
    dt,
    output_folder,
    filename: str = "disturbance_vs_tube_size.png",
):
    """
    Generic: Plot deviation vs tube size for arbitrary state dimension.

    disturbed: (n_rollouts, T, n_states)
    tube:      (T+1, n_states)
    """

    disturbed = np.asarray(disturbed)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    if disturbed.ndim != 3:
        raise ValueError(
            f"disturbed has shape {disturbed.shape}. Expected (n_rollouts, T, n_states)."
        )

    n_rollouts, T, n_states = disturbed.shape

    tube_trim_lower = lower
    tube_trim_upper = upper

    # time axis
    t = np.arange(T) * dt

    # generic labels
    state_labels = [f"state {i}" for i in range(n_states)]

    fig, axes = plt.subplots(
        n_states, 1, figsize=(10, 2 * n_states + 2), sharex=True
    )
    if n_states == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        tube_i_lower = tube_trim_lower[:, idx]
        tube_i_upper = tube_trim_upper[:, idx]
        dev_all = disturbed[:, :, idx]


        # rollouts
        for r_idx, dev in enumerate(dev_all):
            m = np.isfinite(dev)
            ax.plot(
                t[m],
                dev[m],
                alpha=0.8,
                label="|x - nominal|" if r_idx == 0 else None,
            )

        # tube
        ax.plot(t, tube_i_lower, linewidth=3, label="tube size", color="tab:blue")
        ax.plot(t, tube_i_upper, linewidth=3, label="tube size", color="tab:blue")

        ax.set_ylabel(state_labels[idx])
        ax.set_title(f"{state_labels[idx]}: Deviation vs Tube Size")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("time (s)")
    plt.tight_layout()
    output_path = os.path.join(output_folder, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)