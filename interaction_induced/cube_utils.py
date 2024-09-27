import numpy as np


def prepare_grid(
    geometry: np.ndarray, grid_step: float | tuple, grid_overage: float | tuple
) -> dict:
    """
    Prepares a simple scalar grid in 3D space.
    """

    if len(geometry.T) != 3:
        raise ValueError(
            f"`geometry` shloud have shape (N_atom, 3) was {geometry.shape}!"
        )

    # figure out grid sizes
    grid_min = np.zeros(3)
    grid_max = np.zeros(3)

    if isinstance(grid_overage, float):

        for i, axis in enumerate(geometry.T):
            grid_min[i] = np.min(axis) - grid_overage
            grid_max[i] = np.max(axis) + grid_overage

    elif isinstance(grid_overage, tuple):

        if len(grid_overage) != 3:
            raise ValueError(
                "`grid_overage` has to be float or tuple of length 3, "
                f"was length {len(grid_overage)}!"
            )

        for i, axis in enumerate(geometry.T):
            grid_min[i] = np.min(axis) - grid_overage[i]
            grid_max[i] = np.max(axis) + grid_overage[i]

    else:
        raise TypeError(
            f"`grid_overage' has to be float or tuple but was {type(grid_overage)}!"
        )

    # figure out grid points
    grid = {}

    if isinstance(grid_step, float):

        for i, axis in enumerate(["x", "y", "z"]):
            grid[axis] = np.arange(grid_min[i], grid_max[i] + grid_step, grid_step)
            grid[f"step_{axis}"] = grid_step

    elif isinstance(grid_step, tuple):

        if len(grid_step) != 3:
            raise ValueError(
                f"`grid_step` has to be float or tuple of length 3, was length {len(grid_step)}!"
            )

        for i, axis in enumerate(["x", "y", "z"]):
            grid[axis] = np.arange(
                grid_min[i], grid_max[i] + grid_step[i], grid_step[i]
            )
            grid[f"step_{axis}"] = grid_step[i]

    else:
        raise TypeError(
            f"`grid_step' has to be float or tuple but was {type(grid_step)}!"
        )

    grid["n_x"] = grid["x"].size
    grid["n_y"] = grid["y"].size
    grid["n_z"] = grid["z"].size

    return grid


def calculate_isocontour(values: np.ndarray, threshold: float = 0.85):
    """
    Calculate isocontour values for a given `threshlod`,
    assumed as the density fraction.
    """

    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError(
            f"`threshold` must be within range (0.0, 1.0), was {threshold:.2f}!"
        )

    flatened_vals = values.flatten()
    positive_target = threshold * np.sum(
        0.5 * (np.sign(flatened_vals) + 1) * np.abs(flatened_vals)
    )
    negative_target = threshold * np.sum(
        0.5 * (np.sign(flatened_vals) - 1) * np.abs(flatened_vals)
    )

    positive_sum = 0.0
    negative_sum = 0.0
    positive_isoval = 0.0
    negative_isoval = 0.0
    do_positive = True
    do_negative = True
    for i in np.argsort(np.abs(flatened_vals))[::-1]:

        if do_positive and flatened_vals[i] >= 0.0:
            positive_sum += flatened_vals[i]
            if positive_sum >= positive_target:
                positive_isoval = flatened_vals[i]
                do_positive = False

        elif do_negative and flatened_vals[i] < 0.0:
            negative_sum += flatened_vals[i]
            if negative_sum <= negative_target:
                negative_isoval = flatened_vals[i]
                do_negative = False

    return (positive_isoval, negative_isoval)
