from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_results(results: Dict[str, List]):
    """
    Plot the results of a particle filter simulation.

    Parameters
    ----------
    results : Dict[str, List]
        Dictionary containing simulation results
    """
    # Extract data from results
    landmarks = results["landmarks"]

    # Create plot
    plt.figure(figsize=(10, 8))

    # Plot landmarks
    plt.scatter(
        landmarks[:, 0],
        landmarks[:, 1],
        marker="^",
        color="blue",
        s=100,
        label="Landmarks",
    )
    # plot particles prediction
    for particles in results["particles_pre"]:
        plt.scatter(
            particles[:, 0],
            particles[:, 1],
            marker="o",
            color="green",
            s=2,
            alpha=0.5,
        )

    # plot particles after resampling
    # plot particles prediction
    for particles in results["particles_post"]:
        plt.scatter(
            particles[:, 0],
            particles[:, 1],
            marker="o",
            color="blue",
            s=2,
            alpha=0.5,
        )
    # plot true position
    true_positions = np.array(results["true_position"])
    plt.plot(
        true_positions[:, 0],
        true_positions[:, 1],
        marker="x",
        color="k",
        markersize=5,
        label="True Position",
    )

    # Plot estimated position
    estimated_positions = np.array(results["estimated_position"])
    plt.plot(
        estimated_positions[:, 0],
        estimated_positions[:, 1],
        marker="o",
        color="red",
        markersize=5,
        label="Estimated Position",
    )

    # Set labels and title
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Particle Filter Results")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.show()
