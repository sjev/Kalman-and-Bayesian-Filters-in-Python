import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple

from particle_filter import (
    create_gaussian_particles,
    create_uniform_particles,
    predict,
    update_weights,
    calculate_neff,
    resample_particles,
    estimate,
)
from filterpy.monte_carlo import systematic_resample


def run_pf_with_plotting(
    n_particles: int,
    iters: int = 18,
    sensor_std_err: float = 0.1,
    plot_particles: bool = False,
    xlim: Tuple[float, float] = (0, 20),
    ylim: Tuple[float, float] = (0, 20),
    initial_x: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Run particle filter with visual plotting of results."""
    # Define landmarks (known locations)
    landmarks = np.array([[-1, 2], [5, 10], [12, 14], [18, 21]])
    n_landmarks = len(landmarks)

    plt.figure()

    # Create particles and weights
    if initial_x is not None:
        particles = create_gaussian_particles(
            mean=initial_x, std=(5, 5, np.pi / 4), n_particles=n_particles
        )
    else:
        particles = create_uniform_particles((0, 20), (0, 20), (0, 6.28), n_particles)

    weights = np.ones(n_particles) / n_particles

    if plot_particles:
        alpha = 0.20
        if n_particles > 5000:
            alpha *= np.sqrt(5000) / np.sqrt(n_particles)
        plt.scatter(particles[:, 0], particles[:, 1], alpha=alpha, color="g")

    # Storage for results
    xs = []
    robot_pos = np.array([0.0, 0.0])

    print("Iteration | True Position | Estimated Position | Error (Distance)")
    print("-" * 65)

    for i in range(iters):
        # Move robot
        robot_pos += (1, 1)

        # Distance from robot to each landmark
        zs = np.linalg.norm(landmarks - robot_pos, axis=1) + (
            np.random.randn(n_landmarks) * sensor_std_err
        )

        # Move particles according to motion model
        particles = predict(particles, u=(0.00, 1.414), std=(0.2, 0.05))

        # Update particle weights
        weights = update_weights(
            particles, weights, z=zs, R=sensor_std_err, landmarks=landmarks
        )

        # Resample if too few effective particles
        if calculate_neff(weights) < n_particles / 2:
            indexes = systematic_resample(weights)
            particles, weights = resample_particles(particles, indexes)
            assert np.allclose(weights, 1 / n_particles)

        # Calculate state estimate
        mu, var = estimate(particles, weights)
        xs.append(mu)

        # Calculate error (distance between true and estimated position)
        error = np.linalg.norm(robot_pos - mu)

        # Print statistics
        print(
            f"{i:9} | ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}) | ({mu[0]:.2f}, {mu[1]:.2f}) | {error:.4f}"
        )

        # Plot current state
        if plot_particles:
            plt.scatter(particles[:, 0], particles[:, 1], color="k", marker=",", s=1)
        p1 = plt.scatter(robot_pos[0], robot_pos[1], marker="+", color="k", s=180, lw=3)
        p2 = plt.scatter(mu[0], mu[1], marker="s", color="r")

    # Convert position history to numpy array
    xs = np.array(xs)

    # Final plotting
    plt.legend([p1, p2], ["Actual", "PF"], loc=4, numpoints=1)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    print("Final position error, variance:\n\t", mu - np.array([iters, iters]), var)
    plt.show()

    return xs, (mu, var)


if __name__ == "__main__":
    from numpy.random import seed

    seed(2)
    run_pf_with_plotting(n_particles=5000, plot_particles=True)
