from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from filterpy.monte_carlo import systematic_resample
from numpy.linalg import norm
from numpy.random import randn, uniform

ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...]]


def create_uniform_particles(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    hdg_range: Tuple[float, float],
    n_particles: int,
) -> np.ndarray:
    """
    Create uniformly distributed particles within the specified ranges.

    Parameters
    ----------
    x_range : Tuple[float, float]
        Range for x position (min, max)
    y_range : Tuple[float, float]
        Range for y position (min, max)
    hdg_range : Tuple[float, float]
        Range for heading angle in radians (min, max)
    n_particles : int
        Number of particles to create

    Returns
    -------
    np.ndarray
        Array of shape (n_particles, 3) containing particles with columns [x, y, heading]
    """
    particles = np.empty((n_particles, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=n_particles)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=n_particles)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=n_particles)
    particles[:, 2] %= 2 * np.pi
    return particles


def create_gaussian_particles(
    mean: ArrayLike, std: ArrayLike, n_particles: int
) -> np.ndarray:
    """
    Create normally distributed particles around a mean position.

    Parameters
    ----------
    mean : ArrayLike
        Mean position [x, y, heading]
    std : ArrayLike
        Standard deviations [std_x, std_y, std_heading]
    n_particles : int
        Number of particles to create

    Returns
    -------
    np.ndarray
        Array of shape (n_particles, 3) containing particles with columns [x, y, heading]
    """
    particles = np.empty((n_particles, 3))
    particles[:, 0] = mean[0] + (randn(n_particles) * std[0])
    particles[:, 1] = mean[1] + (randn(n_particles) * std[1])
    particles[:, 2] = mean[2] + (randn(n_particles) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles


def predict(
    particles: np.ndarray,
    u: Tuple[float, float],
    std: Tuple[float, float],
    dt: float = 1.0,
) -> np.ndarray:
    """
    Move particles according to control input u with noise.

    Parameters
    ----------
    particles : np.ndarray
        Array of shape (n_particles, 3) containing particles with columns [x, y, heading]
    u : Tuple[float, float]
        Control input as (heading change, velocity)
    std : Tuple[float, float]
        Standard deviations as (std for heading change, std for velocity)
    dt : float, optional
        Time step size, by default 1.0

    Returns
    -------
    np.ndarray
        Updated particles after applying the motion model
    """
    # Create a copy to avoid modifying the input
    new_particles = particles.copy()
    n_particles = len(particles)

    # Update heading
    new_particles[:, 2] += u[0] + (randn(n_particles) * std[0])
    new_particles[:, 2] %= 2 * np.pi

    # Move in the (noisy) commanded direction
    dist = (u[1] * dt) + (randn(n_particles) * std[1])
    new_particles[:, 0] += np.cos(new_particles[:, 2]) * dist
    new_particles[:, 1] += np.sin(new_particles[:, 2]) * dist

    return new_particles


def update_weights(
    particles: np.ndarray,
    weights: np.ndarray,
    z: np.ndarray,
    R: float,
    landmarks: np.ndarray,
) -> np.ndarray:
    """
    Update particle weights based on measurement likelihood.

    Parameters
    ----------
    particles : np.ndarray
        Array of shape (n_particles, 3) containing particles with columns [x, y, heading]
    weights : np.ndarray
        Current weights for each particle
    z : np.ndarray
        Measurement values (distances to landmarks)
    R : float
        Measurement noise standard deviation
    landmarks : np.ndarray
        Array of landmark positions, shape (n_landmarks, 2)

    Returns
    -------
    np.ndarray
        Updated weights normalized to sum to 1
    """
    # Create a copy to avoid modifying the input
    new_weights = weights.copy()

    # Update weights based on measurement likelihood
    for i, landmark in enumerate(landmarks):
        # Calculate distance from each particle to the landmark
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        # Update weights based on likelihood of the measurement
        new_weights *= scipy.stats.norm(distance, R).pdf(z[i])

    # Handle potential numerical issues
    new_weights += 1.0e-300  # Avoid round-off to zero

    # Normalize weights
    if np.sum(new_weights) > 0:
        new_weights /= np.sum(new_weights)

    return new_weights


def estimate(
    particles: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the weighted mean and variance of the particle positions.

    Parameters
    ----------
    particles : np.ndarray
        Array of shape (n_particles, 3) containing particles with columns [x, y, heading]
    weights : np.ndarray
        Weights for each particle (normalized to sum to 1)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing (mean, variance) of particle positions
    """
    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean) ** 2, weights=weights, axis=0)
    return mean, var


def calculate_neff(weights: np.ndarray) -> float:
    """
    Calculate the effective number of particles.

    Parameters
    ----------
    weights : np.ndarray
        Weights for each particle (normalized to sum to 1)

    Returns
    -------
    float
        Effective number of particles
    """
    return 1.0 / np.sum(np.square(weights))


def resample_particles(
    particles: np.ndarray, indexes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample particles based on provided indexes.

    Parameters
    ----------
    particles : np.ndarray
        Array of shape (n_particles, 3) containing particles with columns [x, y, heading]
    indexes : np.ndarray
        Indexes of particles to keep from resampling

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing (new_particles, new_weights)
    """
    new_particles = particles[indexes].copy()
    n_particles = len(new_particles)
    new_weights = np.ones(n_particles) / n_particles

    return new_particles, new_weights


def run_particle_filter(
    n_particles: int,
    iters: int = 18,
    sensor_std_err: float = 0.1,
    do_plot: bool = True,
    plot_particles: bool = False,
    xlim: Tuple[float, float] = (0, 20),
    ylim: Tuple[float, float] = (0, 20),
    initial_x: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Run the complete particle filter simulation.

    Parameters
    ----------
    n_particles : int
        Number of particles to use
    iters : int, optional
        Number of iterations to run, by default 18
    sensor_std_err : float, optional
        Standard deviation of sensor error, by default 0.1
    do_plot : bool, optional
        Whether to create plots, by default True
    plot_particles : bool, optional
        Whether to plot particles, by default False
    xlim : Tuple[float, float], optional
        x-axis limits for plot, by default (0, 20)
    ylim : Tuple[float, float], optional
        y-axis limits for plot, by default (0, 20)
    initial_x : Optional[Tuple[float, float, float]], optional
        Initial position estimate [x, y, heading], by default None

    Returns
    -------
    Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        Tuple containing (final state estimate positions, (mean, variance))
    """
    # Define landmarks (known locations)
    landmarks = np.array([[-1, 2], [5, 10], [12, 14], [18, 21]])
    n_landmarks = len(landmarks)

    if do_plot:
        plt.figure()

    # Create particles and weights
    if initial_x is not None:
        particles = create_gaussian_particles(
            mean=initial_x, std=(5, 5, np.pi / 4), n_particles=n_particles
        )
    else:
        particles = create_uniform_particles((0, 20), (0, 20), (0, 6.28), n_particles)

    weights = np.ones(n_particles) / n_particles

    if plot_particles and do_plot:
        alpha = 0.20
        if n_particles > 5000:
            alpha *= np.sqrt(5000) / np.sqrt(n_particles)
        plt.scatter(particles[:, 0], particles[:, 1], alpha=alpha, color="g")

    # Storage for results
    xs = []
    robot_pos = np.array([0.0, 0.0])

    for _ in range(iters):
        # Move robot
        robot_pos += (1, 1)

        # Distance from robot to each landmark
        zs = norm(landmarks - robot_pos, axis=1) + (randn(n_landmarks) * sensor_std_err)

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

        # Plot current state
        if do_plot:
            if plot_particles:
                plt.scatter(
                    particles[:, 0], particles[:, 1], color="k", marker=",", s=1
                )
            p1 = plt.scatter(
                robot_pos[0], robot_pos[1], marker="+", color="k", s=180, lw=3
            )
            p2 = plt.scatter(mu[0], mu[1], marker="s", color="r")

    # Convert position history to numpy array
    xs = np.array(xs)

    # Final plotting
    if do_plot:
        plt.legend([p1, p2], ["Actual", "PF"], loc=4, numpoints=1)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        print("Final position error, variance:\n\t", mu - np.array([iters, iters]), var)
        plt.show()

    return xs, (mu, var)


if __name__ == "__main__":
    from numpy.random import seed

    seed(2)
    run_particle_filter(n_particles=5000, plot_particles=True)
