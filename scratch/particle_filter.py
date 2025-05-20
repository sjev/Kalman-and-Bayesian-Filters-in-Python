from typing import Callable, List, Optional, Tuple, Union
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
    """Create uniformly distributed particles within specified ranges."""
    particles = np.empty((n_particles, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=n_particles)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=n_particles)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=n_particles)
    particles[:, 2] %= 2 * np.pi
    return particles


def create_gaussian_particles(
    mean: ArrayLike, std: ArrayLike, n_particles: int
) -> np.ndarray:
    """Create normally distributed particles around a mean position."""
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
    """Move particles according to control input u with noise."""
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
    """Update particle weights based on measurement likelihood."""
    new_weights = weights.copy()

    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
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
    """Compute the weighted mean and variance of the particle positions."""
    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean) ** 2, weights=weights, axis=0)
    return mean, var


def calculate_neff(weights: np.ndarray) -> float:
    """Calculate the effective number of particles."""
    return 1.0 / np.sum(np.square(weights))


def resample_particles(
    particles: np.ndarray, indexes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample particles based on provided indexes."""
    new_particles = particles[indexes].copy()
    n_particles = len(new_particles)
    new_weights = np.ones(n_particles) / n_particles

    return new_particles, new_weights


def run_particle_filter(
    n_particles: int,
    iters: int = 18,
    sensor_std_err: float = 0.1,
    xlim: Tuple[float, float] = (0, 20),
    ylim: Tuple[float, float] = (0, 20),
    initial_x: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Run particle filter simulation with console output for statistics."""
    # Define landmarks (known locations)
    landmarks = np.array([[-1, 2], [5, 10], [12, 14], [18, 21]])
    n_landmarks = len(landmarks)

    # Create particles and weights
    if initial_x is not None:
        particles = create_gaussian_particles(
            mean=initial_x, std=(5, 5, np.pi / 4), n_particles=n_particles
        )
    else:
        particles = create_uniform_particles((0, 20), (0, 20), (0, 6.28), n_particles)

    weights = np.ones(n_particles) / n_particles

    # Storage for results
    xs = []
    robot_pos = np.array([0.0, 0.0])

    print("Iteration | True Position | Estimated Position | Error (Distance)")
    print("-" * 65)

    for i in range(iters):
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

        # Calculate error (distance between true and estimated position)
        error = np.linalg.norm(robot_pos - mu)

        # Print statistics
        print(
            f"{i:9} | ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}) | ({mu[0]:.2f}, {mu[1]:.2f}) | {error:.4f}"
        )

    # Convert position history to numpy array
    xs = np.array(xs)

    # Final statistics
    final_mu, final_var = estimate(particles, weights)
    final_error = np.linalg.norm(robot_pos - final_mu)
    print("\nFinal statistics:")
    print(f"True position: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})")
    print(f"Estimated position: ({final_mu[0]:.2f}, {final_mu[1]:.2f})")
    print(f"Error: {final_error:.4f}")
    print(f"Variance: {final_var}")

    return xs, (final_mu, final_var)


if __name__ == "__main__":
    from numpy.random import seed

    seed(2)
    run_particle_filter(n_particles=5000)
