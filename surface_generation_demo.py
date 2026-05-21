"""
HumpDay Surface Generation Demo

Demonstrates automated stochastic surface generation for optimizer testing.
Run this to see various types of generated optimization landscapes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal
import random
from typing import Tuple, Callable, List
import time


class SurfaceGenerator:
    """Automated generator for optimization test surfaces"""

    def __init__(self, seed: int = None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)

    def gaussian_random_field(self, size: Tuple[int, int],
                             correlation_length: float = 0.3,
                             roughness: float = 2.0) -> np.ndarray:
        """Generate smooth correlated landscape using Gaussian random fields"""

        # Create coordinate grid
        x = np.linspace(0, 1, size[0])
        y = np.linspace(0, 1, size[1])
        X, Y = np.meshgrid(x, y)

        # Create distance matrix
        coords = np.column_stack([X.ravel(), Y.ravel()])
        distances = squareform(pdist(coords))

        # Gaussian correlation function
        correlation_matrix = np.exp(-(distances / correlation_length) ** roughness)

        # Generate correlated random field
        L = np.linalg.cholesky(correlation_matrix + 1e-6 * np.eye(len(correlation_matrix)))
        white_noise = np.random.randn(len(correlation_matrix))
        correlated_field = L @ white_noise

        return correlated_field.reshape(size)

    def multi_modal_surface(self, size: Tuple[int, int],
                           n_peaks: int = 5,
                           peak_strength_var: float = 2.0) -> np.ndarray:
        """Generate surface with multiple peaks/valleys"""

        x = np.linspace(-2, 2, size[0])
        y = np.linspace(-2, 2, size[1])
        X, Y = np.meshgrid(x, y)

        surface = np.zeros(size)

        for i in range(n_peaks):
            # Random peak location
            cx = np.random.uniform(-1.5, 1.5)
            cy = np.random.uniform(-1.5, 1.5)

            # Random peak characteristics
            amplitude = np.random.normal(0, peak_strength_var)
            width_x = np.random.uniform(0.2, 0.8)
            width_y = np.random.uniform(0.2, 0.8)
            rotation = np.random.uniform(0, np.pi)

            # Create rotated Gaussian
            cos_rot, sin_rot = np.cos(rotation), np.sin(rotation)
            x_rot = cos_rot * (X - cx) - sin_rot * (Y - cy)
            y_rot = sin_rot * (X - cx) + cos_rot * (Y - cy)

            peak = amplitude * np.exp(-(x_rot/width_x)**2 - (y_rot/width_y)**2)
            surface += peak

        return surface

    def fractal_landscape(self, size: Tuple[int, int],
                         octaves: int = 6,
                         persistence: float = 0.5) -> np.ndarray:
        """Generate fractal landscape using Perlin-like noise"""

        def noise(x: float, y: float) -> float:
            # Simple hash-based noise
            n = int(x + y * 57)
            n = (n << 13) ^ n
            return (1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0)

        def smooth_noise(x: float, y: float) -> float:
            corners = (noise(x-1, y-1) + noise(x+1, y-1) + noise(x-1, y+1) + noise(x+1, y+1)) / 16
            sides = (noise(x-1, y) + noise(x+1, y) + noise(x, y-1) + noise(x, y+1)) / 8
            center = noise(x, y) / 4
            return corners + sides + center

        def interpolated_noise(x: float, y: float) -> float:
            int_x, int_y = int(x), int(y)
            frac_x, frac_y = x - int_x, y - int_y

            v1 = smooth_noise(int_x, int_y)
            v2 = smooth_noise(int_x + 1, int_y)
            v3 = smooth_noise(int_x, int_y + 1)
            v4 = smooth_noise(int_x + 1, int_y + 1)

            # Cosine interpolation
            ft = frac_x * np.pi
            f = (1 - np.cos(ft)) * 0.5
            i1 = v1 * (1 - f) + v2 * f

            ft = frac_y * np.pi
            f = (1 - np.cos(ft)) * 0.5
            i2 = v3 * (1 - f) + v4 * f

            ft = frac_x * np.pi
            f = (1 - np.cos(ft)) * 0.5
            return i1 * (1 - f) + i2 * f

        # Generate fractal
        surface = np.zeros(size)

        for y in range(size[1]):
            for x in range(size[0]):
                # Scale coordinates
                scaled_x = x * 8.0 / size[0]
                scaled_y = y * 8.0 / size[1]

                value = 0
                amplitude = 1.0

                for octave in range(octaves):
                    frequency = 2 ** octave
                    value += interpolated_noise(scaled_x * frequency, scaled_y * frequency) * amplitude
                    amplitude *= persistence

                surface[y, x] = value

        return surface

    def rugged_surface_with_trends(self, size: Tuple[int, int],
                                  n_ridges: int = 3,
                                  noise_level: float = 0.3) -> np.ndarray:
        """Generate surface with ridges and local ruggedness"""

        x = np.linspace(-3, 3, size[0])
        y = np.linspace(-3, 3, size[1])
        X, Y = np.meshgrid(x, y)

        # Base quadratic trend
        surface = 0.1 * (X**2 + Y**2)

        # Add ridges
        for i in range(n_ridges):
            # Ridge parameters
            angle = np.random.uniform(0, np.pi)
            offset = np.random.uniform(-1, 1)
            strength = np.random.uniform(0.5, 2.0)
            width = np.random.uniform(0.3, 1.0)

            # Ridge direction
            ridge_coord = X * np.cos(angle) + Y * np.sin(angle) + offset
            ridge = strength * np.exp(-(ridge_coord / width)**2)
            surface += ridge

        # Add local noise
        noise_field = self.gaussian_random_field(size, correlation_length=0.1, roughness=1.5)
        surface += noise_level * noise_field

        return surface

    def adversarial_surface(self, size: Tuple[int, int],
                           target_optimizer: str = 'powell') -> np.ndarray:
        """Generate surface designed to challenge specific optimizers"""

        x = np.linspace(-2, 2, size[0])
        y = np.linspace(-2, 2, size[1])
        X, Y = np.meshgrid(x, y)

        if target_optimizer.lower() == 'powell':
            # Create surface with misleading ridges
            surface = 0.5 * (X**2 + Y**2)  # Base bowl
            # Add false ridges that lead away from optimum
            surface += 2 * np.exp(-((X-1)**2 + (Y-0.5)**2) / 0.3)
            surface -= 3 * np.exp(-((X+0.2)**2 + (Y+0.2)**2) / 0.1)  # True optimum

        elif target_optimizer.lower() == 'nelder-mead':
            # Create narrow valleys that cause simplex collapse
            surface = abs(X) + abs(Y)  # Diamond shape
            # Add narrow channel
            channel_mask = np.abs(Y - 0.1 * X) < 0.05
            surface[channel_mask] *= 0.1

        else:  # Generic challenging surface
            surface = self.multi_modal_surface(size, n_peaks=10, peak_strength_var=3.0)

        return surface


def plot_surfaces_grid(generator: SurfaceGenerator, save_plots: bool = False):
    """Generate and plot various surface types"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Automated Stochastic Surface Generation for Optimizer Testing',
                 fontsize=16, fontweight='bold')

    surfaces = [
        ("Gaussian Random Field\n(Smooth, Correlated)",
         generator.gaussian_random_field((50, 50), correlation_length=0.4, roughness=2.0)),

        ("Multi-Modal Surface\n(Multiple Peaks/Valleys)",
         generator.multi_modal_surface((50, 50), n_peaks=6, peak_strength_var=1.5)),

        ("Fractal Landscape\n(Self-Similar, Multi-Scale)",
         generator.fractal_landscape((50, 50), octaves=5, persistence=0.6)),

        ("Rugged Surface with Trends\n(Ridges + Local Noise)",
         generator.rugged_surface_with_trends((50, 50), n_ridges=4, noise_level=0.4)),

        ("Adversarial: Anti-Powell\n(Misleading Ridges)",
         generator.adversarial_surface((50, 50), target_optimizer='powell')),

        ("Adversarial: Anti-Nelder-Mead\n(Narrow Valleys)",
         generator.adversarial_surface((50, 50), target_optimizer='nelder-mead'))
    ]

    for idx, (title, surface) in enumerate(surfaces):
        ax = axes[idx // 3, idx % 3]

        # Create 3D surface plot
        x = np.linspace(-2, 2, surface.shape[0])
        y = np.linspace(-2, 2, surface.shape[1])
        X, Y = np.meshgrid(x, y)

        # Normalize surface for better visualization
        surface_norm = (surface - surface.min()) / (surface.max() - surface.min())

        contour = ax.contourf(X, Y, surface_norm, levels=20, cmap='viridis', alpha=0.8)
        ax.contour(X, Y, surface_norm, levels=10, colors='white', alpha=0.3, linewidths=0.5)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_aspect('equal')

        # Add colorbar
        plt.colorbar(contour, ax=ax, shrink=0.6)

    plt.tight_layout()

    if save_plots:
        plt.savefig('stochastic_surfaces_demo.png', dpi=300, bbox_inches='tight')
        print("📊 Plots saved as 'stochastic_surfaces_demo.png'")

    plt.show()


def demonstrate_parameterization():
    """Show how surface characteristics can be controlled"""

    generator = SurfaceGenerator(seed=42)

    print("🎛️  Surface Parameterization Demo")
    print("=" * 50)

    # Test correlation length effects
    print("Testing correlation length effects on smoothness...")
    correlation_lengths = [0.1, 0.3, 0.8]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, corr_len in enumerate(correlation_lengths):
        surface = generator.gaussian_random_field((40, 40),
                                                 correlation_length=corr_len,
                                                 roughness=2.0)

        x = np.linspace(-1, 1, 40)
        y = np.linspace(-1, 1, 40)
        X, Y = np.meshgrid(x, y)

        contour = axes[i].contourf(X, Y, surface, levels=15, cmap='plasma')
        axes[i].set_title(f'Correlation Length = {corr_len}')
        axes[i].set_aspect('equal')
        plt.colorbar(contour, ax=axes[i], shrink=0.7)

    plt.suptitle('Effect of Correlation Length on Surface Smoothness')
    plt.tight_layout()
    plt.show()

    # Test multimodality control
    print("Testing multimodality control...")
    n_peaks_list = [2, 5, 10]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, n_peaks in enumerate(n_peaks_list):
        surface = generator.multi_modal_surface((40, 40),
                                              n_peaks=n_peaks,
                                              peak_strength_var=1.5)

        x = np.linspace(-2, 2, 40)
        y = np.linspace(-2, 2, 40)
        X, Y = np.meshgrid(x, y)

        contour = axes[i].contourf(X, Y, surface, levels=15, cmap='coolwarm')
        axes[i].set_title(f'Number of Peaks = {n_peaks}')
        axes[i].set_aspect('equal')
        plt.colorbar(contour, ax=axes[i], shrink=0.7)

    plt.suptitle('Effect of Peak Number on Surface Complexity')
    plt.tight_layout()
    plt.show()


def benchmark_generation_speed():
    """Test surface generation performance"""

    generator = SurfaceGenerator(seed=123)

    print("⚡ Surface Generation Speed Benchmark")
    print("=" * 50)

    sizes = [(30, 30), (50, 50), (100, 100)]
    methods = [
        ("Gaussian Random Field", generator.gaussian_random_field),
        ("Multi-Modal", lambda size: generator.multi_modal_surface(size, n_peaks=5)),
        ("Fractal Landscape", lambda size: generator.fractal_landscape(size, octaves=4)),
        ("Rugged Surface", lambda size: generator.rugged_surface_with_trends(size, n_ridges=3))
    ]

    for method_name, method_func in methods:
        print(f"\n{method_name}:")
        for size in sizes:
            start_time = time.time()
            surface = method_func(size)
            end_time = time.time()

            print(f"  {size[0]}×{size[1]}: {end_time - start_time:.3f}s")


def main():
    """Main demonstration function"""

    print("🌄 HumpDay Stochastic Surface Generation Demo")
    print("=" * 60)
    print("Generating diverse optimization landscapes for automated testing...\n")

    # Create generator
    generator = SurfaceGenerator(seed=42)

    # Generate and display various surfaces
    print("📊 Generating surface gallery...")
    plot_surfaces_grid(generator, save_plots=True)

    # Demonstrate parameter control
    demonstrate_parameterization()

    # Benchmark performance
    benchmark_generation_speed()

    print("\n✅ Demo complete!")
    print("\nKey insights:")
    print("• Can generate diverse, parameterized test problems automatically")
    print("• Surfaces range from smooth to highly multimodal")
    print("• Generation is fast enough for real-time use")
    print("• Can create adversarial problems targeting specific optimizers")
    print("• Perfect for systematic optimizer evaluation!")


if __name__ == "__main__":
    main()