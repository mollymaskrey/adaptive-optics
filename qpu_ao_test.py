"""
Quantum Adaptive Optics Wavefront Correction

This module demonstrates quantum computing applied to adaptive optics, using D-Wave's
quantum annealer to optimize deformable mirror actuator positions. Unlike classical
gradient-based methods, this approach leverages quantum mechanics to explore the
solution space and find optimal corrections for atmospheric distortions.

The quantum advantage emerges from the ability to handle the quadratic interactions
between actuators naturally - the quantum annealer can explore many actuator
combinations simultaneously through quantum superposition and tunneling.

Key Innovations:
- Formulates AO correction as a Quadratic Unconstrained Binary Optimization (QUBO)
- Uses integer variables to approximate continuous actuator positions
- Leverages quantum annealing to handle actuator coupling interactions
- Demonstrates potential for real-time wavefront correction

Physical Context:
In astronomy, atmospheric turbulence distorts light from stars and galaxies.
Adaptive optics systems use deformable mirrors with many actuators to correct
these distortions in real-time. Finding optimal actuator positions is challenging
because actuators influence each other - a perfect quantum optimization problem.

Requirements:
    - numpy: Numerical operations
    - matplotlib: Visualization
    - aotools: Zernike polynomial generation
    - dwave-system: Access to D-Wave quantum annealer
    - dimod: Quantum problem formulation
    - DWAVE_SAMPLER_TOKEN: Authentication for D-Wave Leap cloud
"""

import numpy as np
import matplotlib.pyplot as plt
import aotools
from dwave.system import LeapHybridCQMSampler
import dimod
import os

# === Token Setup ===
"""
D-Wave authentication setup.

The quantum annealer is accessed through D-Wave's cloud service.
Token can be obtained from: https://cloud.dwavesys.com/leap/
"""
token = os.getenv("DWAVE_SAMPLER_TOKEN")
if token is None:
    raise ValueError("DWAVE_SAMPLER_TOKEN is not defined!")

# === Step 1: Generate distorted wavefront with Z4, Z5, Z6 ===
"""
Create realistic atmospheric distortions using Zernike polynomials.

Zernike modes represent different types of optical aberrations:
- Z4: 45° astigmatism (cylindrical distortion)
- Z5: Vertical coma (comet-like blur)
- Z6: Horizontal coma

These low-order modes dominate atmospheric turbulence and are
prime targets for AO correction.
"""
N = 64  # Grid resolution (64x64 pixels for faster quantum processing)
z_modes = [4, 5, 6]  # Zernike mode numbers
amps = [1.0, 0.8, 1.2]  # Relative strengths of each aberration

# Generate Zernike basis functions
zernike_modes = aotools.zernike.zernikeArray(max(z_modes), N)

# Combine modes to create realistic distortion pattern
distorted_wavefront = sum(a * zernike_modes[z - 1] for a, z in zip(amps, z_modes))

# === Step 2: Create circular aperture mask ===
"""
Define the telescope aperture (circular boundary).

Real telescopes have circular mirrors, so we only correct
aberrations within this circular region. Points outside
the aperture are ignored.
"""
x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x)
R = np.sqrt(X**2 + Y**2)
mask = R <= 1  # Binary mask: True inside aperture
wavefront_region = distorted_wavefront * mask

# === Step 3: Build influence functions for 10x10 grid ===
"""
Model how each actuator affects the mirror surface.

Each actuator on the deformable mirror creates a localized
deformation pattern (influence function). We use Gaussian
profiles to model this physical behavior.

Grid configuration:
- 10x10 actuators = 100 control points
- Even spacing across the aperture
- Influence overlap ensures smooth correction
"""
grid_size = 10
actuator_spacing = 2 / grid_size

# Create actuator grid positions
ax = np.linspace(-1 + actuator_spacing / 2, 1 - actuator_spacing / 2, grid_size)
ay = ax.copy()
act_X, act_Y = np.meshgrid(ax, ay)
positions = np.vstack([act_X.ravel(), act_Y.ravel()]).T

def influence_fn(x0, y0, sigma=0.18):
    """
    Compute influence pattern for a single actuator.
    
    Models the physical deformation caused by pushing/pulling one actuator
    on the deformable mirror. Uses a 2D Gaussian profile to represent
    the localized but smooth deformation.
    
    Args:
        x0, y0 (float): Actuator position in normalized coordinates [-1, 1]
        sigma (float): Influence width (0.18 = moderate coupling)
            - Smaller sigma = more localized control
            - Larger sigma = smoother surface but less precise
            
    Returns:
        np.ndarray: 2D influence pattern (64x64) showing surface deformation
        
    Physical Interpretation:
        Real DM actuators create volcano-like deformations with ~2-3 actuator
        spacing influence radius, well-approximated by Gaussians.
    """
    return np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

# Pre-compute all 100 influence functions (only within aperture for efficiency)
influences = np.array([influence_fn(x, y)[mask] for x, y in positions])
num_actuators = influences.shape[0]

# Target: flatten the distorted wavefront (negative of distortion)
target = wavefront_region[mask]

# === Step 4: Build CQM with INTEGER variables and interactions ===
"""
Formulate the quantum optimization problem.

The core insight: wavefront correction is a quadratic optimization problem
because actuator influences overlap and interact. This maps naturally to
quantum annealing hardware designed for quadratic problems.

Integer Variable Trick:
We use integers in [-150, 150] to represent continuous values in [-1.5, 1.5]
with 0.01 resolution. This discretization is necessary because current
quantum annealers work with discrete variables.

Mathematical Formulation:
Minimize: ||target - Σ(weight_i * influence_i)||²
Expands to: Σᵢⱼ wᵢwⱼ(Iᵢ·Iⱼ) - 2Σᵢ wᵢ(Iᵢ·target) + const

Where:
- wᵢ: Weight for actuator i (decision variable)
- Iᵢ: Influence function for actuator i
- target: Desired flat wavefront
"""
scale = 100  # Scaling factor: integer 100 = real value 1.0
lower_bound, upper_bound = -150, 150  # Allows actuator range [-1.5, 1.5]

# Initialize quadratic model
qm = dimod.QuadraticModel()

# Add integer variables (one per actuator)
for i in range(num_actuators):
    qm.add_variable(dimod.INTEGER, f"w_{i}", lower_bound=lower_bound, upper_bound=upper_bound)

# Add quadratic interaction terms (actuator coupling)
# These represent how actuators interfere with each other
for i in range(num_actuators):
    for j in range(i, num_actuators):
        # Coefficient represents overlap between influence functions
        coeff = 2 * np.dot(influences[i], influences[j])
        qm.add_quadratic(f"w_{i}", f"w_{j}", coeff)

# Add linear terms (direct correction strength)
# These represent how well each actuator corrects the target aberration
for i in range(num_actuators):
    coeff = -2 * scale * np.dot(influences[i], target)
    qm.add_linear(f"w_{i}", coeff)

# Add small L2 regularization (prevents extreme actuator positions)
for i in range(num_actuators):
    qm.add_quadratic(f"w_{i}", f"w_{i}", 0.01)  # Tiny penalty on squared weights

# Convert to Constrained Quadratic Model (required for D-Wave)
cqm = dimod.ConstrainedQuadraticModel()
cqm.set_objective(qm)

# === Step 5: Solve CQM ===
"""
Submit problem to D-Wave quantum annealer.

The quantum annealer explores the energy landscape using quantum
mechanics, potentially finding better solutions faster than classical
methods for this 100-variable quadratic problem.

Quantum Process:
1. Problem encoded into qubit couplings
2. System initialized in superposition state
3. Quantum annealing gradually reduces quantum fluctuations
4. System settles into low-energy state (good solution)
5. Multiple samples taken to ensure quality
"""
sampler = LeapHybridCQMSampler(token=token)

# Submit to quantum cloud (60 second time limit allows thorough search)
sampleset = sampler.sample_cqm(cqm, 
                               time_limit=60, 
                               label="AO Hybrid Wavefront Correction (Integer)")

# === Step 6: Extract best result and scale weights ===
"""
Convert quantum solution back to physical actuator positions.

The quantum annealer returns integer values which we scale back
to real actuator positions in the range [-1.5, 1.5].
"""
best = sampleset.first.sample  # Best solution found
weights = np.array([best[f"w_{i}"] / scale for i in range(num_actuators)])

# === Step 7: Apply correction ===
"""
Compute the corrected wavefront using the quantum-optimized actuator positions.

This simulates the physical process where the DM actuators deform the mirror
surface to compensate for atmospheric distortions.
"""
# Calculate total correction from all actuators
flat_correction = np.dot(weights, influences)

# Apply correction to distorted wavefront
corrected = target - flat_correction
corrected_wavefront = distorted_wavefront.copy()
corrected_wavefront[mask] = corrected

# Compute residual error (how well we corrected)
residual = corrected  # Remaining error after correction

# === Step 8: Plot results ===
"""
Visualize the quantum optimization results.

Three panels show:
1. Original distorted wavefront (what atmosphere does)
2. Corrected wavefront (after quantum-optimized DM)
3. Residual error map (remaining aberrations)

Success is indicated by:
- Corrected wavefront appearing flat (uniform color)
- Residual map showing small values (close to zero)
- RMS error reduction of >80%
"""
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Common color scale for wavefront maps
vmin, vmax = -2, 2

# Panel 1: Input aberration
axs[0].imshow(distorted_wavefront, cmap='viridis', vmin=vmin, vmax=vmax)
axs[0].set_title("Distorted Wavefront")

# Panel 2: After quantum correction
axs[1].imshow(corrected_wavefront, cmap='viridis', vmin=vmin, vmax=vmax)
axs[1].set_title("Corrected Wavefront")

# Panel 3: Remaining error (ideally near zero)
residual_map = np.zeros_like(distorted_wavefront)
residual_map[mask] = residual
axs[2].imshow(residual_map, cmap='bwr', vmin=-0.5, vmax=0.5)
axs[2].set_title("Residual (Corrected - Target)")

plt.tight_layout()
plt.show()
