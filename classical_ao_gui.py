"""
Adaptive Optics Wavefront Correction Simulator

This module provides an interactive simulation of adaptive optics (AO) systems used
in astronomy and vision science. It demonstrates how deformable mirrors can correct
for atmospheric distortions in real-time using classical optimization techniques.

Key Features:
- Models Zernike polynomial wavefront aberrations (modes 4-6)
- Simulates deformable mirror with configurable actuator grid
- Uses influence functions to model actuator behavior
- Implements smoothness constraints via Laplacian penalty
- Provides real-time visualization through Gradio interface

The simulation shows the fundamental challenge in AO: finding optimal actuator
commands to flatten a distorted wavefront, similar to how telescopes correct
for atmospheric turbulence to achieve sharp images of stars.

Requirements:
    - numpy: Numerical operations
    - matplotlib: Visualization
    - scipy: Optimization algorithms
    - aotools: Zernike polynomial generation
    - gradio: Interactive web interface
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import aotools
import gradio as gr

def ao_simulator(use_z4, z4_amp, use_z5, z5_amp, use_z6, z6_amp, actuator_grid, smoothing_weight):
    """
    Simulate adaptive optics correction of Zernike wavefront aberrations.
    
    This function models a complete AO correction cycle: measuring wavefront
    distortions, computing optimal deformable mirror commands, and applying
    the correction. It uses classical optimization (L-BFGS-B) to find the
    best actuator positions that minimize residual wavefront error.
    
    Args:
        use_z4 (bool): Enable Zernike mode 4 (45° astigmatism)
        z4_amp (float): Amplitude of Z4 aberration (-2 to 2)
        use_z5 (bool): Enable Zernike mode 5 (vertical coma)
        z5_amp (float): Amplitude of Z5 aberration (-2 to 2)
        use_z6 (bool): Enable Zernike mode 6 (horizontal coma)
        z6_amp (float): Amplitude of Z6 aberration (-2 to 2)
        actuator_grid (int): Number of actuators per side (4-20)
            - Higher values = better correction but more computation
            - Typical AO systems use 10-50 actuators across
        smoothing_weight (float): Laplacian regularization strength (0-0.01)
            - Prevents adjacent actuators from fighting each other
            - Higher values = smoother DM surface but worse correction
            
    Returns:
        matplotlib.figure.Figure: 3x2 subplot figure containing:
            - Distorted wavefront (input aberration)
            - DM correction surface (what the mirror does)
            - Corrected wavefront (residual error)
            - Actuator weight map (individual actuator commands)
            - RMS error vs time plot (convergence history)
            - Residual error map (spatial error distribution)
            
    Technical Details:
        The optimization minimizes: RMS_error + λ * smoothness_penalty
        where smoothness is measured by the discrete Laplacian of actuator weights.
        This prevents unrealistic mirror shapes while maintaining good correction.
    """
    # Create computational grid (256x256 pixels over unit circle)
    N = 256
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    mask = R <= 1  # Circular aperture mask

    # Build distorted wavefront from selected Zernike modes
    distorted_wavefront = np.zeros((N, N))
    if use_z4:
        z4 = aotools.zernike.zernikeArray(4, N)[-1]
        distorted_wavefront += z4_amp * z4
    if use_z5:
        z5 = aotools.zernike.zernikeArray(5, N)[-1]
        distorted_wavefront += z5_amp * z5
    if use_z6:
        z6 = aotools.zernike.zernikeArray(6, N)[-1]
        distorted_wavefront += z6_amp * z6
    distorted_wavefront *= mask

    # Configure deformable mirror actuator positions
    actuator_spacing = 2 / actuator_grid
    actuator_x = np.linspace(-1 + actuator_spacing/2, 1 - actuator_spacing/2, actuator_grid)
    actuator_y = actuator_x.copy()
    act_X, act_Y = np.meshgrid(actuator_x, actuator_y)
    actuator_positions = np.vstack([act_X.ravel(), act_Y.ravel()]).T

    def influence_fn(x0, y0, sigma=0.25):
        """
        Compute influence function for a single actuator.
        
        Models how pushing/pulling one actuator affects the mirror surface
        using a 2D Gaussian profile. Real DMs have more complex coupling.
        
        Args:
            x0, y0 (float): Actuator position in normalized coordinates
            sigma (float): Influence width (default 0.25 = 1/4 aperture)
            
        Returns:
            np.ndarray: 2D influence pattern (256x256)
        """
        return np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

    # Pre-compute all actuator influence functions
    influences = np.array([influence_fn(x0, y0) for x0, y0 in actuator_positions])
    
    # Flatten for efficient matrix operations (only compute over aperture)
    flat_influence = influences.reshape(len(influences), -1)[:, mask.ravel()]

    # Track optimization progress
    rms_history = []

    def laplacian_penalty(weights_2d):
        """
        Compute discrete Laplacian penalty for actuator smoothness.
        
        Penalizes large differences between adjacent actuators to prevent
        unrealistic mirror shapes (e.g., adjacent actuators pulling in
        opposite directions). Uses periodic boundary conditions.
        
        Args:
            weights_2d (np.ndarray): Actuator weights in grid layout
            
        Returns:
            float: Sum of squared Laplacian values (smoothness penalty)
        """
        lap = (
            -4 * weights_2d
            + np.roll(weights_2d, 1, axis=0)   # Up neighbor
            + np.roll(weights_2d, -1, axis=0)  # Down neighbor
            + np.roll(weights_2d, 1, axis=1)   # Left neighbor
            + np.roll(weights_2d, -1, axis=1)  # Right neighbor
        )
        return np.sum(lap**2)

    def loss(weights_flat):
        """
        Compute total loss function for optimization.
        
        Combines wavefront correction quality (RMS error) with actuator
        smoothness constraint. This is the function being minimized.
        
        Args:
            weights_flat (np.ndarray): Flattened actuator weights (1D)
            
        Returns:
            float: Total loss = RMS_error + λ * smoothness_penalty
            
        Side Effects:
            Appends current RMS error to rms_history for plotting
        """
        # Compute DM surface from actuator weights
        correction = np.dot(weights_flat, flat_influence)
        
        # Calculate residual wavefront error
        residual = distorted_wavefront[mask] - correction
        rms = np.sqrt(np.mean(residual**2))
        rms_history.append(rms)
        
        # Add smoothness regularization
        weights_2d = weights_flat.reshape(actuator_grid, actuator_grid)
        smooth_penalty = laplacian_penalty(weights_2d)
        
        return rms + smoothing_weight * smooth_penalty

    # Initialize optimization with all actuators at zero position
    w0 = np.zeros(actuator_grid**2)
    
    # Run L-BFGS-B optimization (gradient-based, handles bounds)
    opt = minimize(loss, w0, method='L-BFGS-B')
    best_weights = opt.x

    # Reconstruct full correction surface from optimal weights
    correction_surface = np.tensordot(best_weights, influences, axes=1)
    corrected_wavefront = distorted_wavefront - correction_surface
    actuator_map = best_weights.reshape((actuator_grid, actuator_grid))
    residual = corrected_wavefront * mask

    # Create comprehensive visualization
    fig, axs = plt.subplots(3, 2, figsize=(12, 14))
    vmin, vmax = -3, 3  # Consistent colorbar range

    # Top left: Input aberration
    axs[0, 0].imshow(distorted_wavefront, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title("Distorted Wavefront")

    # Top right: What the DM does
    axs[0, 1].imshow(correction_surface, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title("DM Correction Surface")

    # Middle left: Result after correction
    axs[1, 0].imshow(corrected_wavefront, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1, 0].set_title("Corrected Wavefront")

    # Middle right: Individual actuator commands
    axs[1, 1].imshow(actuator_map, cmap='coolwarm', interpolation='nearest')
    axs[1, 1].set_title("Actuator Weight Map")

    # Bottom left: Convergence history with realistic timing
    axs[2, 0].set_title("RMS Error per Optimization Step (2.5ms actuator)")
    actuator_step_time = 0.0025  # 2.5 milliseconds per step (400Hz)
    time_axis = np.arange(len(rms_history)) * actuator_step_time
    axs[2, 0].plot(time_axis, rms_history)
    axs[2, 0].set_xlabel("Time (seconds)")
    axs[2, 0].set_ylabel("RMS Error")

    # Bottom right: Spatial distribution of residual error
    axs[2, 1].imshow(residual, cmap='plasma', vmin=vmin, vmax=vmax)
    axs[2, 1].set_title("Residual Map")

    plt.tight_layout()
    return fig

# Create Gradio interface
with gr.Blocks() as demo:
    """
    Gradio interface for interactive AO simulation.
    
    Provides real-time control over:
    - Aberration types and strengths
    - DM actuator density
    - Smoothness constraints
    
    This mimics actual AO system control interfaces used at observatories.
    """
    gr.Markdown("## Adaptive Optics: Zernike Mode Correction Classical Demo")

    with gr.Column():
        # Zernike mode 4 controls (45° astigmatism)
        with gr.Row():
            z4_checkbox = gr.Checkbox(label="Z4", value=True)
            z4_slider = gr.Slider(-2, 2, value=1.0, step=0.1, label="Z4 Amplitude")

        # Zernike mode 5 controls (vertical coma)
        with gr.Row():
            z5_checkbox = gr.Checkbox(label="Z5", value=True)
            z5_slider = gr.Slider(-2, 2, value=1.0, step=0.1, label="Z5 Amplitude")

        # Zernike mode 6 controls (horizontal coma)
        with gr.Row():
            z6_checkbox = gr.Checkbox(label="Z6", value=True)
            z6_slider = gr.Slider(-2, 2, value=1.0, step=0.1, label="Z6 Amplitude")

        # System configuration
        with gr.Row():
            actuator_slider = gr.Slider(4, 20, value=10, step=1, label="Actuator Grid Size")
            smooth_slider = gr.Slider(0.0, 0.01, value=0.001, step=0.0001, label="Smoothing Weight")

        run_btn = gr.Button("Run Simulation")
        output_plot = gr.Plot()

        # Connect interface to simulator
        run_btn.click(
            fn=ao_simulator,
            inputs=[
                z4_checkbox, z4_slider,
                z5_checkbox, z5_slider,
                z6_checkbox, z6_slider,
                actuator_slider,
                smooth_slider
            ],
            outputs=output_plot
        )

if __name__ == "__main__":
    demo.launch()
