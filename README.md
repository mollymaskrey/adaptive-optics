# Adaptive Optics Wavefront Correction Simulator

This project simulates Zernike-mode-based adaptive optics (AO) wavefront correction, with an interactive interface for visualizing correction steps, residual error, and actuator influence. It is designed as a modular, educational, and research-grade tool for exploring fundamental AO concepts in a clear and extensible way.

## Features

- ✅ Simulated wavefront error based on Zernike polynomials  
- ✅ Interactive UI for adjusting correction parameters  
- ✅ Real-time visual output of corrected vs uncorrected wavefronts  
- ✅ Modular architecture for easy adaptation to different optical systems  
- ✅ D-Wave QPU-ready scripts for quantum-based AO optimization (optional)

## Technologies Used

- Python (NumPy, SciPy, Matplotlib)
- Gradio (for GUI simulation interface)
- Zernike mode decomposition & reconstruction
- Optional: D-Wave Ocean SDK for QUBO-based actuator optimization

## Project Structure

adaptive-optics/
├── ao_core/ # Core AO logic: Zernike, correction matrix, error calc
├── gui/ # Gradio UI layout and visual controls
├── quantum/ # Optional D-Wave integration (QUBO model, solver call)
├── tests/ # Test cases and demo modes
├── assets/ # Visual references, diagrams
├── README.md
└── requirements.txt

bash
Copy
Edit



