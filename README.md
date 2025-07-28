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

adaptive-optics/#
├── xxx.py/#
├── yyy.py
└── README.md


## Background
This simulator was built as a proof of concept by Molly Maskrey, drawing on years of experience in adaptive optics at AMOS atop Haleakalā. The goal is to provide a research-grade sandbox for exploring classical and quantum AO optimization strategies.

## Future Enhancements
🌐 Web-based deployment using Dash or Flask

📡 Real-time actuator data integration

🧠 Reinforcement learning for dynamic correction control

⚛️ Hybrid quantum-classical solvers for multi-mode AO




