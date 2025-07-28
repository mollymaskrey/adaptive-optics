# Adaptive Optics Wavefront Correction Simulator

This project simulates Zernike-mode-based adaptive optics (AO) wavefront correction, with an interactive interface for visualizing correction steps, residual error, and actuator influence. It is designed as a modular, educational, and research-grade tool for exploring fundamental AO concepts in a clear and extensible way.

## Features

- Simulated wavefront error based on Zernike polynomials  
- Interactive UI for adjusting correction parameters  
- Real-time visual output of corrected vs uncorrected wavefronts  
- Modular architecture for easy adaptation to different optical systems  
- D-Wave QPU-ready scripts for quantum-based AO optimization (optional)

## Technologies Used

- Python (NumPy, SciPy, Matplotlib)
- Gradio (for GUI simulation interface)
- Zernike mode decomposition & reconstruction
- Optional: D-Wave Ocean SDK for QUBO-based actuator optimization

## Project Structure
```
adaptive-optics/#
├── classical_ao_gui.py    # classic AO with UI Interface
├── qpu_ao_test.py         # time limitations forced me to run in a notebook
├── AO_Paper.pdf           # Overview paper
└── README.md
```

## Background
This simulator was built as a proof of concept by Molly Maskrey, drawing on years of experience in adaptive optics at AMOS atop Haleakalā. The goal is to provide a research-grade sandbox for exploring classical and quantum AO optimization strategies. The idea came fom my experience taking the D-Wave Quantum Core training. Finding that I had solver time remaining after completing the class, I wanted
gain more experience using the actual hardware while I had the chance.

## Future Enhancements

• Web-based deployment using Dash or Flask

• Real-time actuator data integration

• Reinforcement learning for dynamic correction control

• Hybrid quantum-classical solvers for multi-mode AO




