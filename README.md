**Author:** Kevin Paniagua Romero  

[<img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" width="60">](https://www.linkedin.com/in/kevinpr/)

BayesInterpret is a computational project focused on creating interactive dashboards for interpreting and visualizing machine learning models using Dash. This project is part of the BayesInterpret initiative, where the main goal is to apply an intuitive interface to existing implementations, making model outputs and analyses easier to understand and use.

This work is based on ideas and research from the Computational Intelligence Group (CIG) and integrates interface design with machine learning tools to enhance interpretability in a simple and practical way. I'll be using Dash for the new interface.

[<img src="https://cig.fi.upm.es/wp-content/uploads/2023/11/cropped-logo_CIG.png" alt="CIG UPM" width="50">](https://cig.fi.upm.es)

# MPE Dashboard - Most Probable Explanation for Bayesian Networks

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Dash](https://img.shields.io/badge/dash-2.0+-green.svg)](https://dash.plotly.com/)
[![pgmpy](https://img.shields.io/badge/pgmpy-latest-orange.svg)](https://pgmpy.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Interactive dashboard for computing Most Probable Explanations (MPE) in Bayesian Networks** - Part of the BayesInterpret project.

## 🔬 What is Most Probable Explanation (MPE)?

MPE finds the **complete configuration of variable values** that maximizes the joint probability given observed evidence. Unlike marginal inference that provides probability distributions, MPE gives you the single most likely "explanation" for your evidence.

**Key Insight**: You cannot obtain the MPE by simply taking the maximum marginal probability of each variable independently. MPE ensures global consistency across all variables.

### Example
In a medical diagnosis network:
- **Evidence**: Patient has fever and cough
- **MPE**: Most probable complete diagnosis explaining these symptoms
- **Result**: `{Disease=Flu, Severity=Mild, Treatment=Rest}` with P=0.73

## 🎯 Features

### ✅ Current Implementation
- **📁 Dataset Upload**: Support for .bif files + default Asia network
- **🎯 Evidence Selection**: Interactive checkbox interface with dynamic dropdowns
- **⚡ Exact MPE Algorithm**: Variable Elimination with `map_query()` (pgmpy)
- **📊 Network Visualization**: Interactive Cytoscape.js with MPE states highlighted
- **📈 Probability Calculation**: Joint probability of the MPE configuration
- **🔧 Robust Error Handling**: Type validation and conversion for reliability

### 🔄 In Development
- **🤖 Approximate MPE**: Genetic Algorithms and EDAs for large networks
- **⏱️ Performance Benchmarks**: Exact vs Approximate algorithm comparison
- **📋 Enhanced Export**: CSV/JSON results export
- **🔍 Sensitivity Analysis**: MPE stability under parameter changes

## 🚀 Quick Start

### Prerequisites
```bash
pip install dash dash-bootstrap-components dash-cytoscape pgmpy pandas numpy
```

### Run the Dashboard
```bash
cd backend/cigmodelsdjango/cigmodelsdjangoapp/mpe-dash
python dash_mpe.py
```

Open your browser to `http://localhost:8057`

## 📖 Usage Guide

### 1. Load Your Bayesian Network
- **Upload .bif file**: Click the upload area and select your network
- **Use default**: Check "Use default Asia network" for quick testing

### 2. Select Evidence Variables
- **Choose variables**: Check boxes for observed variables
- **Set values**: Select the observed state for each evidence variable
- **Bulk operations**: Use "Select All" or "Clear All" buttons

### 3. Run MPE Computation
- Click **"Run MPE"** button
- View the most probable assignment for all variables
- See the joint probability of this explanation
- Explore the interactive network visualization

### 4. Interpret Results
- **MPE Table**: Shows variable assignments in the optimal explanation
- **Joint Probability**: Likelihood of this complete configuration
- **Network View**: Variables colored by evidence (blue) vs MPE assignment
- **State Labels**: Human-readable variable states when available

## 🔧 Technical Details

### Algorithms Implemented

#### Exact MPE (Current)
- **Method**: Variable Elimination with maximization (max-product)
- **Library**: pgmpy `VariableElimination.map_query()`
- **Complexity**: Exponential in treewidth, optimal for moderate networks
- **Guarantees**: Finds the global optimum

#### Approximate MPE (Planned)
- **Method**: Evolutionary algorithms (GA, EDA)
- **Use case**: Large networks where exact inference is intractable
- **Trade-off**: Speed vs optimality guarantee

### Supported File Formats
- **BIF**: Bayesian Interchange Format (`.bif`)
- **XMLBIF**: XML variant (`.xml`)
- **JSON**: Custom network format (`.json`)

## 🏗️ Architecture

```
mpe-dash/
├── dash_mpe.py           # Main application and callbacks
├── mpe_solver.py         # MPE algorithms and probability calculations
├── layout_mpe.py         # UI components (planned)
├── callbacks_mpe.py      # Dash callbacks (planned)
├── assets/               # CSS and JS files
├── models/               # Example networks (asia.bif)
└── networks/             # Additional test networks
```

## 🧪 Example Networks

### Default: Asia Network
A classic medical diagnosis network with 8 variables:
- **Nodes**: Visit to Asia, Smoking, Tuberculosis, Lung Cancer, Bronchitis, etc.
- **Use case**: Medical reasoning and diagnostic inference
- **Size**: Small (8 nodes) - perfect for testing and demonstrations

## 📊 Performance Notes

| Network Size | Exact MPE | Approximate MPE |
|--------------|-----------|-----------------|
| Small (≤20)  | < 1 sec   | < 1 sec         |
| Medium (≤50) | 1-10 sec  | < 1 sec         |
| Large (>50)  | Variable* | < 5 sec         |

*Depends on network structure (treewidth)

## 🔬 Scientific Background

MPE is a fundamental inference task in probabilistic graphical models:

- **Formal Definition**: `argmax_x P(x | evidence)`
- **Applications**: Diagnosis, troubleshooting, configuration optimization
- **Complexity**: NP-hard in general case
- **Related Work**: MAP inference, Viterbi algorithm (for temporal models)

### Key References
- Darwiche, A. (2009). "Modeling and Reasoning with Bayesian Networks"
- Pearl, J. (1988). "Probabilistic Reasoning in Intelligent Systems"
- Chan & Darwiche (2012). "On the robustness of Most Probable Explanations"

## 🤝 Contributing

This dashboard is part of the **BayesInterpret** project. For contributions:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📞 Support

For questions about MPE Dashboard:
- **Issues**: [GitHub Issues](https://github.com/KeevinPR/mpe-dash/issues)
- **Documentation**: [BayesInterpret Docs](https://bayes-interpret.com/)
- **Research Group**: [CIG - UPM](https://cig.fi.upm.es/)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**MPE Dashboard** - Making Bayesian network explanations intuitive and accessible 🧠✨ 