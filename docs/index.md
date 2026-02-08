# Physics-informed ML ODW

**Physics-informed Machine Learning for Optically-Derived Water (ODW) prediction.**

## Overview

This package provides pre-trained machine learning models for predicting Optically-Derived Water (ODW) values from satellite imagery. It integrates with [sensingpy](https://github.com/sensingpy/sensingpy) for image processing workflows.

## Features

- **Pre-trained Models**: Ready-to-use ML pipelines for ODW prediction
- **Multiple Model Types**: 
    - `ML`: Standard Machine Learning model (MLP)
    - `CS_ML`: Caballero & Stumpf Machine Learning model
- **Easy Integration**: Works seamlessly with sensingpy Image objects
- **2D Prediction**: Generate full raster predictions from satellite imagery

## Installation

```bash
pip install physics_informed_ml_odw
```

### Dependencies

- Python >= 3.11
- sensingpy
- numpy
- pandas
- scikit-learn == 1.7.2

## Quick Start

```python
from sensingpy import reader
from physics_informed_ml_odw import predict_2d

# Load satellite image
image = reader.open('path/to/image.tif')

# Predict ODW values
image['ODW'] = predict_2d(image, model='ML')
```

## Available Models

| Model | Description | File |
|-------|-------------|------|
| `ML` | Machine Learning MLP model | `ML__MLP.pkl` |
| `CS_ML` | Caballero & Stumpf ML model | `CS_ML__MLP.pkl` |

## License

This project is open source. See the repository for license details.
