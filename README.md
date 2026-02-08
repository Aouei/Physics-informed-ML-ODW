# Physics-informed ML ODW

Physics-informed Machine Learning for Optically-Derived Water (ODW) prediction from satellite imagery.

## Installation

```bash
pip install physics_informed_ml_odw
```

### Requirements

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

| Model | Description |
|-------|-------------|
| `ML` | Machine Learning MLP model |
| `CS_ML` | Caballero & Stumpf ML model |

## Example

```python
from sensingpy import reader, plot
from physics_informed_ml_odw import predict_2d

# Load image
image = reader.open('data/formosa_2018.tif')

# Predict using both models
image['ODW_ML'] = predict_2d(image, 'ML')
image['ODW_CS_ML'] = predict_2d(image, 'CS_ML')

# Visualize
fig, axs = plot.get_geofigure(image.crs, 1, 3, figsize=(18, 6))

plot.plot_rgb(image, 'Rrs_B4', 'Rrs_B3', 'Rrs_B2', axs[0], brightness=30)
plot.plot_band(image, 'ODW_ML', axs[1])
plot.plot_band(image, 'ODW_CS_ML', axs[2])

axs[0].set_title('True Color')
axs[1].set_title('ODW ML')
axs[2].set_title('ODW CS + ML')
```

## Documentation

```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
mkdocs serve
```

## License

MIT
