# API Reference

This page provides detailed documentation for all public functions in the `physics_informed_ml_odw` package.

## Module: physics_informed_ml_odw

::: physics_informed_ml_odw.predict
    options:
      show_root_heading: false
      heading_level: 2
      members:
        - predict_2d
        - load_model

---

## Functions

### predict_2d

```python
predict_2d(image: Image, model: Literal["ML", "CS_ML"] = "ML") -> np.ndarray
```

Predict ODW values for a 2D image using a pre-trained model.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | `Image` | *required* | The input image from sensingpy |
| `model` | `Literal["ML", "CS_ML"]` | `"ML"` | The model type to use for prediction |

**Returns**

| Type | Description |
|------|-------------|
| `np.ndarray` | 2D array with predicted ODW values |

**Model Options**

- `"ML"`: Standard Machine Learning model
- `"CS_ML"`: Caballero & Stumpf Machine Learning model

**Example**

```python
from sensingpy import reader
from physics_informed_ml_odw import predict_2d

image = reader.open('satellite_image.tif')
odw_prediction = predict_2d(image, model='ML')
```

---

### load_model

```python
load_model(model: Literal["ML", "CS_ML"]) -> Pipeline
```

Load a pre-trained model from the package resources.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `Literal["ML", "CS_ML"]` | *required* | The model type to load |

**Returns**

| Type | Description |
|------|-------------|
| `Pipeline` | The loaded scikit-learn pipeline |

**Raises**

| Exception | Description |
|-----------|-------------|
| `ValueError` | If the model type is not recognized |

**Example**

```python
from physics_informed_ml_odw import load_model

# Load the ML model pipeline
pipeline = load_model('ML')

# Access model components
print(pipeline.feature_names_in_)
```

---

## Constants

### MODEL_FILES

Dictionary mapping model names to their corresponding pickle files.

```python
MODEL_FILES = {
    "ML": "ML__MLP.pkl",
    "CS_ML": "CS_ML__MLP.pkl",
}
```
