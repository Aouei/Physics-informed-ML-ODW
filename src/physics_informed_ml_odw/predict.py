import numpy as np
import pandas as pd
import joblib
from importlib import resources
from typing import Literal

from sensingpy.image import Image
from sklearn.pipeline import Pipeline


MODEL_FILES = {
    "ML": "ML__MLP.pkl",
    "CS_ML": "CS_ML__MLP.pkl",
}


def _load_model(model: Literal["ML", "CS_ML"]) -> Pipeline:
    """Load a pre-trained model from the package resources.
    
    Parameters
    ----------
    model : Literal["ML", "CS_ML"]
        The model type to load.
        
    Returns
    -------
    Pipeline
        The loaded scikit-learn pipeline.
        
    Raises
    ------
    ValueError
        If the model type is not recognized.
    """
    if model not in MODEL_FILES:
        raise ValueError(f"Model must be one of {list(MODEL_FILES.keys())}, got '{model}'")
    
    model_filename = MODEL_FILES[model]
    with resources.files("physics_informed_ml_odw.models").joinpath(model_filename).open("rb") as f:
        return joblib.load(f)


def predict_2d(image: Image, model: Literal["ML", "CS_ML"] = "ML") -> np.ndarray:
    """Predict ODW values for a 2D image using a pre-trained model.
    
    Parameters
    ----------
    image : Image
        The input image from sensingpy.
    model : Literal["ML", "CS_ML"], default="ML"
        The model type to use for prediction:
        - "ML": Machine Learning model
        - "CS_ML": Compressed Sensing Machine Learning model
        
    Returns
    -------
    np.ndarray
        2D array with predicted ODW values.
    """
    pipeline = _load_model(model)
    
    X = pd.DataFrame({band: image.select(band).ravel() for band in pipeline.feature_names_in_}).dropna()

    ODW = np.full(shape=(image.height, image.width), fill_value=np.nan, dtype=X.dtypes.iloc[0]).ravel()
    ODW[X.index] = pipeline.predict(X)
    ODW = ODW.reshape((image.height, image.width))

    return ODW
