import numpy as np
import pandas as pd

from sensingpy.image import Image
from sklearn.pipeline import Pipeline


def predict_2d(image: Image, model : Pipeline) -> np.ndarray:
    X = pd.DataFrame( { band : image.select(band).ravel() for band in model.feature_names_in_ }).dropna()

    ODW = np.full(shape = (image.height, image.width), fill_value = np.nan, dtype = X.dtypes.iloc[0]).ravel()
    ODW[X.index] = model.predict(X)
    ODW = ODW.reshape((image.height, image.width))

    return ODW
