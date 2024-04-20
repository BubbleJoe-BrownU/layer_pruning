import numpy as np

def calculate_angular_distance(embed_1, embed_2):
    """
    Calculate the Angular Distance defined as:
    d(x^{l}, x^{l+n}) = \frac {1} {\pi} arccos(cos(embed_1, embed_2))
    """
    return np.arccos(embed_1 @ embed_2.T / np.linalg.norm(embed_1, axis=-1) / np.linalg.norm(embed_2, axis=-1)) / np.pi
