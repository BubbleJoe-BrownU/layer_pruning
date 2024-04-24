import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_angular_distance(embed_1, embed_2):
    """
    Calculate the Angular Distance defined as:
    d(x^{l}, x^{l+n}) = \frac {1} {\pi} arccos(cos(embed_1, embed_2))
    params:
        embed_1: the input of layer x^{l}, shape [N, L, H]
        embed_2: the input of layer x^{l+n}, shape [N, L, H]
    return:
        angular distance averaged on the batch dimension and the sequence length dimension
    """
    embed_1 = F.normalize(embed_1, dim=-1)
    embed_2 = F.normalize(embed_2, dim=-1)
    results = torch.zeros((embed_1.shape[0],))
    for i, (e1, e2) in enumerate(zip(embed_1, embed_2)):
        print(e1.shape)
        results[i] = e1 @ e2.T
    return torch.arccos(results).mean() / torch.pi
    # return np.arccos(embed_1 @ embed_2.T / (np.linalg.norm(embed_1, axis=-1) + 1e-6) / (np.linalg.norm(embed_2, axis=-1) + 1e-6)) / np.pi

if __name__ == "__main__":
    input_1 = torch.zeros((10, 20))
    input_2 = torch.ones((10, 20))

    angular_distance_1 = calculate_angular_distance(input_1, input_2)
    print(angular_distance_1)
    # print(np.linalg.norm(angular_distance_1 - np.ones_like(angular_distance_1)/2))