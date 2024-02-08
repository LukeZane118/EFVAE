import numpy as np
import torch


def construct_index(cd01, K):
    """
    Returns:
        indices(tensor): Index of item after sorting by index of codebook.
        indptr(tensor): Cumulative sum of items under the codebook after sorting.
    """
    # Stable is availabel in PyTorch 1.9. Earlier version is not supported.
    cd01, indices = torch.sort(cd01)
    # save the indices according to the cluster 
    cluster, count = torch.unique_consecutive(cd01, return_counts=True)
    count_all = torch.zeros(K ** 2 + 1, dtype=torch.long, device=cd01.device)
    count_all[cluster + 1] = count
    indptr = count_all.cumsum(dim=-1)
    return indices, indptr