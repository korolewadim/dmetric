import numpy as np
import numba as nb
from ase.atoms import Atoms
from dscribe.descriptors import SOAP
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_kernels


def _get_species(structures):
    species = [structure.get_atomic_numbers().tolist() for structure in structures]
    species = sorted(set([item for sublist in species for item in sublist]))
    return species


def _get_csr_features(structures, species, rcut, nmax, lmax):
    averaged_soap = SOAP(
        species=species,
        rcut=rcut,
        nmax=nmax,
        lmax=lmax,
        average='inner',
        sparse=True,
        periodic=True,
        dtype='float32',
    )
    
    raw_features = averaged_soap.create(structures, n_jobs=-1)
    csr_features = raw_features.tocsr()
    nonzero_csr_features = csr_features[:, csr_features.getnnz(0) > 0]
    
    return nonzero_csr_features.astype(np.float32)


@nb.njit(cache=False)
def _get_max_n_values(data, indices, indptr, K):
    m = indptr.shape[0] - 1
    max_n_indices = []
    max_n_values = np.zeros((m, K), dtype=data.dtype)
    
    for i in nb.prange(m):
        actual_K = min(indptr[i+1] - indptr[i], K)
        max_inds = np.argsort(data[indptr[i]:indptr[i+1]])[::-1][:actual_K]
        max_n_indices.append(indices[indptr[i]:indptr[i+1]][max_inds])
        max_n_values[i][:actual_K] = data[indptr[i]:indptr[i+1]][max_inds]
        
    return max_n_values, max_n_indices


def _get_neighbors(train_structures, test_structures, n_neighbors, kernel_degree, rcut, nmax, lmax):
    all_structures = train_structures + test_structures
    species = _get_species(all_structures)
    
    all_csr_features = _get_csr_features(all_structures, species, rcut, nmax, lmax)
    train_csr_features = all_csr_features[:len(train_structures)]
    test_csr_features = all_csr_features[len(train_structures):]
    
    kernels = csr_matrix(
        pairwise_kernels(
            test_csr_features,
            train_csr_features,
            n_jobs=-1,
            metric='cosine',
        ) ** kernel_degree
    ).astype(np.float32)
    
    k_neighbors, max_n_indices = _get_max_n_values(
        kernels.data, kernels.indices, kernels.indptr, n_neighbors
    )
    
    return k_neighbors, max_n_indices


def get_delta_metrics(
    structures_train: list[Atoms],
    structures_test: list[Atoms],
    errors: np.ndarray,
    n_neighbors: int = 100,
    kernel_degree: int = 4,
    rcut: float = 6.0,
    nmax: int = 8,
    lmax: int = 6,
):
    neighbors, ids = _get_neighbors(
        structures_train,
        structures_test,
        n_neighbors=n_neighbors,
        kernel_degree=kernel_degree,
        rcut=rcut, nmax=nmax, lmax=lmax,
    )
    
    delta_metrics = np.array(
        [np.sum(n[:i.shape[0]]*np.abs(errors[i]))/np.sum(n[:i.shape[0]]) for n, i in zip(neighbors, ids)]
    )
    
    return delta_metrics
