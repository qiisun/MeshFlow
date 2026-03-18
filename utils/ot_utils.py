import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def optimal_sum_numpy(data, noise, optimal=False, dimension=3, flip_face=True):
    """Reorder `noise` to best match `data` with Hungarian assignment."""
    num_faces = data.shape[0]
    data_flat = data.reshape(num_faces, -1)

    if optimal:
        if flip_face:
            face_permutations = np.array(list(itertools.permutations([0, 1, 2])), dtype=np.int32)
        else:
            face_permutations = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.int32)

        cost_matrices = []
        for perm in face_permutations:
            noise_perm = noise[:, perm].reshape(num_faces, -1)
            cost_matrices.append(cdist(data_flat, noise_perm))

        cost_matrices = np.stack(cost_matrices)
        cost_matrix = np.min(cost_matrices, axis=0)
        best_perm_indices = np.argmin(cost_matrices, axis=0)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        best_perms_for_matches = face_permutations[best_perm_indices[row_ind, col_ind]]
        noise_reordered = noise[col_ind]
        face_indices = np.arange(num_faces)[:, None]
        noise_reordered = noise_reordered[face_indices, best_perms_for_matches, :]
        return noise_reordered

    cost_matrix = cdist(data_flat, noise.reshape(num_faces, -1))
    _, col_ind = linear_sum_assignment(cost_matrix)
    return noise[col_ind]


if __name__ == "__main__":
    np.random.seed(0)
    num_faces = 800

    data = np.random.randn(num_faces, 3, 3).astype(np.float32)
    noise = np.random.randn(num_faces, 3, 3).astype(np.float32)

    aligned = optimal_sum_numpy(data, noise, optimal=True)
    assert aligned.shape == noise.shape

    error_before = np.mean((data - noise) ** 2)
    error_after = np.mean((data - aligned) ** 2)
    print(f"ot_utils smoke test passed | mse_before={error_before:.6f} | mse_after={error_after:.6f}")
