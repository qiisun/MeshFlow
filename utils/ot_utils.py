import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from tqdm import trange
from time import time 
from joblib import Parallel, delayed
from einops import rearrange
import itertools
from scipy.spatial.distance import cdist
import sys
sys.path.append(".")
# from core.util.sinkhorn import sinkhorn
# from torch_linear_assignment import batch_linear_assignment, assignment_to_indices

b, n = 512, 500

def optimal_sum_scipy(data, noise, length=None, mp=True, optimal=False, dimension=3):
    batch_size, L, _ = data.shape

    if optimal:
        noise = noise.reshape(batch_size, L, 3, dimension) # create 6 perm & flatten
        random_perm = torch.Tensor(list(itertools.permutations([0, 1, 2]))).to(torch.int32) # [6, 3]
        
        cost_matrices = []
        for perm in random_perm:
            noise_perm = noise[:, :, perm].flatten(2,3)
            cost_matrix = torch.cdist(data, noise_perm, p=2)  # [b, n, n]
            cost_matrices.append(cost_matrix) # [b, n, n]
        cost_matrices = torch.stack(cost_matrices) # [6, b, n, n]
        cost_matrix, indices = torch.min(cost_matrices, dim=0) # [b, n, n]
        
        noise = noise.reshape(batch_size, L, 3*dimension)
        del cost_matrices
    else:
        cost_matrix = torch.cdist(data, noise, p=2)
        
    cost_matrix_np = cost_matrix.cpu().numpy()  
    del cost_matrix
    if length == None:
        length = np.array([cost_matrix_np.shape[1]] * batch_size)
    else:
        length = length.cpu().numpy() 

    def process_single(i):
        """处理单个成本矩阵"""
        row_ind, col_ind = linear_sum_assignment(cost_matrix_np[i][:length[i], :length[i]])
        return col_ind

    if mp:
        col_inds = Parallel(n_jobs=-1)(
            delayed(process_single)(i) for i in range(batch_size)
        )
    else:
        col_inds = []
        for i in range(batch_size):
            col_inds.append(process_single(i))

    noise_np = noise.cpu().numpy() # [b, n, 9]
    if optimal:
        indices = indices.cpu().numpy() # [b, n, n]
        random_perm = random_perm.cpu().numpy() # [6, 3]
    for i in range(batch_size):
        current_noise = noise_np[i][:length[i]] # [n0, 9]
        current_noise = current_noise[col_inds[i]]  # [n0, 9] 交换顺序
        if optimal:
            perm_idx = indices[i][:length[i], :length[i]][np.arange(length[i]), col_inds[i]] # [n0]
            perm = random_perm[perm_idx] # [n0, 3]
            current_noise = current_noise.reshape(-1, 3, dimension)[np.arange(length[i])[:, None], perm].reshape(-1, 3*dimension) # [n0, 9]     
        noise_np[i][:length[i]] = current_noise
    return torch.from_numpy(noise_np).to(data.device)

def optimal_sum_numpy(data, noise, optimal=False, dimension=3, flip_face=True, sinhorn=False):
    # data: [N, 3, 3]
    # noise: [N, 3, 3]
    N = data.shape[0]
    data_flat = data.reshape(N, -1)

    if optimal:
        if flip_face:
            random_perm = np.array(
                list(itertools.permutations([0, 1, 2]))).astype(np.int32)
        else:
            random_perm = np.array([[0,1,2], [1,2,0], [2,0,1]]).astype(np.int32)
        cost_matrices = []
        for perm in random_perm:
            noise_perm = noise[:, perm].reshape(N, -1) # [N, 9]
            cost_matrix = cdist(data_flat, noise_perm)  # [N, N]
            cost_matrices.append(cost_matrix)
        cost_matrices = np.stack(cost_matrices) # [6, N, N]
        cost_matrix = np.min(cost_matrices, axis=0) # [N, N]
        best_perm_indices = np.argmin(cost_matrices, axis=0)
        if not sinhorn:
            row_ind, col_ind = linear_sum_assignment(cost_matrix) 
        else:
            row_ind, col_ind = sinkhorn(M_ij=cost_matrix, eps=1e-3, max_iters=10, stop_thresh=1e-5)
        best_perms_for_matches = random_perm[best_perm_indices[row_ind, col_ind]] # [N, 3]
        noise_reordered = noise[col_ind] # [N, 3, 3] 
        n_idx = np.arange(N)[:, None]
        noise_reordered = noise_reordered[n_idx, best_perms_for_matches, :] # [N, 3, 3]
        return noise_reordered
    else:
        cost_matrix = cdist(data.reshape(N, -1), noise.reshape(N, -1)) # [N, N]
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        noise_reordered = noise[col_ind]
        return noise_reordered

def optimal_sinkhorn(data, noise, dimension=3, eps=1e-2, max_iter=10):
    # data: [N, 3, 3]
    # noise: [N, 3, 3]
    data = torch.from_numpy(data)
    noise = torch.from_numpy(noise)
    N = data.shape[0]
    
    # get face correpondence with approximated OT
    data_ct = data.mean(1) # [N, 3]
    noise_ct = noise.mean(1) # [N, 3]
    d, corrs_x_to_y, corr_y_to_x = sinkhorn(data_ct, noise_ct, max_iters=max_iter, eps=eps)
    noise_reordered = noise[corrs_x_to_y] # [N, 3, 3]
    
    # using batch OT to exchange the three vertices
    cost_batch = torch.cdist(data, noise_reordered) # [N, 3, 3] x [N, 3, 3] -> [N, 3, 3]
    assignment = batch_linear_assignment(cost_batch)
    _, col_ind = assignment_to_indices(assignment) # [N, 3]
    n_idx = np.arange(N)[:, None]
    noise_reordered = noise_reordered[n_idx, col_ind.cpu(), :]
    return noise_reordered.numpy()

if __name__ == "__main__":
    data = torch.randn(b, n, 9).to('cuda')
    noise = torch.randn(b, n, 9).to('cuda')
    length = torch.randint(0, n, (b,)).to('cuda')
    
    # data = torch.randn(b*n, 3, 3).to('cuda')
    # noise = torch.randn(b*n, 3, 3).to('cuda')
    # length = None

    import time
    # start = time.time()
    # noise_rearange = optimal_sum_scipy(data, noise, length, mp=True, optimal=False)
    # print(time.time()-start)
    # for length in [512, 1024, 2048, 4096, 8192, 16384]:
    #     print(f"length: {length}")
    #     data = np.random.randn(length, 3, 3).astype(np.float32)
    #     noise = np.random.randn(length, 3, 3).astype(np.float32)
    #     start = time.time()
    #     noise_rearange = optimal_sum_numpy(data, noise, optimal=False)
    #     print(time.time()-start)
    # data = np.random.randn(512, 3, 3).astype(np.float32)
    # noise = np.random.randn(512, 3, 3).astype(np.float32)
    # import time
    # start = time.time()
    # optimal_sum_numpy(data, noise, optimal=True)
    # print(time.time()-start)
    
    
    # def test():
    #     data = torch.randn(1, 1, 9).to('cuda')
    #     noise = data.reshape(1, 1, 3, 3)[:, :, [2, 0, 1]].reshape(1, 1, 9)
    #     length = torch.tensor([1]).to('cuda')
    #     noise_rearange = optimal_sum_scipy(data, noise, length, mp=False, optimal=False)
    #     print(noise_rearange == data)
    # test()
            
    # import torch
    import time
    import matplotlib.pyplot as plt
    # from torch_linear_assignment import batch_linear_assignment, assignment_to_indices

    # cost = torch.randn(1024, 3, 3).cuda()

    # assignment = batch_linear_assignment(cost)
    # print(assignment)
    # row_ind, col_ind = assignment_to_indices(assignment)
    # print(row_ind)
    # print(col_ind)
    time_sinkhorn, time_ot = [], []
    error_origin, error_sinkhorn, error_ot = [], [],  []
    lengths = [ 1024]
    iters = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    length = lengths[0]
    
    # for length in lengths:
    for num_iter in iters:
        print(length)
        print("*"*10)
        data = np.random.randn(length, 3, 3).astype(np.float32)
        noise = np.random.randn(length, 3, 3).astype(np.float32)
        error0 = np.mean((data-noise)**2)
        print(f"data error {error0:.5f}")
        # sinkhorn OT
        start = time.time()
        noise_rearange = optimal_sinkhorn(data, noise, max_iter=num_iter)
        
        t_sinkhorn= (time.time()-start)
        error1 = np.mean((data-noise_rearange)**2)
        print(t_sinkhorn)
        print(f"error: {error1:.5f}")
        
        # exact OT
        start = time.time()
        noise_rearange = optimal_sum_numpy(data, noise, optimal=True)
        t_ot = (time.time()-start)
        error2 = np.mean((data-noise_rearange)**2)
        print(t_ot)
        print(f"error: {error2:.5f}")
        
        time_sinkhorn.append(t_sinkhorn)
        time_ot.append(t_ot)
        error_origin.append(error0)
        error_sinkhorn.append(error1)
        error_ot.append(error2)

    plt.figure(figsize=(10, 6))
    # plt.plot(lengths, time_sinkhorn, color='red', label='Sinkhorn method', marker='o')
    # plt.plot(lengths, time_ot, color='blue', label='OT method', marker='o')
    plt.scatter(iters, time_sinkhorn, color='red', label='Sinkhorn method')
    # plt.scatter(lengths, time_ot, color='blue', label='OT method')
    plt.xlabel('#iterations')
    plt.ylabel('Time (seconds)')
    # plt.title('Comparison of Sinkhorn and OT Methods')
    plt.legend()
    # plt.xscale("log")
    # plt.yscale("log")
    plt.grid(True)
    plt.savefig('sinkhorn_ot_comparison.png')
    
    plt.figure(figsize=(10, 6))
    plt.plot(iters, error_sinkhorn, color='red', label='Sinkhorn method', marker='o')
    plt.plot(iters, error_ot, color='blue', label='OT method', marker='o')
    plt.plot(iters, error_origin, color='gray', label='origin', marker='o', linestyle='--')
    plt.xlabel('#iterations')
    plt.ylabel('Error')
    # plt.title('Comparison of Sinkhorn and OT Methods')
    plt.legend()
    # plt.xscale("log")
    plt.grid(True)
    plt.savefig('sinkhorn_ot_comparison2.png')
    




    # bs = 512, n = 500
    # 3.51s (w/o mp)
    # 2.79s
    
    # bs*n = 512*1000, 
    # 18.20s 
    # 1.90s (w/o mp)
    