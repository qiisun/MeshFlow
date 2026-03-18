"""
From https://github.com/stevenygd/PointFlow/tree/master/metrics
"""
import argparse
import json
import os
import warnings

import numpy as np
import torch
import trimesh
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from tqdm.auto import tqdm

try:
    from chamferdist import ChamferDistance
except ImportError:
    ChamferDistance = None


DEFAULT_CATEGORY_MAP = {
    '02691156': 'airplane',
    '02808440': 'bathtub',
    '02933112': 'cabinet',
    '03211117': 'display',
    '02876657': 'bottle',
    '03691459': 'loudspeaker',
}

DEFAULT_MESHXL_PREFIX_MAP = {
    '02691156': '0_',
    '02808440': '1_',
    '02876657': '2_',
    '02933112': '3_',
    '03211117': '4_',
    '03691459': '5_',
}

def normalize_mesh(vertices, bound=0.95):
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    ori_center = (vmax + vmin) / 2
    ori_scale = 2 * bound / np.max(vmax - vmin)
    vertices = (vertices - ori_center) * ori_scale
    return vertices


_EMD_NOT_IMPL_WARNED = False
def emd_approx(sample, ref):
    global _EMD_NOT_IMPL_WARNED
    emd = torch.zeros([sample.size(0)]).to(sample)
    if not _EMD_NOT_IMPL_WARNED:
        _EMD_NOT_IMPL_WARNED = True
        print('\n\n[WARNING]')
        print('  * EMD is not implemented due to GPU compatability issue.')
        print('  * We will set all EMD to zero by default.')
        print('  * You may implement your own EMD in the function `emd_approx` in ./evaluation/evaluation_metrics.py')
        print('\n')
    return emd



# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


def EMD_CD(sample_pcs, ref_pcs, batch_size, reduced=True, accelerated_cd=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    cd_lst = []
    emd_lst = []
    iterator = range(0, N_sample, batch_size)
    chamfer_dist = ChamferDistance() if accelerated_cd and ChamferDistance is not None else None

    for b_start in tqdm(iterator, desc='EMD-CD'):
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        if chamfer_dist is not None:
            dl, dr = chamfer_dist(sample_batch, ref_batch)
        else:
            dl, dr = distChamfer(sample_batch, ref_batch)

        cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))

        emd_batch = emd_approx(sample_batch, ref_batch)
        emd_lst.append(emd_batch)

    if reduced:
        cd = torch.cat(cd_lst).mean()
        emd = torch.cat(emd_lst).mean()
    else:
        cd = torch.cat(cd_lst)
        emd = torch.cat(emd_lst)

    results = {
        'MMD-CD': cd,
        'MMD-EMD': emd,
    }
    return results


def _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size, verbose=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    iterator = range(N_sample)
    if verbose:
        iterator = tqdm(iterator, desc='Pairwise EMD-CD')
    for sample_b_start in iterator:
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        sub_iterator = range(0, N_ref, batch_size)
        # if verbose:
        #     sub_iterator = tqdm(sub_iterator, leave=False)
        for ref_b_start in sub_iterator:
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            point_dim = ref_batch.size(2)
            sample_batch_exp = sample_batch.view(1, -1, point_dim).expand(
                batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            dl, dr = distChamfer(sample_batch_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))

            emd_batch = emd_approx(sample_batch_exp, ref_batch)
            emd_lst.append(emd_batch.view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=1)
        emd_lst = torch.cat(emd_lst, dim=1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_cd, all_emd


# Adapted from https://github.com/xuqiantong/
# GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat([
        torch.cat((Mxx, Mxy), 1),
        torch.cat((Mxy.transpose(0, 1), Myy), 1)], 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(
        k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s


def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }


def lgan_mmd_cov_match(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }, min_idx.view(-1)


def compute_all_metrics(sample_pcs, ref_pcs, batch_size):
    results = {}

    print("Pairwise EMD CD")
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(ref_pcs, sample_pcs, batch_size)

    ## CD
    res_cd = lgan_mmd_cov(M_rs_cd.t())
    results.update({
        "%s-CD" % k: v for k, v in res_cd.items()
    })
    
    ## EMD
    # res_emd = lgan_mmd_cov(M_rs_emd.t())
    # results.update({
    #     "%s-EMD" % k: v for k, v in res_emd.items()
    # })

    for k, v in results.items():
        print('[%s] %.8f' % (k, v.item()))

    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(ref_pcs, ref_pcs, batch_size)
    M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(sample_pcs, sample_pcs, batch_size)

    # 1-NN results
    ## CD
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results.update({
        "1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if 'acc' in k
    })
    ## EMD
    # one_nn_emd_res = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
    # results.update({
    #     "1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if 'acc' in k
    # })

    return results


#######################################################
# JSD : from https://github.com/optas/latent_3d_points
#######################################################
def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with
    resolution^3 cells, that is placed in the unit-cube. If clip_sphere it True
    it drops the "corner" cells that lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def jsd_between_point_cloud_sets(
        sample_pcs, ref_pcs, resolution=28):
    """Computes the JSD between two sets of point-clouds,
       as introduced in the paper
    ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    """
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(
        sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(
        ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(
        pclouds, grid_resolution, in_sphere=False, verbose=False):
    """Given a collection of point-clouds, estimate the entropy of
    the random variables corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in tqdm(pclouds, desc='JSD'):
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    """another way of computing JSD"""

    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))



def test_simple():
    a = torch.rand([128, 2048, 3]).cuda()
    b = torch.rand([128, 2048, 3]).cuda()
    
    # print(compute_all_metrics(a, b, batch_size=128))
    a = a.cpu().numpy()
    b = b.cpu().numpy()
    jsd = jsd_between_point_cloud_sets(a, b)
    print(jsd)

def sample_point_cloud(meshes, num_points=2048):
    ps = []
    for m in meshes:
        m.vertices = normalize_mesh(m.vertices, bound=1) + 0.5
        point = m.sample(num_points)
        ps.append(point)
    ps = np.stack(ps)
    return ps

def list_obj_paths(root_dir):
    obj_paths = []
    for current_root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.obj'):
                obj_paths.append(os.path.join(current_root, filename))
    return sorted(obj_paths)


def load_meshes_from_paths(obj_paths):
    meshes = []
    for obj_path in tqdm(obj_paths, desc='Loading generated meshes'):
        mesh = trimesh.load_mesh(obj_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        meshes.append(mesh)
    return meshes


def load_generated_meshes(gen_root, max_meshes=None, prefix=None):
    obj_paths = list_obj_paths(gen_root)
    if prefix is not None:
        normalized_root = os.path.abspath(gen_root)
        filtered_paths = []
        for obj_path in obj_paths:
            rel_path = os.path.relpath(obj_path, normalized_root)
            top_level = rel_path.split(os.sep, 1)[0]
            if top_level.startswith(prefix):
                filtered_paths.append(obj_path)
        obj_paths = filtered_paths

    if not obj_paths:
        raise FileNotFoundError(
            f'No .obj files found under {gen_root}' if prefix is None else f'No .obj files found under {gen_root} for prefix {prefix}'
        )

    if max_meshes is not None:
        obj_paths = obj_paths[:max_meshes]

    meshes = load_meshes_from_paths(obj_paths)
    return meshes, obj_paths


def load_gt_meshes(data_root, synset_id, max_meshes=None, max_face_length=800):
    dataset_dir = os.path.join(data_root, f'shapenet-{synset_id}')
    split_path = os.path.join(dataset_dir, 'split', 'test.npz')
    data_dir = os.path.join(dataset_dir, 'objaverse_occ_v5_ids')

    if not os.path.exists(split_path):
        raise FileNotFoundError(f'Test split file not found: {split_path}')
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f'Dataset directory not found: {data_dir}')

    split = np.load(split_path, allow_pickle=True)['npz_list'].tolist()
    gt_meshes = []
    gt_uids = []

    for item in split:
        uid = item['uid']
        mesh_path = os.path.join(data_dir, f'{uid}.npz')
        if not os.path.exists(mesh_path):
            continue

        try:
            loaded = np.load(mesh_path, allow_pickle=True)
            num_faces = int(loaded['faces_num'])
            if num_faces <= 20 or num_faces >= max_face_length:
                continue

            gt_mesh = trimesh.Trimesh(
                vertices=loaded['vertices'].astype(np.float32),
                faces=loaded['faces'].astype(np.int64),
                process=False,
            )
            gt_meshes.append(gt_mesh)
            gt_uids.append(uid)
            if max_meshes is not None and len(gt_meshes) >= max_meshes:
                break
        except Exception as exc:
            print(f'[WARN] Failed to load GT mesh {mesh_path}: {exc}')

    if not gt_meshes:
        raise RuntimeError(f'No valid GT meshes found for {synset_id} under {dataset_dir}')

    return gt_meshes, gt_uids, dataset_dir


def evaluate_mesh_sets(gen_meshes, gt_meshes, batch_size=128, num_points=2048, device=None):
    if len(gen_meshes) != len(gt_meshes):
        target_count = min(len(gen_meshes), len(gt_meshes))
        print(f'[INFO] Aligning mesh counts to {target_count} (gen={len(gen_meshes)}, gt={len(gt_meshes)})')
        gen_meshes = gen_meshes[:target_count]
        gt_meshes = gt_meshes[:target_count]

    ps_gen = sample_point_cloud(gen_meshes, num_points=num_points)
    ps_gt = sample_point_cloud(gt_meshes, num_points=num_points)
    jsd = float(jsd_between_point_cloud_sets(ps_gen, ps_gt))

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ps_gen_t = torch.from_numpy(ps_gen).float().to(device)
    ps_gt_t = torch.from_numpy(ps_gt).float().to(device)
    metric_tensors = compute_all_metrics(ps_gen_t, ps_gt_t, batch_size=batch_size)

    metrics = {
        'JSD': jsd,
        'num_meshes': len(gen_meshes),
    }
    for key, value in metric_tensors.items():
        metrics[key] = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate generated meshes against ShapeNet test splits.')
    parser.add_argument(
        '--gen-root',
        type=str,
        default='output/meshxl',
        help='Directory containing generated .obj files. Recursively searched.',
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='downloaded_data',
        help='Root directory that contains shapenet-<synset> folders.',
    )
    parser.add_argument(
        '--categories',
        nargs='+',
        default=list(DEFAULT_CATEGORY_MAP.keys()),
        help='ShapeNet synset ids to evaluate.',
    )
    parser.add_argument(
        '--prefix-map',
        type=str,
        default=None,
        help='Optional JSON string mapping synset ids to MeshXL directory prefixes, e.g. {"02691156":"0_"}.',
    )
    parser.add_argument('--max-gen-meshes', type=int, default=None)
    parser.add_argument('--max-face-length', type=int, default=800)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-points', type=int, default=2048)
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Optional summary json path. Defaults to <gen-root>/point_eval_summary.json',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prefix_map = dict(DEFAULT_MESHXL_PREFIX_MAP)
    if args.prefix_map is not None:
        prefix_map.update(json.loads(args.prefix_map))

    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for synset_id in args.categories:
        category_name = DEFAULT_CATEGORY_MAP.get(synset_id, synset_id)
        print(f'\n[INFO] Evaluating {category_name} ({synset_id})')
        prefix = prefix_map.get(synset_id)
        if prefix is None:
            print(f'[WARN] No MeshXL prefix mapping found for {synset_id}, skipping')
            continue

        try:
            gen_meshes, gen_paths = load_generated_meshes(
                args.gen_root,
                max_meshes=args.max_gen_meshes,
                prefix=prefix,
            )
        except FileNotFoundError as exc:
            print(f'[WARN] {exc}; skipping {synset_id}')
            continue

        print(f'[INFO] Loaded {len(gen_meshes)} generated meshes from {args.gen_root} for prefix {prefix}')
        gt_meshes, gt_uids, dataset_dir = load_gt_meshes(
            args.data_root,
            synset_id,
            max_meshes=len(gen_meshes),
            max_face_length=args.max_face_length,
        )
        metrics = evaluate_mesh_sets(
            gen_meshes,
            gt_meshes,
            batch_size=args.batch_size,
            num_points=args.num_points,
            device=device,
        )
        metrics.update({
            'category_name': category_name,
            'synset_id': synset_id,
            'meshxl_prefix': prefix,
            'dataset_dir': dataset_dir,
            'generated_root': args.gen_root,
            'generated_count': len(gen_meshes),
            'gt_count': len(gt_meshes),
            'gt_uids_preview': gt_uids[:5],
            'generated_preview': gen_paths[:5],
        })
        results[synset_id] = metrics

        per_category_json = os.path.join(args.gen_root, f'point_eval_{synset_id}.json')
        with open(per_category_json, 'w') as handle:
            json.dump(metrics, handle, indent=2)
        print(f'[INFO] Saved metrics to {per_category_json}')
        for key in sorted(metrics.keys()):
            if isinstance(metrics[key], float):
                print(f'  {key}: {metrics[key]:.8f}')

    output_json = args.output_json or os.path.join(args.gen_root, 'point_eval_summary.json')
    with open(output_json, 'w') as handle:
        json.dump(results, handle, indent=2)
    print(f'\n[INFO] Saved summary to {output_json}')


if __name__ == '__main__':
    main()

