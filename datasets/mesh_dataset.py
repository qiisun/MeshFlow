import os
import random
import trimesh
import numpy as np
import sys
sys.path.append('.')
from utils.ot_utils import optimal_sum_numpy
from models.equivae import float_to_index_np, index_to_float_np
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm


def _highlight(msg: str):
    # yellow highlight in terminal
    print(f"\033[1;33m[HIGHLIGHT]\033[0m {msg}")


def _green(msg: str):
    return f"\033[1;32m{msg}\033[0m"


def _red(msg: str):
    return f"\033[1;31m{msg}\033[0m"


def _cyan(msg: str):
    return f"\033[1;36m{msg}\033[0m"


def _status(flag: bool):
    return _green("ON") if flag else _red("OFF")


def generate_custom_prior(num_samples, var_scale=0.05):
    """
    return [N, 3, 3]   triangle soup
    """
    centriod = np.random.randn(num_samples, 1, 3) 
    
    # draw two vertices from smaller Gaussian
    offsets23 = np.random.randn(num_samples, 2, 3) * var_scale # [N, 2, 3]
    vertice23 = centriod + offsets23  # [N, 2, 3]
    offsets1 = -np.sum(offsets23, axis=1, keepdims=True) # [N, 1, 3]
    vertice1 =  offsets1 + centriod  # [N, 1, 3]
    return np.concatenate([vertice23, vertice1], axis=1)  # [N, 3, 3]


def save_mesh(tokens: np.ndarray, path: str, clean: bool = True, num_bins=2048, max_val=0.95):
    # [N, 3, 3] -> mesh
    def simple_detokenize_mesh(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coords = tokens.reshape(-1, 3).astype(np.float32)
        vertices = coords

        faces = np.arange(len(vertices)).reshape(-1, 3)
        # vertices = vertices[:, [2, 1, 0]]  # zyx to xyz

        return vertices, faces

    vertices, faces = simple_detokenize_mesh(tokens=tokens)
    vertices = float_to_index_np(vertices, min_val=-max_val, max_val=max_val, num_bins=num_bins)
    vertices = index_to_float_np(vertices, min_val=-max_val, max_val=max_val, num_bins=num_bins)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    if clean:
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.fix_normals()
    mesh.export(path)


def sort_triangle_soup(triangle_soup: np.ndarray):
    """
    sorts a triangle soup. tiangle_soup [N, 3, 3]
    """
    # Cast into shape that we can work on.
    N = triangle_soup.shape[0]
    orig_shape = triangle_soup.shape
    triangle_soup = triangle_soup.reshape(N, 3, 3)
    
    # sort each face's vertices
    triangle_soup_w_face_id = np.concatenate(
        (triangle_soup[..., ::-1], np.arange(N).reshape(-1, 1, 1).repeat(3, axis=1)), 
        axis=2) # [N, 3, 4]
    triangle_soup_min_v_id = np.lexsort(triangle_soup_w_face_id.transpose(2, 0, 1).reshape(4, -1), axis=-1).reshape(N, 3)[:, 0] # [N]
    triangle_soup_min_v_id -= np.arange(N) * 3 # [N]
    triangle_soup_v_sorted_id = np.stack([triangle_soup_min_v_id, triangle_soup_min_v_id+1, triangle_soup_min_v_id+2], axis=1) % 3 # [N, 3]
    v_sorted_triangle_soup = np.take_along_axis(triangle_soup, triangle_soup_v_sorted_id[..., None], axis=1) # [N, 3, 3]   
    
    # sort face's based on vertices 
    triangle_soup_sorted_id = np.lexsort(v_sorted_triangle_soup.transpose(1, 2, 0).reshape(9, N)[::-1], axis=-1) # [N]
    triangle_soup_sorted = v_sorted_triangle_soup[triangle_soup_sorted_id] # [N, 3, 3]

    # Back to original shape
    triangle_soup_sorted = triangle_soup_sorted.reshape(*orig_shape)
    return triangle_soup_sorted


def rotate_mesh_with_normal(vertices):
    angle_options = np.array([0, np.pi/2, np.pi, 3*np.pi/2])  # 0, 90, 180, 270 degrees
    angle_y = np.random.choice(angle_options)

    R_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)]])

    rotated_vertices = np.dot(vertices, R_y.T)
    return rotated_vertices


class ObjaverseDataset(Dataset):
    def __init__(self, data_pth, 
                noise_sort='ot',
                training=True,
                do_dataset_normalize=True,
                use_custom_prior=False,
                use_decimated_dataset=False,
                max_face_length=800, 
                vae=False,
                overfit=False,
                use_rot_aug=False,
                use_scale_aug=False,
                use_repa=False,
                use_permut_aug=True):
        
        self.vae = vae
        self.training = training
        self.use_custom_prior = use_custom_prior
        self.use_decimated_dataset = use_decimated_dataset
        self.do_dataset_normalize = do_dataset_normalize
        self.max_face_length = max_face_length
        self.use_rot_aug = use_rot_aug 
        self.use_scale_aug = use_scale_aug 
        self.noise_sort = noise_sort 
        self.use_repa = use_repa
        self.use_permut_aug = use_permut_aug

        if data_pth == 'all':
            self.dataset_paths = [
                "downloaded_data/objaverse",
                "downloaded_data/shapenet-new"
            ]
            print(f'[INFO] Joint Training Mode: Loading Objaverse + ShapeNet')
        else:
            self.dataset_paths = [data_pth]
            print(f'[INFO] Single Dataset Mode: Loading {data_pth}')
        raw_data_list = []
        
        for current_data_pth in self.dataset_paths:
            print(f'--- Processing dataset: {current_data_pth} ---')
            split_filename = "train.npz" if self.training else "test.npz"
            split_path = os.path.join(current_data_pth, 'split', split_filename)
            
            if not os.path.exists(split_path):
                print(f"[WARNING] Split file not found: {split_path}, skipping...")
                continue

            split = np.load(split_path, allow_pickle=True)['npz_list'].tolist()
            split_uids = set([item['uid'] for item in split])
            print(f"Loaded split {split_filename}, count: {len(split_uids)}")

            folder_prefix = "objaverse_occ_v5_ids" # _filter_128"

            if self.training and self.use_decimated_dataset:
                data_subfolder = f"{folder_prefix}_decimated"
            else:
                data_subfolder = folder_prefix
            
            data_path = os.path.join(current_data_pth, data_subfolder)
            
            if not os.path.exists(data_path):
                print(f"[WARNING] Predicted folder {data_path} not found.") 
                continue

            print(f'[INFO] Loading meshes from {data_path}...')
            
            all_files = os.listdir(data_path)
            current_dataset_count = 0
            
            for mesh_file in tqdm.tqdm(all_files, desc=f"Loading {os.path.basename(current_data_pth)}"):
                if mesh_file.endswith('.npz'):
                    uid = mesh_file.split(".")[0]
                    if uid in split_uids: 
                        mesh_path = os.path.join(data_path, mesh_file)
                        try:
                            loaded_data = np.load(mesh_path, allow_pickle=True)
                            if loaded_data['faces_num']: 
                                raw_data_list.append(loaded_data)
                                current_dataset_count += 1
                        except Exception as e:
                            print(f"Error loading {mesh_path}: {e}")
            
            print(f"Loaded {current_dataset_count} valid samples from {current_data_pth}")

        self.data = []
        print(f"[INFO] Filtering data with max_face_length={max_face_length}...")
        
        for cur_data in raw_data_list: 
            num_faces = cur_data['faces_num']
            if num_faces > 20 and num_faces < max_face_length:
                self.data.append(cur_data)
        
        if self.training and overfit:
            from termcolor import colored
            print(colored("[INFO] Overfit mode enabled: Duplicating dataset.", "yellow"))
            # self.data = self.data[:1] * 1000  # Overfit mode for debugging

        print(f"Total Combined Dataset Size: {len(self.data)}")
        
        if do_dataset_normalize:
            self.std = 0.3

        # one-time config logs for debugging training behavior
        _highlight(f"{_cyan('noise_sort')} = {self.noise_sort} | {_cyan('OT')} = {_status(self.noise_sort == 'ot')}")
        _highlight(f"{_cyan('dataset_normalize')} = {_status(self.do_dataset_normalize)} | std = {getattr(self, 'std', 'N/A')}")
        _highlight(f"{_cyan('final token scale')} = coords * (0.95 / 0.5) = 1.9")

        # shuffle & augmentation behavior
        _highlight(f"{_cyan('raw mesh canonical sort (vertices/faces)')} = {_status(True)}")
        _highlight(f"{_cyan('triangle/face shuffle')} = {_status(self.use_permut_aug)}")
        _highlight(f"{_cyan('triangle vertex-order shuffle')} = {_status(self.use_permut_aug)}")
        _highlight(f"{_cyan('rotation aug')} = {_status(self.use_rot_aug)}")
        if self.use_scale_aug:
            _highlight(f"{_cyan('scale aug')} = {_status(True)} | range = [0.75, 1.25] per-axis")
        else:
            _highlight(f"{_cyan('scale aug')} = {_status(False)}")
            
    def __len__(self):
        return len(self.data)
    
    def save_mesh(self, dir, name, vertices,faces):
        cur_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        cur_mesh.export(f"{dir}/{name}.ply")
    
    def save_data(self, dir, name, data):
        vertices, faces = data["vertices"], data["faces"]
        cur_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        cur_mesh.export(f"{dir}/{name}.ply")

    def sort_vertices_and_faces(self, vertices_, faces_, num_tokens=None):
        assert (vertices_ <= 0.5).all() and (vertices_ >= -0.5).all() # [-0.5, 0.5]
        
        if num_tokens is not None:
            vertices = (vertices_+0.5) * num_tokens # [0, num_tokens]
            vertices -= 0.5 # for evenly distributed, [-0.5, num_tokens -0.5] will be round to 0 or num_tokens (-1)
            vertices_quantized_ = np.clip(vertices.round(), 0, num_tokens-1).astype(int)  # [0, num_tokens -1]
            origin_face_num = len(faces_)

            cur_mesh = trimesh.Trimesh(vertices=vertices_quantized_, faces=faces_)

            cur_mesh.merge_vertices()
            cur_mesh.update_faces(cur_mesh.nondegenerate_faces())
            cur_mesh.update_faces(cur_mesh.unique_faces())
            cur_mesh.remove_unreferenced_vertices()
            vertices_ = cur_mesh.vertices
            faces_ = cur_mesh.faces

        sort_inds = np.lexsort((vertices_.T[0,:], vertices_.T[2,:], vertices_.T[1,:])) # 按照yzx排序
        
        vertices = vertices_[sort_inds]
        faces = [np.argsort(sort_inds)[f] for f in faces_]

        faces = [sorted(sub_arr) for sub_arr in faces]
        def sort_faces(face):
            return face[0], face[1], face[2]
        
        faces = sorted(faces, key=sort_faces)
        
        if num_tokens is not None:
            vertices = vertices / num_tokens - 0.5  # [0, num_tokens -1] to [-0.5, 0.5)  for computing
        return vertices, faces

    def sample_noise(self, triangle_soup):
        if not self.use_custom_prior:
            noise = np.random.randn(*triangle_soup.shape)
        else:
            noise = generate_custom_prior(triangle_soup.shape[0])
        if self.noise_sort == 'random':
            return triangle_soup, noise
        elif self.noise_sort == 'sort':
            return sort_triangle_soup(triangle_soup), sort_triangle_soup(noise)
        elif self.noise_sort == 'ot':
            noise = optimal_sum_numpy(triangle_soup, noise, optimal=True)
            return triangle_soup, noise
        else:
            raise ValueError(f"Invalid noise sort: {self.noise_sort}")
    
    def tokenize_mesh(self, vertices, faces, shuffle_face=True, shuffle_vertex=True, discrete_bins=None):
        triangle_soup = vertices[faces]  # [N, 3, 3]
        triangle_soup = sort_triangle_soup(triangle_soup) # [N, 3, 3]
        N = triangle_soup.shape[0]
        if shuffle_face:
            perm_idx = np.random.permutation(N)
            triangle_soup = triangle_soup[perm_idx]
        else:
            perm_idx = np.arange(N)
        if shuffle_vertex:
            all_perm = np.array([[0,1,2], [1,2,0], [2,0,1]])
            perm = np.random.randint(0,3, size=(triangle_soup.shape[0],))
            triangle_soup = triangle_soup[np.arange(N)[:, None], all_perm[perm]]
        
        if discrete_bins is not None:
            triangle_soup = np.round(triangle_soup * discrete_bins) / discrete_bins
        return triangle_soup, perm_idx # [N, 3, 3]
    
    def __getitem__(self, idx):
        data = self.data[idx]
        vertices = data['vertices']
        faces = data['faces']
        faces_num = data['faces_num']
        
        if self.use_repa:
            f_feature = data['f_feature']  # [f, feat_dim]
            assert len(f_feature) == len(faces)
        assert vertices.shape[1] == 3 and faces.shape[1] == 3
        data_dict = {}
        if 'category' in data:
            data_dict['uid'] = data['category']+'_'+data['uid']
        else:
            data_dict['uid'] = data['uid']
        # scale x, y, z
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
        vertices = vertices / (bounds[1] - bounds[0]).max()
        # aligned from now on
        if self.use_scale_aug:
            x_lims = (0.75, 1.25)
            y_lims = (0.75, 1.25)
            z_lims = (0.75, 1.25)
            x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
            y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
            z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
            vertices = np.stack([vertices[:, 0] * x, vertices[:, 1] * y, vertices[:, 2] * z], axis=-1)

        # normalize x, y, z
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
        if self.use_rot_aug:
            vertices = rotate_mesh_with_normal(vertices)

        # scale to -0.5 to 0.5
        bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
        vertices = vertices / (bounds[1] - bounds[0]).max()
        vertices = vertices.clip(-0.5, 0.5)
        assert vertices.min() >= -0.5 and vertices.max() <= 0.5
        vertices, faces = self.sort_vertices_and_faces(vertices, faces)
        if vertices is None:
            # sample another data
            return self.__getitem__(np.random.randint(0, len(self.data)))
        coords, perm_idx = self.tokenize_mesh(vertices, faces, shuffle_face=self.use_permut_aug, shuffle_vertex=self.use_permut_aug) # [N, 3, 3] FIXME: shuffle faces/vertices
        if self.do_dataset_normalize and (not self.vae):
            coords = coords / self.std
        
        if not self.vae:    
            coords, noise = self.sample_noise(coords) # [N, 3, 3], [N, 3, 3]
            data_dict['noise'] = noise.reshape(faces_num, -1)
            if self.use_repa:
                data_dict['f_feature'] = f_feature[perm_idx]
        # data_dict['face_area'] = compute_face_area_from_soup(coords) # [F]
        data_dict['coords'] = coords.reshape(faces_num, -1)  * 0.95/ 0.5
        data_dict['num_faces'] = faces_num
        data_dict['len'] = faces_num

        return data_dict
    
def collate_fn(batch, max_seq_length=800):
    num_faces = [item['num_faces'] for item in batch]

    max_len = max([item['len'] for item in batch])
    max_len = min(max_len, max_seq_length)
    
    tokens = []
    noises = []
    masks = []
    features = []
    face_areas = []
    
    input_c = batch[0]['coords'].shape[1] 
        
    for item in batch:
        if max_len >= item['len']:
            pad_len = max_len - item['len']
            
            tokens.append(np.concatenate([
                item['coords'], 
                np.full((pad_len, input_c), 0), 
            ], axis=0)) 
            
            if 'face_area' in item:
                face_areas.append(np.concatenate([
                    item['face_area'],
                    np.full((pad_len), 0), 
                ], axis=0))
            
            if "noise" in item.keys():
                noises.append(np.concatenate([
                    item['noise'], 
                    np.full((pad_len, input_c), 0), 
                ], axis=0)) 
            
            if "f_feature" in item.keys():
                features.append(np.concatenate([
                    item['f_feature'],
                    np.full((pad_len, item['f_feature'].shape[1]), 0),
                ], axis=0))
                
            masks.append(np.concatenate([
                np.ones(item['len']), 
                np.zeros(pad_len)
            ], axis=0))

        else:
            tokens.append(np.concatenate([
                item['coords'][:max_len], 
            ], axis=0))

            # if 'face_area' in item:
            #     face_areas.append(np.concatenate([
            #         item['face_area'][:max_len], 
            #     ], axis=0))

            masks.append(np.ones(max_len))

    results = {}
    results['num_faces'] = torch.from_numpy(np.stack(num_faces, axis=0)).long()
    results['tokens'] = torch.from_numpy(np.stack(tokens, axis=0)).float() 
    
    # if len(face_areas) > 0:
    #     results['face_area'] = torch.from_numpy(np.stack(face_areas, axis=0)).float() 
        
    if len(noises) > 0:
        results['noise'] = torch.from_numpy(np.stack(noises, axis=0)).float() 
    if len(features) > 0:
        results['f_feature'] = torch.from_numpy(np.stack(features, axis=0)).float() 
    results['masks'] = torch.from_numpy(np.stack(masks, axis=0)).bool()
    
    return results

if __name__ == "__main__":
    # test script: python core/provider_dit.py dit
    # import tyro
    # from core.options import AllConfigs
    from functools import partial
    # tokenizer

    dataset = ObjaverseDataset(data_pth='../downloaded_data', training=True)
    # dataset = MeshDataset(opt, training=True)
    print(len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=partial(collate_fn, max_seq_length=800),
    )
    save_dir = 'debug/tmp_train_decimated'
    os.makedirs(save_dir, exist_ok=True)
    
    for i, results in enumerate(dataloader):
        if i >= 10:
            break
        # restore mesh
        for b in range(len(results['masks'])):
            masks = results['masks'][b].numpy()
            tokens = results['tokens'][b].numpy()[masks] # [M, 9]
            noise = results['noise'][b].numpy()[masks]
            
            # write obj using the original order to check face orientation
            vertices, faces = dataset.detokenize_mesh(tokens, discrete_bins=None, tokenizer=None)
            with open(f'{save_dir}/{i}_{b}.obj', 'w') as f:
                for v in vertices:
                    f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
                for face in faces:
                    f.write(f'f {" ".join([str(v+1) for v in face])}\n')

            print(f'[INFO] tokens: {tokens.shape[0]}, faces: {faces.shape[0]}, ratio={100 * tokens.shape[0] / (9 * faces.shape[0]):.2f}%')
