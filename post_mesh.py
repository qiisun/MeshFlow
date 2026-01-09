import torch
import os
import argparse
import yaml
import numpy as np
import trimesh
from tqdm import tqdm
from glob import glob

from models.equivae import AutoencoderKL, float_to_index, index_to_float
from datasets.mesh_dataset import save_mesh

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def preprocess_mesh(mesh_path, max_seq_length=800):
    mesh = trimesh.load(mesh_path, process=False)
    vertices = mesh.vertices # [-0.95, 0.95]
    # print(vertices.min(), vertices.max())
    faces = mesh.faces
    face_vertices = vertices[faces].reshape(-1, 3) 
    current_faces = min(len(faces), max_seq_length)
    valid_vertex_count = current_faces * 3
    tokens = torch.tensor(face_vertices[:valid_vertex_count], dtype=torch.float32)
    tokens_reshaped = tokens.reshape(-1, 9) 
    padded_tokens = torch.full((max_seq_length, 9), 0.0) # Float padding
    mask = torch.zeros(max_seq_length, dtype=torch.bool)
    
    padded_tokens[:current_faces] = tokens_reshaped[:current_faces]
    mask[:current_faces] = True
    
    return {
        'tokens': padded_tokens.unsqueeze(0), # [1, N, 9]
        'masks': mask.unsqueeze(0),           # [1, N]
        'num_faces': torch.tensor([current_faces]).long(),
        'filename': os.path.basename(mesh_path)
    }
    
def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(args.config)
    
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Initializing Model...")
    model = AutoencoderKL(latent_channels=config['model']['latent_channels'],
                          decoder_type=config['model']['decoder_type'],
                          num_bins=config['data']['num_bins'],
                          use_rmsnorm=config['model']['use_rms'],
                          face_bin=config['model']['face_bin'],
                          fixed_std=config['model']['fixed_std'] if 'fixed_std' in config['model'] else 0.0,
                          use_identity_encoder=config['model']['use_identity_encoder'] if 'use_identity_encoder' in config['model'] else False,
                          hidden_dim=config['model']['hidden_dim'],
                          num_layers=config['model']['num_layers'],
                          num_heads=config['model']['num_heads']).to(device)
    model.eval()

    # 2. Load Checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt['ema'] if 'ema' in ckpt else ckpt['model']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    mesh_files = sorted(glob(os.path.join(args.input_folder, "*.obj")))
    if not mesh_files:
        print(f"No .obj files found in {args.input_folder}")
        return

    print(f"Found {len(mesh_files)} meshes. Starting inference...")

    for mesh_path in tqdm(mesh_files):
        data = preprocess_mesh(mesh_path, max_seq_length=800)
        x = data['tokens'].to(device)       # [1, 800, 9]
        mask = data['masks'].to(device)     # [1, 800]
        num_faces = data['num_faces'].to(device)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                recon = model.decode(x.reshape(1, -1, 3), cond=num_faces, mask=mask) + x
                # recon, _, _ = model(x, cond=num_faces, mask=mask)
        
        recon_coords = recon # continuous output
        valid_recon = recon_coords[mask].reshape(-1, 3, 3).float().cpu().numpy()            
        # Save
        save_name = f"recon_{data['filename']}"
        save_path = os.path.join(args.output_dir, save_name)
        save_mesh(valid_recon, save_path)
            
    print(f"Done! Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config yaml")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model .pt")
    parser.add_argument('--input_folder', type=str, required=True, help="Folder containing .obj files")
    parser.add_argument('--output_dir', type=str, default='recon_results', help="Where to save outputs")
    
    args = parser.parse_args()
    
    run_inference(args)