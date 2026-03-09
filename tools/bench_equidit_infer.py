import time
import numpy as np
import torch
import argparse
from models.equidit import DiT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--mode', type=str, default='infer', choices=['infer', 'train'])
    parser.add_argument('--grad-ckpt', action='store_true')
    args = parser.parse_args()

    model = DiT(
        hidden_dim=768,
        num_heads=12,
        max_length=3200,
        num_layers=12,
        gradient_checkpointing=args.grad_ckpt,
        use_coord_encoding=True,
        version=3,
        pe_freq=20,
        mixed_precision='bf16',
        use_dit_like_pe=False,
        face_cond=True,
        face_bin=20,
        use_rmsnorm=True,
        is_latent=False,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if args.mode == 'infer':
        model.eval()
    else:
        model.train()

    faces_list = [400, 800, 1600, 3200]
    batch_size = args.batch_size
    warmup = args.warmup
    repeats = args.repeats

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    print(f'benchmark target: EquiDiT {args.mode} (models/equidit.py: DiT.forward)')
    print(f'device={device}, batch_size={batch_size}, warmup={warmup}, repeats={repeats}')
    print('unit: ms')

    for n_faces in faces_list:
        x = torch.randn(batch_size, n_faces, 9, device=device)
        t = torch.rand(batch_size, device=device)
        y = torch.full((batch_size,), n_faces, device=device, dtype=torch.long)
        mask = torch.ones(batch_size, n_faces, device=device, dtype=torch.bool)

        for _ in range(warmup):
            if args.mode == 'infer':
                with torch.inference_mode():
                    if device.type == 'cuda':
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            _ = model(x, t, y, mask)
                    else:
                        _ = model(x, t, y, mask)
            else:
                model.zero_grad(set_to_none=True)
                if device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        out = model(x, t, y, mask)
                        loss = out.float().pow(2).mean()
                else:
                    out = model(x, t, y, mask)
                    loss = out.float().pow(2).mean()
                loss.backward()
        sync()

        times = []
        for _ in range(repeats):
            sync()
            t0 = time.perf_counter()
            if args.mode == 'infer':
                with torch.inference_mode():
                    if device.type == 'cuda':
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            _ = model(x, t, y, mask)
                    else:
                        _ = model(x, t, y, mask)
            else:
                model.zero_grad(set_to_none=True)
                if device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        out = model(x, t, y, mask)
                        loss = out.float().pow(2).mean()
                else:
                    out = model(x, t, y, mask)
                    loss = out.float().pow(2).mean()
                loss.backward()
            sync()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        arr = np.array(times, dtype=np.float64)
        print(
            f'faces={n_faces:4d} | mean={arr.mean():8.2f} ms | std={arr.std(ddof=0):7.2f} ms '
            f'| min={arr.min():8.2f} ms | max={arr.max():8.2f} ms | runs={repeats}'
        )


if __name__ == '__main__':
    main()
