import argparse
from pathlib import Path
import numpy as np


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _safe_uid(obj, file_stem, rec_idx):
    oid = obj.get("object_id", None)
    if oid is None:
        return f"{file_stem}_{rec_idx:06d}"
    oid = str(oid).strip()
    if oid == "":
        return f"{file_stem}_{rec_idx:06d}"
    return oid


def convert(src_dir: Path, dst_dir: Path):
    out_mesh_dir = dst_dir / "objaverse_occ_v5_ids"
    out_split_dir = dst_dir / "split"
    out_mesh_dir.mkdir(parents=True, exist_ok=True)
    out_split_dir.mkdir(parents=True, exist_ok=True)

    # clean old output
    for p in out_mesh_dir.glob("*.npz"):
        p.unlink()

    train_records = []
    test_records = []
    used_uids = set()

    files = sorted(src_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No npz files found in {src_dir}")

    total = 0
    kept = 0

    for fidx, fpath in enumerate(files):
        arr = np.load(fpath, allow_pickle=True)["arr_0"]
        is_test_file = ("val" in fpath.stem.lower()) or ("test" in fpath.stem.lower())

        for ridx, rec in enumerate(arr):
            total += 1
            obj = rec if isinstance(rec, dict) else (rec.item() if hasattr(rec, "item") else rec)
            if not isinstance(obj, dict):
                continue
            if "vertices" not in obj or "faces" not in obj:
                continue

            vertices = _to_numpy(obj["vertices"])
            faces = _to_numpy(obj["faces"])

            if vertices.ndim != 2 or vertices.shape[1] != 3:
                continue
            if faces.ndim != 2 or faces.shape[1] != 3:
                continue
            if len(vertices) == 0 or len(faces) == 0:
                continue

            uid_base = _safe_uid(obj, fpath.stem, ridx)
            uid = uid_base
            dedup_i = 1
            while uid in used_uids:
                uid = f"{uid_base}_{dedup_i}"
                dedup_i += 1
            used_uids.add(uid)

            vertices = vertices.astype(np.float32)
            faces = faces.astype(np.int64)

            out_path = out_mesh_dir / f"{uid}.npz"
            np.savez(
                out_path,
                vertices=vertices,
                faces=faces,
                min_length=np.float32(0.0),
                uid=np.array(uid),
                vertices_num=np.int64(vertices.shape[0]),
                faces_num=np.int64(faces.shape[0]),
            )

            rec_meta = {
                "uid": uid,
                "min_length": np.float32(0.0),
                "vertices_num": int(vertices.shape[0]),
                "faces_num": int(faces.shape[0]),
            }
            if is_test_file:
                test_records.append(rec_meta)
            else:
                train_records.append(rec_meta)
            kept += 1

        print(f"[INFO] processed {fidx + 1}/{len(files)} files | kept so far: {kept}")

    if len(test_records) == 0 and len(train_records) > 0:
        # fallback: use one sample as test if no val/test shard exists
        test_records = [train_records[0]]

    np.savez(out_split_dir / "train.npz", npz_list=np.array(train_records, dtype=object))
    np.savez(out_split_dir / "test.npz", npz_list=np.array(test_records, dtype=object))

    print("[DONE] conversion finished")
    print("  src:", src_dir)
    print("  dst:", dst_dir)
    print("  total records:", total)
    print("  kept records:", kept)
    print("  train split:", len(train_records))
    print("  test split:", len(test_records))


def main():
    parser = argparse.ArgumentParser(description="Convert MeshXL Sketchfab NPZ shards into dummy-style mesh dataset")
    parser.add_argument("--src", type=str, default="/data1/sunqi/MeshFlow2/downloaded_data/MeshXL_Sketchfab_Dataset")
    parser.add_argument("--dst", type=str, default="/data1/sunqi/MeshFlow2/downloaded_data/sketchfab")
    args = parser.parse_args()

    convert(Path(args.src), Path(args.dst))


if __name__ == "__main__":
    main()
