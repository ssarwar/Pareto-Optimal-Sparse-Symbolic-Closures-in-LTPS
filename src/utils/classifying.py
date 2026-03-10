import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter1d

def first_sustained_true(b, min_run):
    if b.ndim != 1 or b.size == 0:
        return None
    edges = np.flatnonzero(np.diff(np.r_[False, b, False]))
    if edges.size == 0:
        return None
    segs = list(zip(edges[::2], edges[1::2]))
    for s, e in segs:
        if (e - s) >= min_run:
            return s
    return None

def split_index_from_quasineutral(ne, ni, tau=0.05, sigma=2.0, min_run=8, left_guard=5):
    ne = np.asarray(ne, float)
    ni = np.asarray(ni, float)
    L  = ne.size
    eps = 1e-30

    rel = np.abs(ne - ni) / (0.5 * (ne + ni) + eps)
    rel_s = gaussian_filter1d(rel, sigma=sigma)

    cand = rel_s > tau
    cand[:max(0, left_guard)] = False 

    idx = first_sustained_true(cand, min_run=min_run)
    if idx is None:
        idx = L
    return int(idx)

def run_knn():
    try:
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(os.path.dirname(cur_dir), 'data/')
        problem = "rf"
        case = "pic"
        PATH_TO_DATA = os.path.join(base_path, problem, case, "ml_data", "half_ccp_dataset.npz")

        print(f"Attempting to load data from: {PATH_TO_DATA}")

        npz_data = np.load(PATH_TO_DATA)
        load_data = npz_data["profiles"]

        fluxi = load_data[..., 1]
        ne    = load_data[..., 2]
        ni    = load_data[..., 3]
        phi   = load_data[..., 4]
        E     = load_data[..., 5]
        Te    = load_data[..., 6]

        X_features_all = np.stack([fluxi, ni, phi, E, Te], axis=-1)
        freq, pres = load_data[..., 9], load_data[..., 8]
        input_feature_names = ['flux', 'ni', 'phi', 'E', 'Te']
    except FileNotFoundError:
        print(f"Error: Data file not found at '{PATH_TO_DATA}'.")
        return
    except KeyError as e:
        print(f"Error: A key was not found in the .npz file: {e}")
        return

    num_snapshots, num_points, _ = X_features_all.shape

    labels_reshaped = np.zeros((num_snapshots, num_points), dtype=int)
    split_indices   = np.full(num_snapshots, num_points, dtype=int) 

    TAU = 0.05
    SIGMA = 2.0 
    MIN_RUN = 8 
    LEFT_GUARD = 5 

    for i in range(num_snapshots):
        idx = split_index_from_quasineutral(
            ne[i], ni[i], tau=TAU, sigma=SIGMA, min_run=MIN_RUN, left_guard=LEFT_GUARD)
        split_indices[i] = idx
        lab = np.zeros(num_points, dtype=int)
        if idx < num_points:
            lab[idx:] = 1
        labels_reshaped[i] = lab

    output_dir = os.path.join(base_path, problem, case)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "split_indices.npy"), split_indices)
    print(output_dir)
    # print(f"\nSaving plots to: {output_dir}")

    # for i in range(num_snapshots):
    #     snapshot_labels = labels_reshaped[i]
    #     snapshot_features_all = X_features_all[i, :, :]
    #     snapshot_pres = pres[i, 0]
    #     snapshot_freq = freq[i, 0]
    #     spatial_coords = np.arange(num_points)

    #     fig, axes = plt.subplots(5, 1, figsize=(5, 14), sharex=True)
    #     sns.set_theme(style="whitegrid")
    #     fig.suptitle(f'(Pressure: {snapshot_pres:.2f}, Freq: {snapshot_freq:.2f})', fontsize=16)

    #     for j in range(len(input_feature_names)):
    #         ax = axes[j]
    #         feature_data = snapshot_features_all[:, j]
    #         ax.plot(spatial_coords, feature_data, color='navy', lw=2.0, zorder=10)

    #         y_min_val = np.min(feature_data)
    #         y_max_val = np.max(feature_data)
    #         y_range = y_max_val - y_min_val if y_max_val > y_min_val else 1.0
    #         ax.set_ylim(bottom=y_min_val - 0.1 * y_range, top=y_max_val + 0.1 * y_range)

    #         # shade bulk (left) and sheath (right) according to labels
    #         ax.fill_between(spatial_coords, ax.get_ylim()[0], ax.get_ylim()[1],
    #                         where=snapshot_labels == 0, color='skyblue', alpha=0.45,
    #                         interpolate=False, zorder=0)
    #         ax.fill_between(spatial_coords, ax.get_ylim()[0], ax.get_ylim()[1],
    #                         where=snapshot_labels == 1, color='salmon', alpha=0.45,
    #                         interpolate=False, zorder=0)

    #         # draw the split index
    #         s = split_indices[i]
    #         if 0 < s < num_points:
    #             ax.axvline(s, color='k', lw=1.2, ls='--', alpha=0.8)

    #         ax.set_ylabel(input_feature_names[j], fontsize=12)

    #     legend_patch_A = mpatches.Patch(color='skyblue', alpha=0.45, label='Bulk (ne≈ni)')
    #     legend_patch_B = mpatches.Patch(color='salmon',  alpha=0.45, label='Sheath')
    #     fig.legend(handles=[legend_patch_A, legend_patch_B], fontsize=12,
    #                loc='upper right', bbox_to_anchor=(1, 0.96))
    #     axes[-1].set_xlabel('Spatial Point Index', fontsize=14)

    #     fig.tight_layout(rect=[0, 0, 1, 0.96])
    #     save_path = os.path.join(output_dir, f'{snapshot_freq:.2f}MHz{snapshot_pres:.2f}mTorr.png')
    #     plt.savefig(save_path, dpi=150)
    #     plt.close(fig)

    #     if (i + 1) % 10 == 0 or i == num_snapshots - 1:
    #         print(f"  ... Saved plot for snapshot {i + 1}/{num_snapshots}")

if __name__ == '__main__':
    run_knn()
