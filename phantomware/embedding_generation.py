'''

This code implements the embedding generation process.
It extracts embeddings from a pre-trained MalConv model,
selects representative goodware and malware samples,
and generates synthetic malware embeddings.
The generated embeddings can be used to finetune machine learning models for malware detection.

'''

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from malconv.MalConvGCT_nocat import MalConvGCT
from malconv.binaryLoader import BinaryDataset, pad_collate_func

import numpy as np
import os
import pandas as pd
from tqdm import tqdm


def select_goodware_extremes(
        good_embeds,
        n_components=20,
        samples_per_side=50,
        device="cuda"
):
    """
    Select goodware samples lying on the outer shell
    of the goodware manifold.

    Returns:
        frontier_samples [N,D]
    """

    good_cpu = good_embeds.detach().cpu().numpy()

    pca = PCA(
        n_components=min(
            n_components,
            good_cpu.shape[1]
        ),
        random_state=0
    )

    good_proj = pca.fit_transform(good_cpu)
    selected_idx = set()

    for dim in range(good_proj.shape[1]):
        proj = good_proj[:, dim]

        # most positive
        pos_idx = np.argsort(proj)[-samples_per_side:]

        # most negative
        neg_idx = np.argsort(proj)[:samples_per_side]

        selected_idx.update(pos_idx.tolist())
        selected_idx.update(neg_idx.tolist())

    selected_idx = sorted(list(selected_idx))
    frontier_samples = good_embeds[selected_idx]

    return frontier_samples.to(device)


def select_goodware_representatives(
        good_embeds: torch.Tensor,
        mal_embeds: torch.Tensor,
        k_centroids: int = 1000,
        k_boundary: int = 1000,
        device: str = "cuda"
):
    """
    Returns:
        combined goodware subset:
        [k_centroids + k_boundary, D]
    """

    good_cpu = good_embeds.detach().cpu().numpy()
    mal = mal_embeds.to(device)

    # -------------------------------------------------------
    # 1. K-MEANS CENTROIDS (global goodware structure)
    # -------------------------------------------------------
    kmeans = KMeans(
        n_clusters=k_centroids,
        random_state=0,
        n_init="auto"
    )

    # use this for normalized space
    kmeans.fit(good_cpu)

    centroids = torch.tensor(
        kmeans.cluster_centers_,
        dtype=torch.float32,
        device=device
    )

    # -------------------------------------------------------
    # 2. BOUNDARY SAMPLES (closest goodware to malware)
    # -------------------------------------------------------
    good = good_embeds.to(device)

    d = torch.cdist(good, mal)
    hard_score = torch.logsumexp(-d, dim=1)
    boundary_idx = torch.topk(hard_score, k=k_boundary).indices

    boundary_samples = good[boundary_idx]

    # -------------------------------------------------------
    # 3. COMBINE
    # -------------------------------------------------------
    goodware_subset = torch.cat([centroids, boundary_samples], dim=0)

    return boundary_samples.to(device), goodware_subset.to(device)


def sample_goodware(good_embeds):

    idx = torch.randint(
        0,
        good_embeds.size(0),
        (1,),
        device=good_embeds.device
    )

    return good_embeds[idx]


# -------------------------------
# Extract embeddings from loader
# -------------------------------
def extract_embeddings(model, loader, device):
    model.eval()
    good_embeds, mal_embeds = [], []

    with torch.no_grad():
        for x, labels, _, _ in tqdm(loader):
            x = x.to(device)
            labels = labels.to(device)
            _, penult, _ = model(x)

            good_mask = labels == 0
            mal_mask = labels == 1

            if good_mask.any():
                good_embeds.append(penult[good_mask].cpu())

            if mal_mask.any():
                mal_embeds.append(penult[mal_mask].cpu())

    if len(good_embeds) > 0:
        good_embeds = torch.cat(good_embeds, dim=0)

    if len(mal_embeds) > 0:
        mal_embeds = torch.cat(mal_embeds, dim=0)

    return good_embeds, mal_embeds


def compute_discriminative_direction(
    target_embeds,
    good_embeds,
    eps=1e-8,
    power=1.5
):
    """
    Computes malware-discriminative latent directions.

    Returns:
        direction: [D] signed perturbation direction
        importance: [D] normalized importance weights
    """

    mu_t = target_embeds.mean(dim=0)
    mu_g = good_embeds.mean(dim=0)

    std_t = target_embeds.std(dim=0)
    std_g = good_embeds.std(dim=0)

    # Signed malware direction
    direction = mu_t - mu_g

    # Fisher-like discriminative score
    fisher = direction.abs() / (std_t + std_g + eps)

    # Normalize importance to [0,1]
    importance = fisher / (fisher.max() + eps)

    # Sharpen important dimensions
    importance = importance.pow(power)

    # Weighted malware direction
    discriminative_direction = direction * importance

    return discriminative_direction


def generate_embedding(
    good_embed,
    discriminative_direction,
    alpha_range=(0.1, 1.0),
    normalize_direction=True,
    noise_std=0.02,
):
    direction = discriminative_direction

    if normalize_direction:
        direction = F.normalize(direction, dim=0)

    alpha = torch.empty(1).uniform_(*alpha_range).item()

    synthetic = good_embed + alpha * direction

    if noise_std > 0:
        synthetic += noise_std * torch.randn_like(synthetic)

    return synthetic


def generate_synthetic_embeddings(
    mal_target_embeds,
    num_samples,
    good_embeds_centroid_and_boundary,
    generation_pool
):
    discriminative_direction = compute_discriminative_direction(
        mal_target_embeds,
        good_embeds_centroid_and_boundary
    )

    all_samples = []

    for _ in range(num_samples):
        # ---------------------------------------------
        # Sample anchors
        # ---------------------------------------------
        random_good = sample_goodware(generation_pool)

        synthetic = generate_embedding(
            random_good,
            discriminative_direction
        )

        synthetic = F.normalize(synthetic, dim=0)
        all_samples.append(synthetic.cpu())

    return torch.cat(all_samples, dim=0)


device = "cuda"

with open("../target_family.txt", "r") as f:
    target_family = f.read().strip()

# -----------------------
# Load dataset
# -----------------------
training_goodware_samples = pd.read_csv('../data/real_data/training_goodware.csv')
training_malware_samples = pd.read_csv('../data/real_data/training_malware.csv')
training_target_family_malware_samples = training_malware_samples[training_malware_samples["family"] == target_family]

dataset = BinaryDataset(
    "../data/real_data/goodware/",
    "../data/real_data/malware/",
    training_goodware_samples,
    training_target_family_malware_samples,
    sort_by_size=True,
    max_len=16000000
)

loader_goodware_and_target = DataLoader(
    dataset,
    batch_size=128,
    num_workers=0,
    collate_fn=pad_collate_func
)

# -----------------------
# Load MalConv
# -----------------------
model_path = os.path.join('../models', target_family + ".checkpoint")

malconv_model = MalConvGCT(
    channels=128,
    window_size=256,
    stride=64,
    embd_size=8,
    low_mem=False
).to(device)

checkpoint = torch.load(model_path, map_location=device)
malconv_model.load_state_dict(checkpoint["model_state_dict"])

# -----------------------
# Extract embeddings
# -----------------------
goodware_embeddings, target_family_embeddings = extract_embeddings(malconv_model, loader_goodware_and_target, device)
print("Good:", goodware_embeddings.shape, "Mal:", target_family_embeddings.shape, flush=True)

goodware_embeddings = goodware_embeddings.to(device)
target_family_embeddings = target_family_embeddings.to(device)

goodware_embeddings = F.normalize(goodware_embeddings, dim=1)
target_family_embeddings = F.normalize(target_family_embeddings, dim=1)

# -----------------------
# Sample centroid and frontier samples from the goodware manifold
# -----------------------
goodware_boundary_samples, goodware_centroid_and_boundary_samples = select_goodware_representatives(
    goodware_embeddings,
    target_family_embeddings,
    k_boundary=1000,
    k_centroids=1000,
    device=device
)

goodware_frontier_samples = select_goodware_extremes(
    goodware_embeddings,
    n_components=20,
    samples_per_side=50,
    device=device
)

print('good_frontier.shape', goodware_frontier_samples.shape)

goodware_anchors_pool = torch.cat([
    goodware_boundary_samples[:50],
    goodware_frontier_samples
], dim=0)

goodware_anchors_pool = torch.unique(
    goodware_anchors_pool,
    dim=0
)

# -----------------------
# Generate synthetic
# -----------------------
synthetic_embeddings = generate_synthetic_embeddings(
    target_family_embeddings,
    500,
    goodware_centroid_and_boundary_samples,
    goodware_anchors_pool
)

# -----------------------
# Save synthetic
# -----------------------
save_path = os.path.join('../synthetic_data', f"synthetic_embeddings_{target_family}.csv")
stacked_synth_embs = synthetic_embeddings.cpu().numpy()
df = pd.DataFrame(stacked_synth_embs)
print('\ngenerated', len(df), 'synthetic samples', flush=True)
df.to_csv(save_path, index=False)