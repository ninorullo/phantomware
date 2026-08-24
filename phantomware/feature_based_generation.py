'''

Anchor-Conditioned Adversarial Latent Augmentation

'''


import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
import os
import pandas as pd
from tqdm import tqdm
import time

from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from MalConvGCT_nocat import MalConvGCT
from binaryLoader import BinaryDataset, pad_collate_func


def compute_discriminative_mask(
    target_embeds,
    good_embeds,
    topk=32,
    eps=1e-8
):
    """
    Finds malware-specific latent dimensions.

    Returns:
        mask: [D]
    """

    # target_embeds = F.normalize(target_embeds, dim=1)
    # good_embeds = F.normalize(good_embeds, dim=1)

    mu_t = target_embeds.mean(dim=0)
    mu_g = good_embeds.mean(dim=0)

    std_t = target_embeds.std(dim=0)
    std_g = good_embeds.std(dim=0)

    fisher = (mu_t - mu_g).abs() / (std_t + std_g + eps)
    topk_idx = torch.topk(fisher, k=topk).indices

    mask = torch.zeros_like(fisher)
    mask[topk_idx] = 1.0

    return mask


def select_benign_extremes(
        good_embeds,
        n_components=20,
        samples_per_side=50,
        device="cuda"
):
    """
    Select benign samples lying on the outer shell
    of the benign manifold.

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


def select_benign_representatives(
        good_embeds: torch.Tensor,
        mal_embeds: torch.Tensor,
        k_centroids: int = 1000,
        k_boundary: int = 500,
        device: str = "cuda"
):
    """
    Returns:
        combined benign subset:
        [k_centroids + k_boundary, D]
    """

    good_cpu = good_embeds.detach().cpu().numpy()
    mal = mal_embeds.to(device)

    # -------------------------------------------------------
    # 1. K-MEANS CENTROIDS (global benign structure)
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
    # 2. BOUNDARY SAMPLES (closest benign to malware)
    # -------------------------------------------------------
    good = good_embeds.to(device)

    d = torch.cdist(good, mal)
    hard_score = torch.logsumexp(-d, dim=1)
    boundary_idx = torch.topk(hard_score, k=k_boundary).indices

    boundary_samples = good[boundary_idx]

    # -------------------------------------------------------
    # 3. COMBINE
    # -------------------------------------------------------
    benign_subset = torch.cat([centroids, boundary_samples], dim=0)

    return centroids.to(device), boundary_samples.to(device), benign_subset.to(device)


def sample_target(target_embeds):
    idx = torch.randint(
        0,
        target_embeds.size(0),
        (1,),
        device=target_embeds.device
    )

    return target_embeds[idx]


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

    mal_embeds = torch.cat(mal_embeds, dim=0)

    return good_embeds, mal_embeds


def generate_synthetic_embeddings_goodware_centered(
    mal_target_embeds,
    num_samples,
    good_embeds_centroid_and_boundary,
    generation_pool,
    perturb_scale=0.3,
    noise_scale=0.03
):
    """
    Benign-centered embeddings with malware perturbations.
    """

    # ---------------------------------------------
    # Malware-discriminative dimensions
    # ---------------------------------------------
    malware_mask = compute_discriminative_mask(
        mal_target_embeds,
        good_embeds_centroid_and_boundary,
        topk=8
    ).unsqueeze(0)

    all_samples = []

    # ---------------------------------------------
    # Estimate benign manifold scale
    # ---------------------------------------------
    benign_radius = torch.pdist(good_embeds_centroid_and_boundary).median()

    for _ in range(num_samples):

        # ---------------------------------------------
        # Sample anchors
        # ---------------------------------------------
        random_good = sample_goodware(generation_pool)
        # random_target = sample_target(mal_target_embeds)

        # ---------------------------------------------
        # Malware direction
        # ---------------------------------------------
        mal_proto = mal_target_embeds.mean(dim=0, keepdim=True)
        malware_direction = mal_proto - random_good

        # only keep malware-specific dimensions
        malware_direction = malware_direction * malware_mask

        # normalize perturbation direction
        malware_direction = F.normalize(
            malware_direction,
            dim=1
        )

        # ---------------------------------------------
        # Small perturbation magnitude
        # ---------------------------------------------
        magnitude = (
            torch.rand(1, 1, device=device)
            * benign_radius
            * perturb_scale
        )

        # ---------------------------------------------
        # Benign-centered perturbation
        # ---------------------------------------------
        synth_sample = (
            random_good
            + magnitude * malware_direction
        )

        # ---------------------------------------------
        # Small isotropic noise
        # ---------------------------------------------
        noise = torch.randn_like(synth_sample)
        noise = F.normalize(noise, dim=1)

        synth_sample = (
            synth_sample
            + noise_scale * noise
        )

        # ---------------------------------------------
        # Normalize
        # ---------------------------------------------
        synth_sample = F.normalize(
            synth_sample,
            dim=1
        )

        all_samples.append(synth_sample.cpu())

    return torch.cat(all_samples, dim=0)


def generate_synthetic_embeddings_goodware_centered2(
    mal_target_embeds,
    num_samples,
    good_embeds_centroid_and_boundary,
    generation_pool,
    perturb_scale=0.8,
    noise_scale=0.03
):
    """
    Benign-centered embeddings with malware perturbations.
    """

    # ---------------------------------------------
    # Malware-discriminative dimensions
    # ---------------------------------------------
    malware_mask = compute_discriminative_mask(
        mal_target_embeds,
        good_embeds_centroid_and_boundary,
        topk=8
    ).unsqueeze(0)

    all_samples = []

    # ---------------------------------------------
    # Estimate benign manifold scale
    # ---------------------------------------------
    benign_radius = torch.pdist(good_embeds_centroid_and_boundary).median()

    for _ in range(num_samples):

        # ---------------------------------------------
        # Sample anchors
        # ---------------------------------------------
        random_good = sample_goodware(generation_pool)
        # random_target = sample_target(mal_target_embeds)

        # ---------------------------------------------
        # Malware direction
        # ---------------------------------------------
        mal_proto = mal_target_embeds.mean(dim=0, keepdim=True)
        malware_direction = mal_proto - random_good

        # only keep malware-specific dimensions
        malware_direction = malware_direction * malware_mask

        # normalize perturbation direction
        malware_direction = F.normalize(
            malware_direction,
            dim=1
        )

        # ---------------------------------------------
        # Small perturbation magnitude
        # ---------------------------------------------
        magnitude = (
            torch.rand(1, 1, device=device)
            * benign_radius
            * perturb_scale
        )

        # ---------------------------------------------
        # Benign-centered perturbation
        # ---------------------------------------------
        alpha = 1.0 + torch.rand(1, 1, device=device)

        synth_sample = (
                random_good
                + alpha * magnitude * malware_direction
        )

        # ---------------------------------------------
        # Small isotropic noise
        # ---------------------------------------------
        noise = torch.randn_like(synth_sample)
        noise = F.normalize(noise, dim=1)

        synth_sample = (
            synth_sample
            + noise_scale * noise
        )

        # ---------------------------------------------
        # Normalize
        # ---------------------------------------------
        synth_sample = F.normalize(
            synth_sample,
            dim=1
        )

        all_samples.append(synth_sample.cpu())

    return torch.cat(all_samples, dim=0)


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


def generate_synthetic_embedding(
    good_embed,
    discriminative_direction,
    alpha_range=(0.2, 1.0),
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

#paper version
def generate_synthetic_embeddings_goodware_centered4(
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

        synthetic = generate_synthetic_embedding(
            random_good,
            discriminative_direction
        )

        synthetic = F.normalize(synthetic, dim=0)
        all_samples.append(synthetic.cpu())

    return torch.cat(all_samples, dim=0)


def compute_discriminative_mask2(
    target_embeds,
    good_embeds,
    eps=1e-8
):
    mu_t = target_embeds.mean(dim=0)
    mu_g = good_embeds.mean(dim=0)

    std_t = target_embeds.std(dim=0)
    std_g = good_embeds.std(dim=0)

    fisher = (mu_t - mu_g).abs() / (std_t + std_g + eps)

    # normalize to [0,1]
    mask = fisher / (fisher.max() + eps)
    mask = mask.pow(1.5)  # sharpen important dims but keep smoothness

    return mask


def nearest_malware(x, mal_embeds):
    x = x.view(1, -1)   # always [1,128]
    sims_cos = F.cosine_similarity(x, mal_embeds)
    sims_l2 = -torch.cdist(x, mal_embeds).squeeze(0)
    sims = sims_cos + 0.3 * sims_l2
    idx = torch.argmax(sims)
    return mal_embeds[idx].unsqueeze(0)

def generate_synthetic_embeddings_goodware_centered3(
    mal_target_embeds,
    num_samples,
    good_embeds_centroid_and_boundary,
    generation_pool,
    perturb_scale=0.8,
    noise_scale=0.01
):
    """
    Benign-centered embeddings with malware perturbations.
    """

    # ---------------------------------------------
    # Malware-discriminative dimensions
    # ---------------------------------------------
    malware_mask = compute_discriminative_mask2(
        mal_target_embeds,
        good_embeds_centroid_and_boundary
    ).unsqueeze(0)

    all_samples = []

    for _ in range(num_samples):

        # ---------------------------------------------
        # Sample anchors
        # ---------------------------------------------
        random_good = sample_goodware(generation_pool)

        # ---------------------------------------------
        # Malware direction
        # ---------------------------------------------
        random_target = nearest_malware(random_good, mal_target_embeds)
        malware_direction = random_target - random_good

        # only keep malware-specific dimensions
        malware_direction = malware_direction * malware_mask

        # ---------------------------------------------
        # Benign-centered perturbation
        # ---------------------------------------------
        # small step toward malware direction (stay near benign region)
        direction = F.normalize(malware_direction, dim=1)
        direction = direction * malware_mask
        direction = F.normalize(direction, dim=1)
        alpha = torch.rand(1, 1, device=device) * 0.3 + 0.1  # [0.1, 0.4]
        benign_std = good_embeds_centroid_and_boundary.std(dim=0).mean()
        step_size = alpha * benign_std * perturb_scale

        synth_sample = random_good + step_size * direction

        # ---------------------------------------------
        # Small isotropic noise
        # ---------------------------------------------
        noise = torch.randn_like(synth_sample)
        noise = noise * malware_mask
        noise = F.normalize(noise, dim=1)

        synth_sample = (
            synth_sample
            + noise_scale * noise
        )

        # ---------------------------------------------
        # Normalize
        # ---------------------------------------------
        synth_sample = F.normalize(
            synth_sample,
            dim=1
        )

        all_samples.append(synth_sample.cpu())

    return torch.cat(all_samples, dim=0)


def generate_synthetic_embeddings_malware_centered(
    mal_target_embeds,
    num_samples,
    good_embeds
):
    malware_mask = compute_discriminative_mask(
        mal_target_embeds,
        good_embeds,
        topk=16
    ).unsqueeze(0)

    all_samples = []
    global_radius = torch.pdist(good_embeds).median()

    for _ in range(num_samples):
        random_good = sample_goodware(good_embeds)
        random_target = sample_target(mal_target_embeds)

        direction = F.normalize(random_target - random_good, dim=1)
        magnitude = torch.rand(1, 1, device=device) * global_radius
        shared = random_good + magnitude * direction

        synth_sample = (
                malware_mask * random_target
                +
                (1.0 - malware_mask) * shared
        )

        synth_sample = F.normalize(synth_sample, dim=1)
        all_samples.append(synth_sample.cpu())

    return torch.cat(all_samples, dim=0)


families = [
    'tofsee'

    # #'ganelp',
    #  'sfone',
    #  'wacatac',
    #  'sillyp2p',
    # #'upatre',
    #  'wabot',
    #  'small',
    #  'dinwod',
    # #'mira',
    #  'berbew',
    # #'ceeinject',
    # #'gepys',
    #  'benjamin',
    # #'musecador',
    #  'autoit',
    # #'gandcrab',
    # #'drolnux',
    #  'smokeloader',
    #  'unruy',
    #  'qukart',
    #  'delf',
    #  'padodor',
    # #'autorun',
    # #'urelas',
    #  'mintluks',
    #  'picsys',
    #  'fakeav',
    # #'bladabindi',
    #  'zbot',
    # #'vflooder',
    # #'lunam',
    # #'tofsee',
    #  'sytro',
    # #'fuerboos',
    # #'mydoom',
    # #'pykspa',
    #  'agent',
    #  'soltern',
    #  'qqpass',
    # #'blocker',
    # #'ircbot',
    # #'coinminer',
    # #'salgorea',
    #  'stormser',
    #  'fasong',
    #  'cryptinject',
    # #'vobfus',
    #  'dorv',
    #  'nitol',
    #  'stration',
    #  'eggnog',
    #  'occamy',
    #  'banload',
    #  'glupteba',
    # #'shifu',
    #  'pluto',
    #  'ditertag'
]

device = "cuda"
script_dir = os.path.dirname(os.path.abspath(__file__))

for TARGET in families:
    print(f"\n=== FAMILY: {TARGET} ===", flush=True)

    family_start_time = time.time()

    # -----------------------
    # Load dataset
    # -----------------------
    loading_dataset_start_time = time.time()
    print('loading dataset...', flush=True)
    ben_train = pd.read_csv(f'../MalwareDataset/windows/ben_train.csv')
    mal_train = pd.read_csv(f'../MalwareDataset/windows/mal_train_{TARGET}.csv')
    mal_train = mal_train[mal_train["family"] == TARGET]

    dataset = BinaryDataset(
        "../MalwareDataset/windows/goodware/",
        "../MalwareDataset/windows/altered/",
        ben_train,
        mal_train,
        "windows",
        sort_by_size=True,
        max_len=16000000
    )

    loader_goodware_and_target = DataLoader(
        dataset,
        batch_size=128,
        num_workers=0,
        collate_fn=pad_collate_func
    )

    mal_test = pd.read_csv(f'../MalwareDataset/windows/mal_test_{TARGET}.csv')
    mal_test = mal_test[mal_test["family"] == TARGET]

    test_dataset = BinaryDataset(
        None,
        "../MalwareDataset/windows/altered/",
        None,
        mal_test,
        "windows",
        sort_by_size=True,
        max_len=16000000
    )

    loader_mal_test = DataLoader(
        test_dataset,
        batch_size=128,
        num_workers=0,
        collate_fn=pad_collate_func
    )

    print(f'dataset loaded in {(time.time() - loading_dataset_start_time):.1f} seconds', flush=True)

    # -----------------------
    # Load MalConv
    # -----------------------
    model_folder = os.path.join(script_dir, "nocat_MalConvGCT_channels_128_filterSize_256_stride_64_embdSize_8_maxFileLen_16MB_target_family_"+TARGET)
    model_path = os.path.join(model_folder, TARGET + ".checkpoint")

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
    extract_embedding_start_time = time.time()
    print('\ncomputing embeddings...', flush=True)
    good_embeds, mal_target_embeds = extract_embeddings(malconv_model, loader_goodware_and_target, device)
    #_, mal_test_embeds = extract_embeddings(malconv_model, loader_mal_test, device)
    print("Good:", good_embeds.shape, "Mal:", mal_target_embeds.shape, flush=True)
    print(f'embeddings computed in {(time.time() - extract_embedding_start_time):.1f} seconds', flush=True)

    good_embeds = good_embeds.to(device)
    mal_target_embeds = mal_target_embeds.to(device)
    #mal_test_embeds = mal_test_embeds.to(device)

    good_embeds = F.normalize(good_embeds, dim=1)
    mal_target_embeds = F.normalize(mal_target_embeds, dim=1)
    #mal_test_embeds = F.normalize(mal_test_embeds, dim=1)

    # -----------------------
    # Subsample malware and goodware embeddings
    # -----------------------
    selection_start_time = time.time()
    print('\nselecting malware and goodware...', flush=True)
    good_centroid_samples, good_boundary_samples, good_embeds_centroid_and_boundary = select_benign_representatives(
        good_embeds.to(device),
        mal_target_embeds.to(device),
        k_boundary=1000,
        k_centroids=1000,
        device=device
    )

    good_frontier = select_benign_extremes(
        good_embeds,
        n_components=20,
        samples_per_side=50,
        device=device
    )

    print('good_frontier.shape',good_frontier.shape)

    generation_pool = torch.cat([
        good_boundary_samples[:50],
        good_frontier
    ], dim=0)

    generation_pool = torch.unique(
        generation_pool,
        dim=0
    )

    print(f'selection done in {(time.time() - selection_start_time):.1f} seconds', flush=True)

    # -----------------------
    # Generate synthetic
    # -----------------------
    generating_synthetic_embeddings_start_time = time.time()
    print('\ngenerating synthetic embeddings...', flush=True)

    synth_embs = generate_synthetic_embeddings_goodware_centered4(
        mal_target_embeds,
        500,
        good_embeds_centroid_and_boundary,
        generation_pool
    )

    # synth_embs = generate_synthetic_embeddings_malware_centered(
    #     mal_embeds,
    #     500,
    #     good_centroid_samples
    # )

    print(f'generation done in {(time.time() - generating_synthetic_embeddings_start_time):.1f} seconds', flush=True)

    # -----------------------
    # Save synthetic
    # -----------------------
    save_path = os.path.join(model_folder, f"synth_embs_feature_based_{TARGET}.csv")
    print('\nsaving synth data to', save_path)
    stacked_synth_embs = synth_embs.cpu().numpy()
    df = pd.DataFrame(stacked_synth_embs)
    print('\ngenerated', len(df), 'synthetic samples', flush=True)
    df.to_csv(save_path, index=False)


    # # -----------------------
    # # Generate plots
    # # -----------------------
    # # 1. Prepare the data
    # # We stack them into one big matrix to process together
    # synth_embs = synth_embs.to(device)
    #
    # # use this only with normalization
    # all_embeds = torch.cat([
    #         mal_embeds,
    #         good_embeds_centroid_and_boundary,
    #         mal_test_embeds,
    #         synth_embs
    #     ], dim=0).detach().cpu().numpy()
    #
    # pca = PCA(n_components=50, random_state=42)
    # all_embeds = pca.fit_transform(all_embeds)
    #
    # # Keep track of indices for plotting
    # n_mal = mal_embeds.shape[0]
    # n_good = good_embeds_centroid_and_boundary.shape[0]
    # n_test = mal_test_embeds.shape[0]
    # n_synth = synth_embs.shape[0]
    #
    # # 2. Run t-SNE
    # # perplexity usually works well between 5 and 50
    # tsne = TSNE(
    #     n_components=2,
    #     random_state=42,
    #     perplexity=30,
    #     metric="cosine",#only with normalization
    #     init="pca",
    #     learning_rate="auto"
    # )
    # embeds_2d = tsne.fit_transform(all_embeds)
    #
    # mal_2d = embeds_2d[:n_mal]
    # good_2d = embeds_2d[n_mal: n_mal + n_good]
    # test_2d = embeds_2d[n_mal + n_good : n_mal + n_good + n_test]
    # synth_2d = embeds_2d[n_mal + n_good + n_test:]
    #
    # # 4. Create the plot
    # plt.figure(figsize=(12, 10))
    #
    # plt.scatter(
    #     good_2d[:, 0], good_2d[:, 1],
    #     s=10,
    #     alpha=0.35,
    #     c='blue',
    #     marker='x',
    #     zorder=2,
    #     label='Benign'
    # )
    #
    # plt.scatter(
    #     synth_2d[:, 0], synth_2d[:, 1],
    #     s=25,
    #     alpha=1,
    #     c='red',
    #     marker='D',
    #     zorder=20,
    #     edgecolors='white',
    #     label=f'{TARGET} Synth'
    # )
    #
    # plt.scatter(
    #     test_2d[:, 0], test_2d[:, 1],
    #     s=85,
    #     alpha=1,
    #     c='magenta',
    #     marker='D',
    #     zorder=20,
    #     edgecolors='white',
    #     label=f'{TARGET} Test'
    # )
    #
    # plt.scatter(
    #     mal_2d[:, 0], mal_2d[:, 1],
    #     s=85,
    #     alpha=1,
    #     c='orange',
    #     marker='D',
    #     zorder=20,
    #     edgecolors='white',
    #     label=f'{TARGET} Train'
    # )
    #
    # plt.title('t-SNE ' + TARGET)
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(
    #     '../data/plots/2D_tsne_embeddings_feature_based_goodware_centric_2_' + TARGET + '.png',
    #     bbox_inches='tight'
    # )
    # plt.close()
    #
    # tsne3D = TSNE(
    #     n_components=3,
    #     random_state=42,
    #     perplexity=30,
    #     metric="cosine",
    #     init="pca",
    #     learning_rate="auto"
    # )
    # embeds_3d = tsne3D.fit_transform(all_embeds)
    #
    #
    # mal_2d = embeds_3d[:n_mal]
    # good_2d = embeds_3d[n_mal: n_mal + n_good]
    # test_2d = embeds_3d[n_mal + n_good: n_mal + n_good + n_test]
    # synth_2d = embeds_3d[n_mal + n_good + n_test:]
    #
    # # 4. Create the plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.scatter(
    #     good_2d[:, 0], good_2d[:, 1], good_2d[:, 2],
    #     s=10,
    #     alpha=0.35,
    #     c='blue',
    #     marker='x',
    #     zorder=2,
    #     label='Benign'
    # )
    #
    # ax.scatter(
    #     synth_2d[:, 0], synth_2d[:, 1], synth_2d[:, 2],
    #     s=25,
    #     alpha=1,
    #     c='red',
    #     marker='D',
    #     zorder=20,
    #     edgecolors='white',
    #     label=f'{TARGET} Synth'
    # )
    #
    # ax.scatter(
    #     test_2d[:, 0], test_2d[:, 1], test_2d[:, 2],
    #     s=85,
    #     alpha=1,
    #     c='magenta',
    #     marker='D',
    #     zorder=20,
    #     edgecolors='white',
    #     label=f'{TARGET} Test'
    # )
    #
    # ax.scatter(
    #     mal_2d[:, 0], mal_2d[:, 1], mal_2d[:, 2],
    #     s=85,
    #     alpha=1,
    #     c='orange',
    #     marker='D',
    #     zorder=20,
    #     edgecolors='white',
    #     label=f'{TARGET} Train'
    # )
    #
    # plt.title('t-SNE ' + TARGET)
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(
    #     '../data/plots/3D_tsne_embeddings_feature_based_goodware_centric_2_' + TARGET + '.png',
    #     bbox_inches='tight'
    # )
    # plt.close()