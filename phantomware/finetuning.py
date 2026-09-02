import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from binaryLoader import BinaryDataset, pad_collate_func, RandomChunkSampler
from MalConvGCT_nocat import MalConvGCT
from sklearn.cluster import KMeans


class SHADataset(Dataset):
    def __init__(
        self,
        csv_path,
        source_dir,
        sha_column="sha",
        isGoodware=False,
        max_len=16000000
    ):

        self.df = pd.read_csv(csv_path)
        self.sha_column = sha_column
        self.source_dir = source_dir
        self.max_len = max_len

        # keep only valid files
        if not isGoodware:
            self.df["path"] = self.df[sha_column].apply(
                lambda sha: os.path.join(source_dir, str(sha) + ".exe")
            )
        else:
            self.df["path"] = self.df[sha_column].apply(
                lambda sha: os.path.join(source_dir, str(sha))
            )

        self.df = self.df[self.df["path"].apply(os.path.exists)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        sha = row[self.sha_column]
        path = row["path"]

        with open(path, "rb") as f:
            x = f.read(self.max_len) # truncation
            x = np.frombuffer(x, dtype=np.uint8).astype(np.int16) + 1 # padding

        x = torch.tensor(x)

        return x, torch.tensor([1]), str(sha), idx


# =========================================================
# LOAD SYNTHETIC EMBEDDINGS
# =========================================================
def load_synthetic_embeddings(path, quantity=2000, device="cuda"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing synthetic embeddings: {path}")

    df = pd.read_csv(path)
    df2 = df.head(quantity).copy()
    tensor = torch.tensor(df2.values, dtype=torch.float32)

    return tensor.to(device)


def normalize_sha(x):
    """
    Unifies:
      - goodware: sha
      - malware: sha.exe
      - full paths
    """
    x = str(x).lower()
    x = os.path.basename(x)
    if x.endswith(".exe"):
        x = x[:-4]
    return x


def build_representative_datasets(
    ben_train,
    mal_train,
    selector_output,
    target_family,
    family_column="family",
    sha_column="sha"
):
    """
    Returns:
        selected_ben:
            representative goodware samples

        selected_mal:
            representative malware samples
            + all samples from target_family
    """

    # ------------------------------------------------------------
    # 1. Representative sets
    # ------------------------------------------------------------
    good_set = set(
        map(
            normalize_sha,
            selector_output["goodware_representatives"]
        )
    )

    mal_set = set(
        map(
            normalize_sha,
            selector_output["malware_representatives"]
        )
    )

    # ------------------------------------------------------------
    # 2. Normalize CSV keys
    # ------------------------------------------------------------
    ben_train = ben_train.copy()
    mal_train = mal_train.copy()

    ben_train["_key"] = ben_train[sha_column].apply(normalize_sha)
    mal_train["_key"] = mal_train[sha_column].apply(normalize_sha)

    # ------------------------------------------------------------
    # 3. GOODWARE selection
    # ------------------------------------------------------------
    selected_ben = ben_train[
        ben_train["_key"].isin(good_set)
    ].drop(columns=["_key"])

    # ------------------------------------------------------------
    # 4. MALWARE selection
    # ------------------------------------------------------------
    selected_mal = mal_train[
        (mal_train["_key"].isin(mal_set)) | (mal_train[family_column] == target_family)
    ].drop(columns=["_key"])

    # ------------------------------------------------------------
    # 5. Remove duplicates if any
    # ------------------------------------------------------------
    selected_mal = selected_mal.drop_duplicates()

    return selected_ben, selected_mal


# =========================================================
# DATA LOADER
# =========================================================
def build_train_loader(TARGET, representative_samples, batch_size=128, max_len=16000000):

    ben_train = pd.read_csv('../data/real_data/training_goodware.csv')
    mal_train = pd.read_csv('../data/real_data/training_malware.csv')

    ben_train, mal_train = build_representative_datasets(
            ben_train,
            mal_train,
            representative_samples,
            target_family=TARGET,
    )

    print('training: goodware', len(ben_train),' -  malware', len(mal_train))

    dataset = BinaryDataset(
        "../data/real_data/goodware/",
        "../data/real_data/malware/",
        ben_train,
        mal_train,
        sort_by_size=True,
        max_len=max_len
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=pad_collate_func,
        sampler=RandomChunkSampler(dataset, batch_size)
    )

    return loader


def select_symmetric_representatives(
    model,
    good_loader,
    malware_loader,
    device="cuda",
    hard_k=1000,
    num_centroids=1000,
    centroid_k=100
):
    """
    Fully symmetric representative selection:

    Returns:
        dict with:
            - goodware_representatives (paths)
            - malware_representatives (paths)

    Strategy:
        1. extract embeddings for both distributions
        2. compute centroids for both
        3. hard negatives cross-distribution
        4. centroid coverage per class
    """

    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------
    # 1. EXTRACT GOODWARE
    # ------------------------------------------------------------
    good_embs, good_paths = [], []

    with torch.no_grad():
        for x, _, paths, _ in tqdm(good_loader, desc="Goodware"):

            x = x.to(device)
            _, penult, _ = model(x)

            good_embs.append(penult.detach().cpu())
            good_paths.extend(paths)

    good_embs = torch.cat(good_embs, dim=0)
    good_paths = np.array(good_paths)

    # ------------------------------------------------------------
    # 2. EXTRACT MALWARE
    # ------------------------------------------------------------
    mal_embs, mal_paths = [], []

    with torch.no_grad():
        for x, _, paths, _ in tqdm(malware_loader, desc="Malware"):

            x = x.to(device)
            _, penult, _ = model(x)

            mal_embs.append(penult.detach().cpu())
            mal_paths.extend(paths)

    mal_embs = torch.cat(mal_embs, dim=0)
    mal_paths = np.array(mal_paths)

    # move to device for distance ops
    good_embs = good_embs.to(device)
    mal_embs = mal_embs.to(device)

    # ------------------------------------------------------------
    # 3. CENTROIDS
    # ------------------------------------------------------------
    good_center = good_embs.mean(dim=0, keepdim=True)
    mal_center = mal_embs.mean(dim=0, keepdim=True)

    # ------------------------------------------------------------
    # 4. HARD NEGATIVES (cross-class)
    # ------------------------------------------------------------

    # good samples closest to malware center
    good_to_mal = torch.norm(good_embs - mal_center, dim=1)

    hard_good_idx = torch.topk(
        good_to_mal,
        k=min(hard_k, len(good_embs))
    ).indices

    hard_good_paths = good_paths[hard_good_idx.cpu().numpy()]

    # malware samples closest to good center
    mal_to_good = torch.norm(mal_embs - good_center, dim=1)

    hard_mal_idx = torch.topk(
        mal_to_good,
        k=min(hard_k, len(mal_embs))
    ).indices

    hard_mal_paths = mal_paths[hard_mal_idx.cpu().numpy()]

    # ------------------------------------------------------------
    # 5. KMEANS STRUCTURE (GOODWARE)
    # ------------------------------------------------------------
    good_cpu = good_embs.cpu().numpy()

    kmeans_good = KMeans(
        n_clusters=min(num_centroids, len(good_cpu)),
        random_state=0,
        n_init="auto"
    ).fit(good_cpu)

    good_centroids = torch.tensor(
        kmeans_good.cluster_centers_,
        device=device
    )

    # pick centroids close to malware distribution
    dist_good_cent = torch.norm(good_centroids - mal_center, dim=1)

    good_cent_idx = torch.topk(
        -dist_good_cent,
        k=min(centroid_k, len(good_centroids))
    ).indices

    good_centroids_sel = good_centroids[good_cent_idx]

    good_centroid_paths = []
    for c in good_centroids_sel:
        d = torch.norm(good_embs - c.unsqueeze(0), dim=1)
        good_centroid_paths.append(
            good_paths[torch.argmin(d).item()]
        )

    # ------------------------------------------------------------
    # 6. KMEANS STRUCTURE (MALWARE)
    # ------------------------------------------------------------
    mal_cpu = mal_embs.cpu().numpy()

    kmeans_mal = KMeans(
        n_clusters=min(num_centroids, len(mal_cpu)),
        random_state=0,
        n_init="auto"
    ).fit(mal_cpu)

    mal_centroids = torch.tensor(
        kmeans_mal.cluster_centers_,
        device=device
    )

    dist_mal_cent = torch.norm(mal_centroids - good_center, dim=1)

    mal_cent_idx = torch.topk(
        -dist_mal_cent,
        k=min(centroid_k, len(mal_centroids))
    ).indices

    mal_centroids_sel = mal_centroids[mal_cent_idx]

    mal_centroid_paths = []
    for c in mal_centroids_sel:
        d = torch.norm(mal_embs - c.unsqueeze(0), dim=1)
        mal_centroid_paths.append(
            mal_paths[torch.argmin(d).item()]
        )

    # ------------------------------------------------------------
    # 7. MERGE RESULTS
    # ------------------------------------------------------------
    good_final = set(
        list(hard_good_paths) + good_centroid_paths
    )

    mal_final = set(
        list(hard_mal_paths) + mal_centroid_paths
    )

    return {
        "goodware_representatives": list(good_final),
        "malware_representatives": list(mal_final)
    }


# =========================================================
# FINETUNING FUNCTIONS
# =========================================================
def finetune_model(
        model,
        synthetic_embeddings,
        train_loader,
        device="cuda",
        epochs=10,
        lr=1e-4,
        lambda_synth=2.0
):

    model = model.to(device)
    model.train()

    # freeze batchnorm
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()

    ce_loss_fn = nn.CrossEntropyLoss()

    # freeze everything except head
    for p in model.parameters():
        p.requires_grad = False

    for p in model.fc_2.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.fc_2.parameters(), lr=lr)
    bs_syn = 64

    for epoch in range(epochs):
        total_loss = 0

        for inputs, labels, _, _ in tqdm(train_loader, desc="Fine-tuning"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # -------------------------------------------------
            # real data loss
            # -------------------------------------------------
            logits, _, _ = model(inputs)
            ce_real = ce_loss_fn(logits, labels)

            # -------------------------------------------------
            # synthetic batch
            # -------------------------------------------------
            idx = torch.randint(0, synthetic_embeddings.size(0), (bs_syn,), device=device)
            S = synthetic_embeddings.index_select(0, idx)

            logits_syn = model.classify_from_penult(S)

            # FORCE synthetic → malware
            label_syn = torch.ones(S.size(0), dtype=torch.long, device=device)
            ce_syn = ce_loss_fn(logits_syn, label_syn)

            # -------------------------------------------------
            # final loss
            # -------------------------------------------------
            loss = ce_real + lambda_synth*ce_syn

            # -------------------------------------------------
            # backprop and optimize
            # -------------------------------------------------
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    return model


device = "cuda"
target_family = 'family1'  # Replace with the desired target family

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
)

checkpoint = torch.load(model_path, map_location=device)
malconv_model.load_state_dict(checkpoint["model_state_dict"])

# -----------------------
# Load synthetic embeddings
# -----------------------
quantity = 50
synthetic_embeddings_path = os.path.join('../models', f"synthetic_embeddings_{target_family}.csv")
synthetic_embeddings = load_synthetic_embeddings(synthetic_embeddings_path, quantity, device)

# -----------------------
# select representative training malware and goodware samples (hard negatives + centroid representatives)
# ----------------------
loader_malware = DataLoader(
    SHADataset(
        csv_path='../data/real_data/training_malware.csv',
        source_dir="../data/real_data/malware/",
        sha_column="sha",
        isGoodware=False,
        max_len=16000000
    ),
    batch_size=4,
    shuffle=False, #keep False to maintain order for SHA-family mapping
    num_workers=2,
    pin_memory=True,
    collate_fn=pad_collate_func
)

loader_goodware = DataLoader(
    SHADataset(
        csv_path='../data/real_data/training_goodware.csv',
        source_dir="../data/real_data/goodware/",
        sha_column="sha",
        isGoodware=True,
        max_len=16000000
    ),
    batch_size=4,
    shuffle=False,  # keep False to maintain order for SHA-family mapping
    num_workers=2,
    pin_memory=True,
    collate_fn=pad_collate_func
)

representative_samples = select_symmetric_representatives(
    model=malconv_model,
    good_loader=loader_goodware,
    malware_loader=loader_malware,
    device=device,
    hard_k=1000,  # top 1000 hard malware representatives (closest to target family centroid)
    centroid_k=1000  # top 1000 centroids representing global malware structure (excluding target family)
)

# -----------------------
# Fine-tune model
# -----------------------
train_loader = build_train_loader(target_family, representative_samples)

model = finetune_model(
    model=malconv_model,
    train_loader=train_loader,
    synthetic_embeddings=synthetic_embeddings,
    device=device,
    epochs=10,
    lambda_synth=2.0,
)

# -----------------------
# Save model
# -----------------------

model_path = os.path.join('../models', target_family + ".checkpoint")
model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

torch.save({
    'epoch': 10,
    'model_state_dict': model_state,
    'channels': 128,
    'filter_size': 256,
    'stride': 64,
    'embd_dim': 8,
    'non_neg': False,
}, model_path)