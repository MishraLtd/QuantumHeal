"""
QuantumHealth Sentinel — CNN Embedding  (Phase 2, item: real CNN embedding)
=============================================================================
Replaces Phase 1's handcrafted radiomics/HOG image features with a small,
trainable convolutional network whose penultimate layer serves as the
imaging embedding fed into the imaging VQC arm.

Honest scope note: a pretrained ImageNet backbone (ResNet/DenseNet, as
scoped in the original Phase 2 plan) needs pretrained weights, and this
sandboxed environment cannot reach the hosts that serve them
(download.pytorch.org / torchvision's model zoo), the same restriction
that blocks a real chest-X-ray dataset (see imaging_arm.py — confirmed
blocked at zenodo.org). So this ships a small CNN trained **from scratch**
on whatever imaging data is passed in (synthetic for now) — a real learned
embedding, just not a pretrained one. The moment real images or a
reachable pretrained checkpoint are available, only this file's
`SmallCNN` needs swapping for a torchvision backbone; the embedding
interface (`fit_transform`/`transform`) and everything downstream
(quantum arm, fusion) stay the same.
"""

import numpy as np
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Tiny conv net: a few conv+pool blocks down to a low-dim, tanh-bounded
    embedding (bounded so it scales cleanly onto quantum rotation angles),
    plus a linear classification head used only to supervise training."""

    def __init__(self, embed_dim=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64 -> 32
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.embed = nn.Linear(16 * 4 * 4, embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        h = self.features(x).flatten(1)
        z = torch.tanh(self.embed(h))       # embedding, bounded to [-1, 1]
        logit = self.classifier(z)
        return z, logit


class CNNFeaturePipeline:
    """Same fit_transform/transform interface as ImageFeaturePipeline
    (imaging_arm.py), so it's a drop-in swap in train_multimodal.py /
    dashboard.py — trains the CNN once on the training split, then reuses
    its (frozen) embedding layer at inference time."""

    def __init__(self, embed_dim=4, epochs=25, lr=1e-3, seed=42):
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.model = None

    def fit_transform(self, images, labels):
        torch.manual_seed(self.seed)
        self.model = SmallCNN(embed_dim=self.embed_dim)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        X = torch.tensor(np.asarray(images), dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(np.asarray(labels), dtype=torch.float32).unsqueeze(1)

        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            _, logit = self.model(X)
            loss = loss_fn(logit, y)
            loss.backward()
            opt.step()
        return self._embed(images)

    def transform(self, images):
        return self._embed(images)

    def _embed(self, images):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(np.asarray(images), dtype=torch.float32).unsqueeze(1)
            z, _ = self.model(X)
        return z.numpy() * np.pi  # tanh output in [-1,1] -> quantum-angle range
