"""
QuantumHealth Sentinel — Imaging Arm  (Phase 1: Multimodal Expansion)
=======================================================================
Adds a second data modality (chest-X-ray-style imagery) alongside the
existing tabular biomarker arm. Each modality gets classically-extracted,
low-dimensional features that are then quantum-encoded and classified by
its own small VQC (see hybrid_qml_mvp.build_vqc_arm), and the two arms'
outputs are combined by fusion.py.

DATASET NOTE (read this before swapping in real data):
This sandbox cannot reach medical-imaging hosts (NIH ChestX-ray14,
Kaggle, MedMNIST's Zenodo mirror, etc.), so Phase 1 ships a *synthetic*
X-ray-style image generator: it procedurally paints a label-correlated
"opacity" pattern (a brighter, larger, more textured blob for
disease-positive cases; a small clean one for disease-negative cases)
onto a noisy grayscale field. This is a stand-in, not a claim of medical
realism — its only job is to prove the multimodal architecture (imaging
ingestion -> feature extraction -> quantum arm -> fusion) end-to-end
with a fully offline, instant demo, matching this MVP's existing
zero-download design philosophy.

Swapping in real data later touches ONLY `load_synthetic_xrays()` —
replace it with a loader over NIH ChestX-ray14 / a pneumonia dataset and
everything downstream (ImageFeaturePipeline, the imaging VQC arm,
fusion) is unchanged. That swap, plus replacing HOG with a pretrained
CNN embedding (ResNet/DenseNet), is the headline Phase 2 item.
"""

import numpy as np
from skimage.feature import hog
from skimage.filters import sobel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

IMG_SIZE = 64
N_QUBITS_IMG = 4


def load_synthetic_xrays(y, seed=42, img_size=IMG_SIZE):
    """Generate one synthetic grayscale image per label in `y`.

    Disease-positive (y=1): larger, brighter, more textured "opacity"
    blob (loosely mimicking a radiographic infiltrate).
    Disease-negative (y=0): small, faint blob on a cleaner field.
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    images = np.zeros((n, img_size, img_size), dtype=np.float32)
    yy, xx = np.mgrid[0:img_size, 0:img_size]

    center = img_size // 2
    for i, label in enumerate(y):
        base = rng.normal(loc=0.35, scale=0.05, size=(img_size, img_size))
        # Small positional jitter only — the label signal should live in the
        # opacity's SIZE/BRIGHTNESS/TEXTURE, not in where it happens to sit,
        # otherwise HOG's local-gradient-by-position features get swamped by
        # positional nuisance variance rather than the actual disease signal.
        cx, cy = center + rng.randint(-4, 5), center + rng.randint(-4, 5)
        if label == 1:
            radius = rng.uniform(10, 16)
            intensity = rng.uniform(0.35, 0.55)
            texture = rng.normal(0, 0.06, size=(img_size, img_size))
        else:
            radius = rng.uniform(3, 6)
            intensity = rng.uniform(0.05, 0.15)
            texture = rng.normal(0, 0.02, size=(img_size, img_size))
        blob = intensity * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius ** 2)))
        images[i] = np.clip(base + blob + texture, 0, 1)
    return images


class ImageFeaturePipeline:
    """Classical image -> low-dim feature vector, ready for quantum encoding.

    Phase 1 combines classic "radiomics-style" intensity/texture
    descriptors (mean/std/max intensity, thresholded lesion area, edge
    energy, histogram entropy) with a coarse HOG descriptor, as a
    deterministic, training-free stand-in for a learned CNN embedding —
    no GPU/training time needed, which matters for a live demo. Phase 2
    swaps `_raw_features` for a small pretrained CNN embedding; PCA +
    quantum-angle-scaling below stay identical either way.
    """

    def __init__(self, n_components=N_QUBITS_IMG):
        self.n_components = n_components
        self.scaler = None
        self.pca = None
        self.scale = None

    @staticmethod
    def _radiomic_stats(img):
        hist, _ = np.histogram(img, bins=16, range=(0, 1), density=True)
        hist = hist / (hist.sum() + 1e-9)
        entropy = -np.sum(hist * np.log(hist + 1e-9))
        edges = sobel(img)
        return np.array([
            img.mean(), img.std(), img.max(),
            (img > 0.3).sum() / img.size,   # thresholded "lesion" area fraction
            edges.mean(), edges.std(),      # edge energy (texture)
            entropy,
        ])

    @classmethod
    def _raw_features(cls, images):
        radiomic = np.array([cls._radiomic_stats(img) for img in images])
        hog_feats = np.array([
            hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
            for img in images
        ])
        return np.concatenate([radiomic, hog_feats], axis=1)

    def fit_transform(self, images):
        feats = self._raw_features(images)
        self.scaler = StandardScaler().fit(feats)
        feats_s = self.scaler.transform(feats)
        self.pca = PCA(n_components=self.n_components, random_state=42).fit(feats_s)
        feats_p = self.pca.transform(feats_s)
        self.scale = np.max(np.abs(feats_p)) + 1e-9
        return (feats_p / self.scale) * np.pi

    def transform(self, images):
        feats = self._raw_features(images)
        feats_s = self.scaler.transform(feats)
        feats_p = self.pca.transform(feats_s)
        return (feats_p / self.scale) * np.pi
