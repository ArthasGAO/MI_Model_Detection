from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset

class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 wrapper with YAML-driven transform pipelines.

    - train_set: uses train_transforms (aug)
    - test_set: uses test_transforms (clean)
    - in_sample_set: train data but uses test_transforms (clean probing)

    Transform specs:
      {"name": "RandomCrop", "params": {"size": 32, "padding": 4}, "enabled": True}
    """

    def __init__(
        self,
        normalization: str = "cifar10",
        loading: str = "torchvision",
        root_dir: str = "./data",
        img_size: int = 32,
        train_transforms: Optional[List[Dict[str, Any]]] = None,
        test_transforms: Optional[List[Dict[str, Any]]] = None,
        build_dataset: bool = True,
        download: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.loading = loading
        self.normalization = normalization
        self.img_size = int(img_size)
        self.download = download

        stats = self.set_normalization(normalization)
        self.mean = stats["mean"]
        self.std = stats["std"]

        # Build transform pipelines (YAML-driven or defaults)
        self.train_transform = self._build_transform_pipeline(train_transforms, is_train=True)
        self.test_transform = self._build_transform_pipeline(test_transforms, is_train=False)

        self.train_set = None
        self.test_set = None
        self.in_sample_set = None

        if build_dataset:
            self._build_datasets()

    # ---------------------------
    # Build datasets
    # ---------------------------
    def _build_datasets(self):
        self.train_set = self.get_dataset(train=True, transform=self.train_transform, download=self.download)
        self.test_set = self.get_dataset(train=False, transform=self.test_transform, download=self.download)
        self.in_sample_set = self.get_dataset(train=True, transform=self.test_transform, download=self.download)

    # ---------------------------
    # Normalization
    # ---------------------------
    def set_normalization(self, normalization: str):
        # CIFAR-10 stats
        if normalization == "cifar10":
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        elif normalization == "imagenet":
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            raise NotImplementedError(f"Unknown normalization: {normalization}")
        return {"mean": mean, "std": std}

    # ---------------------------
    # Transform factory
    # ---------------------------
    def _build_transform_pipeline(self, specs: Optional[List[Dict[str, Any]]], is_train: bool):
        """
        If specs is None or empty -> fallback to old behavior defaults.
        """
        if not specs:
            return self._default_transform(is_train=is_train)

        steps = []
        for spec in specs:
            if spec is None:
                continue
            enabled = spec.get("enabled", True)
            if not enabled:
                continue

            name = spec.get("name", None)
            if not name:
                raise ValueError(f"Transform spec missing 'name': {spec}")

            params = spec.get("params", {}) or {}
            steps.append(self._make_transform(name, params))

        return transforms.Compose(steps)

    def _default_transform(self, is_train: bool):
        if is_train:
            return transforms.Compose([
                transforms.RandomCrop(self.img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
        else:
            steps = []
            # If you ever set img_size != 32 (e.g., 224), make test pipeline explicit
            if self.img_size != 32:
                steps.append(transforms.Resize((self.img_size, self.img_size)))
            steps.extend([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
            return transforms.Compose(steps)

    def _make_transform(self, name: str, params: Dict[str, Any]):
        name = name.strip()

        # ---- common geometric ----
        if name == "Resize":
            size = params.get("size", self.img_size)
            if isinstance(size, int):
                size = (size, size)
            return transforms.Resize(size=size)

        if name == "RandomCrop":
            size = params.get("size", self.img_size)
            padding = params.get("padding", 0)
            pad_if_needed = params.get("pad_if_needed", False)
            padding_mode = params.get("padding_mode", "constant")
            return transforms.RandomCrop(size=size, padding=padding, pad_if_needed=pad_if_needed, padding_mode=padding_mode)

        if name == "RandomResizedCrop":
            size = params.get("size", self.img_size)
            scale = params.get("scale", (0.8, 1.0))
            ratio = params.get("ratio", (3/4, 4/3))
            interpolation = params.get("interpolation", transforms.InterpolationMode.BILINEAR)
            return transforms.RandomResizedCrop(size=size, scale=scale, ratio=ratio, interpolation=interpolation)

        if name == "RandomHorizontalFlip":
            p = params.get("p", 0.5)
            return transforms.RandomHorizontalFlip(p=p)

        if name == "ColorJitter":
            return transforms.ColorJitter(
                brightness=params.get("brightness", 0.0),
                contrast=params.get("contrast", 0.0),
                saturation=params.get("saturation", 0.0),
                hue=params.get("hue", 0.0),
            )

        # ---- policy-based ----
        if name == "RandAugment":
            num_ops = params.get("num_ops", 2)
            magnitude = params.get("magnitude", 9)
            return transforms.RandAugment(num_ops=num_ops, magnitude=magnitude)

        if name == "AutoAugment":
            policy = params.get("policy", "CIFAR10")
            policy_enum = getattr(transforms.AutoAugmentPolicy, policy)
            return transforms.AutoAugment(policy_enum)

        # ---- tensor + normalize ----
        if name == "ToTensor":
            return transforms.ToTensor()

        if name == "Normalize":
            mean = params.get("mean", self.mean)
            std = params.get("std", self.std)
            return transforms.Normalize(mean=mean, std=std)

        # ---- tensor-level ----
        if name == "RandomErasing":
            return transforms.RandomErasing(
                p=params.get("p", 0.25),
                scale=params.get("scale", (0.02, 0.33)),
                ratio=params.get("ratio", (0.3, 3.3)),
                value=params.get("value", 0),
                inplace=params.get("inplace", False),
            )

        raise NotImplementedError(f"Unknown transform name: '{name}'. Add it in _make_transform().")

    # ---------------------------
    # Dataset retrieval
    # ---------------------------
    def get_dataset(self, train: bool, transform, download: bool = True):
        if self.loading == "torchvision":
            return CIFAR10(
                root=self.root_dir,
                train=train,
                transform=transform,
                download=download,
            )
        elif self.loading == "custom":
            raise NotImplementedError("Custom CIFAR-10 loader not implemented.")
        else:
            raise NotImplementedError(f"Unknown loading mode: {self.loading}")
        
    def subset(self, split: str, indices, clean: bool = False):
        """
        Create a torch.utils.data.Subset from one of the internal datasets.

        split: "train" or "test"
        clean:
          - if split == "train": clean=False -> train_set (aug), clean=True -> in_sample_set (clean probing)
          - if split == "test": clean ignored -> test_set
        """
        split = split.lower().strip()
        if split == "train":
            base = self.in_sample_set if clean else self.train_set
        elif split == "test":
            base = self.test_set
        else:
            raise ValueError(f"Unknown split: {split}")

        if base is None:
            raise ValueError(f"Base dataset is None for split={split}, clean={clean}")

        return Subset(base, indices)

    
import numpy as np
from PIL import Image

class NumpyImageDataset(Dataset):
    """
    CIFAR-compatible dataset backed by numpy arrays.
    """
    def __init__(self, X, Y, transform=None):
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert X.shape[0] == Y.shape[0]

        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]  # (H, W, C), uint8
        label = int(self.Y[idx])

        # Convert numpy array to PIL Image (torchvision-native behavior)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label
    
    
from typing import Any, Dict, List, Optional
import numpy as np

class CIFAR10PseudoLabelDataset(CIFAR10Dataset):
    """
    CIFAR-10 wrapper for pseudo-labeled numpy data that matches CIFAR10Dataset interface.

    Train Attribute (train_set): 500,000 images (Scraped from 80M Tiny Images and labeled by a teacher model).
    Test Attribute (test_set): Does not exist (Size: 0).
    
    Workflow:
      ds = CIFAR10PseudoLabelDataset(...)
      ds.set_data(X, Y)  # builds train_set + in_sample_set (clean probing)
    """

    def __init__(
        self,
        normalization: str = "cifar10",
        img_size: int = 32,
        train_transforms: Optional[List[Dict[str, Any]]] = None,
        test_transforms: Optional[List[Dict[str, Any]]] = None,
    ):
        # Build transforms + stats only; DO NOT build torchvision datasets
        super().__init__(
            normalization=normalization,
            loading="custom",          # <- important (we won't call torchvision CIFAR10)
            root_dir="./data",         # unused but harmless
            img_size=img_size,
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            build_dataset=False,
            download=False,
        )

        # Placeholder fields until set_data() is called
        self.X = None
        self.Y = None
        self.train_set = None
        self.test_set = None
        self.in_sample_set = None

    def set_data(self, X: np.ndarray, Y: np.ndarray):
        """
        Attach numpy arrays and build internal datasets using parent's transforms.

        X: (N,H,W,3) uint8
        Y: (N,)
        """
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise TypeError("X and Y must be numpy arrays.")
        if X.ndim != 4 or X.shape[-1] != 3:
            raise ValueError(f"X must have shape (N,H,W,3). Got {X.shape}.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have same length. Got {X.shape[0]} vs {Y.shape[0]}.")
        if X.dtype != np.uint8:
            raise ValueError(f"X must be uint8 (0-255) for PIL. Got {X.dtype}.")

        self.X = X
        self.Y = Y

        # Build datasets consistent with CIFAR10Dataset semantics
        self.train_set = NumpyImageDataset(X=self.X, Y=self.Y, transform=self.train_transform)
        self.in_sample_set = NumpyImageDataset(X=self.X, Y=self.Y, transform=self.test_transform)

        # Pseudo-labeled data typically does not have a true test split
        self.test_set = None

        return self



from datasets import load_dataset
from PIL import Image

class HFImageDataset(Dataset):
    """
    A PyTorch Dataset wrapper for Hugging Face datasets.
    Extracts the PIL image and label, forces a resize to match CIFAR-10, 
    and applies your YAML transform pipeline.
    """
    def __init__(self, hf_dataset, transform=None, force_size=32):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.force_size = force_size

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Hugging Face yields dictionaries
        item = self.hf_dataset[idx]
        
        # Dynamically handle the column name (usually 'img' for CIFAR)
        img_key = 'img' if 'img' in item else 'image'
        
        # Ensure it's a 3-channel RGB PIL Image
        img = item[img_key].convert("RGB")
        label = int(item['label'])

        # Force resize from CIFARNet (64x64) down to CIFAR-10 (32x32)
        if self.force_size and img.size != (self.force_size, self.force_size):
            img = img.resize((self.force_size, self.force_size), Image.BILINEAR)

        # Apply your parent class's transform pipeline
        if self.transform is not None:
            img = self.transform(img)

        return img, label


class CIFARNetDataset(CIFAR10Dataset):
    """
    CIFAR-10 wrapper for the Hugging Face CIFARNet (ImageNet-derived) dataset.
    Perfect for Out-of-Distribution (OOD) Model Stealing.

    Train Attribute (train_set): 190,000 images (Sampled from ImageNet).
    Test Attribute (test_set): 10,000 images.
    
    """
    def __init__(
        self,
        normalization: str = "cifar10",  # Keep CIFAR-10 stats for the victim model!
        img_size: int = 32,
        train_transforms: Optional[List[Dict[str, Any]]] = None,
        test_transforms: Optional[List[Dict[str, Any]]] = None,
    ):
        # 1. Initialize parent, skip torchvision building
        super().__init__(
            normalization=normalization,
            loading="custom",
            root_dir="./data",
            img_size=img_size,
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            build_dataset=False,
            download=False,
        )

        # 2. Download/Load the Hugging Face dataset automatically
        print("[INFO] Loading CIFARNet from Hugging Face...")
        hf_data = load_dataset("EleutherAI/cifarnet")

        # 3. Build internal sets using the HF wrapper
        # We pass self.img_size to ensure the 64->32 downsampling happens
        self.train_set = HFImageDataset(
            hf_data['train'], 
            transform=self.train_transform, 
            force_size=self.img_size
        ) # size - 190000
        
        self.test_set = HFImageDataset(
            hf_data['test'], 
            transform=self.test_transform, 
            force_size=self.img_size
        ) # size - 10000
        
        self.in_sample_set = HFImageDataset(
            hf_data['train'], 
            transform=self.test_transform, 
            force_size=self.img_size
        )
