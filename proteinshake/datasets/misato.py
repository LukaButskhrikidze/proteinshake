import os
import h5py
import hashlib
import warnings
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Union

from proteinshake.datasets import Dataset
from proteinshake.utils.protein import Protein


class MisatoDataset(Dataset):
    """
    MISATO Dataset: Proteins with identical sequence but different conformations (MD only).

    Downloads and loads the `MD.hdf5` file from Zenodo.
    Converts it into ProteinShake format for benchmarking.
    """

    def __init__(
        self,
        root: str = 'data/misato',
        subset: str = 'train',
        download: bool = True,
        force_download: bool = False,
        verbosity: int = 1,
        **kwargs
    ):
        super().__init__(root=root, **kwargs)
        self.root = Path(root)
        self.raw_dir = self.root / 'raw'
        self.md_file = self.raw_dir / 'MD.hdf5'
        self.verbosity = verbosity
        self.subset = subset
        self.proteins = []
        self.md_data = None
        self._subset_indices = []

        if download or force_download:
            self._download_md(force=force_download)

        self._load_md()
        self._setup_subset()

        if self.verbosity > 0:
            print(f"✓ Loaded MISATO (MD-only) with {len(self)} proteins")

    def _download_md(self, force: bool = False):
        zenodo_url = (
            "https://zenodo.org/record/7711953/files/MD.hdf5?download=1"
        )
        expected_md5 = "c9b6efb6d73d3d2d15a2b8591802b58a"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        if self.md_file.exists() and not force:
            if self.verbosity > 0:
                print("MD.hdf5 already exists, skipping download")
            return

        response = requests.get(zenodo_url, stream=True)
        response.raise_for_status()
        md5_hash = hashlib.md5()
        total_size = int(response.headers.get("content-length", 0))
        with open(self.md_file, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading MD.hdf5"
        ) as pbar:
            for chunk in response.iter_content(8192):
                if chunk:
                    f.write(chunk)
                    md5_hash.update(chunk)
                    pbar.update(len(chunk))

        if expected_md5 and md5_hash.hexdigest() != expected_md5:
            raise ValueError("Checksum mismatch for MD.hdf5")

    def _load_md(self):
        if not self.md_file.exists():
            raise FileNotFoundError("MD.hdf5 not found")

        self.md_data = h5py.File(self.md_file, "r")
        self.proteins = list(self.md_data.keys())

    def _setup_subset(self):
        n = len(self.proteins)
        if self.subset == 'all':
            self._subset_indices = list(range(n))
        elif self.subset == 'train':
            self._subset_indices = list(range(int(0.8 * n)))
        elif self.subset == 'val':
            self._subset_indices = list(range(int(0.8 * n), int(0.9 * n)))
        elif self.subset == 'test':
            self._subset_indices = list(range(int(0.9 * n), n))
        else:
            raise ValueError(f"Invalid subset: {self.subset}")

        if self.verbosity > 0:
            print(f"Subset '{self.subset}' contains {len(self._subset_indices)} proteins")

    def __len__(self) -> int:
        return len(self._subset_indices)

    def __getitem__(self, idx: int) -> Protein:
        if idx >= len(self):
            raise IndexError("Index out of bounds")
        protein_id = self.proteins[self._subset_indices[idx]]
        group = self.md_data[protein_id]

        coords = group['coords'][:]
        atom_types = group.get('atom_types')[:] if 'atom_types' in group else None
        if atom_types is None and 'elements' in group:
            atom_types = group['elements'][:]

        protein = Protein(
            coords=coords,
            atom_types=atom_types,
            protein_id=protein_id
        )
        protein.metadata = {
            'protein_id': protein_id,
            'source': 'MISATO-MD'
        }
        return protein

    def get_protein_ids(self) -> List[str]:
        return [self.proteins[i] for i in self._subset_indices]

    def close(self):
        if self.md_data:
            self.md_data.close()

    def __del__(self):
        self.close()
