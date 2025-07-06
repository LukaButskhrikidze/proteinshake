import os
import h5py
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

from proteinshake.datasets import Dataset

class MisatoDatasetMD(Dataset):
    """
    MISATO (MD-only) dataset integration for ProteinShake.

    Downloads and loads MD.hdf5 from Zenodo, provides ProteinShake-style dict output.
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
        self.proteins: List[str] = []
        self.md_data = None
        self._subset_indices: List[int] = []

        if download or force_download:
            self._download_md(force=force_download)

        self._load_md()
        self._setup_subset()

        if self.verbosity > 0:
            print(f"✔️ Loaded MISATO MD dataset with {len(self)} proteins")

    def _download_md(self, force: bool = False):
        zenodo_url = (
            "https://zenodo.org/record/7711953/files/MD.hdf5?download=1"
        )
        expected_md5 = "c9b6efb6d73d3d2d15a2b8591802b58a"  # Verify if matches actual
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        if self.md_file.exists() and not force:
            if self.verbosity > 0:
                print("MD.hdf5 already exists – skipping download.")
            return

        r = requests.get(zenodo_url, stream=True)
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        md5 = hashlib.md5()
        with open(self.md_file, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading MD.hdf5") as p:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
                    md5.update(chunk)
                    p.update(len(chunk))

        if expected_md5 and md5.hexdigest() != expected_md5:
            raise ValueError("❗ Checksum mismatch for MD.hdf5")

    def _load_md(self):
        if not self.md_file.exists():
            raise FileNotFoundError("MD.hdf5 not found – please download first.")
        self.md_data = h5py.File(self.md_file, "r")
        self.proteins = sorted(self.md_data.keys())

    def _setup_subset(self):
        total = len(self.proteins)
        splits = {
            'all': (0, total),
            'train': (0, int(0.8 * total)),
            'val': (int(0.8 * total), int(0.9 * total)),
            'test': (int(0.9 * total), total)
        }
        if self.subset not in splits:
            raise ValueError(f"Unknown subset `{self.subset}`")
        start, end = splits[self.subset]
        self._subset_indices = list(range(start, end))
        if self.verbosity > 0:
            print(f"Subset '{self.subset}' contains {len(self._subset_indices)} proteins")

    def __len__(self) -> int:
        return len(self._subset_indices)

    def __getitem__(self, idx: int) -> Dict:
        if idx < 0 or idx >= len(self):
            raise IndexError("Protein index out of range")
        pid = self.proteins[self._subset_indices[idx]]
        grp = self.md_data[pid]

        coords = grp['coords'][:]  # shape [N_atoms, 3]
        atom_types = grp.get('atom_types')[:] if 'atom_types' in grp else grp.get('elements')[:]

        return {
            'protein': {'ID': pid},
            'atom': {
                'x': coords[:, 0].tolist(),
                'y': coords[:, 1].tolist(),
                'z': coords[:, 2].tolist(),
                'atom_type': [t.decode() if isinstance(t, bytes) else str(t) for t in atom_types]
            }
        }

    def get_protein_ids(self) -> List[str]:
        return [self.proteins[i] for i in self._subset_indices]

    def close(self):
        if self.md_data:
            self.md_data.close()

    def __del__(self):
        self.close()
