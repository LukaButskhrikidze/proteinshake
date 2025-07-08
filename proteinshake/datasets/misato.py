import os
import glob
import numpy as np
import requests
from joblib import Parallel, delayed
from proteinshake.datasets import Dataset
from proteinshake.utils import download_url, unzip_file, progressbar

class ZenodoProteinDataset(Dataset):
    def __init__(self, zenodo_id=7711953, **kwargs):
        self.zenodo_id = zenodo_id
        super().__init__(**kwargs)

    def get_raw_files(self):
        return glob.glob(f'{self.root}/raw/files/*.npy')

    def get_id_from_filename(self, filename):
        return os.path.basename(filename).split('.')[0].lower()

    def download(self):
        os.makedirs(f'{self.root}/raw/files', exist_ok=True)
        print(f"Querying Zenodo record {self.zenodo_id}...")

        r = requests.get(f'https://zenodo.org/api/records/{self.zenodo_id}')
        r.raise_for_status()
        record = r.json()

        files = [
            f for f in record['files']
            if f['key'].endswith('.npy')
        ]

        if self.limit:
            files = files[:self.limit]

        def fetch(f):
            url = f['links']['self']
            download_url(url, f"{self.root}/raw/files", verbosity=self.verbosity)

        Parallel(n_jobs=self.n_jobs)(delayed(fetch)(f) for f in progressbar(files, desc="Downloading .npy files", verbosity=self.verbosity))

    def parse_pdb(self, path):
        try:
            pdbid = self.get_id_from_filename(path)
            coords = np.load(path)  # shape: [N, 3]
            n_atoms = coords.shape[0]

            protein = {
                'protein': {
                    'ID': pdbid,
                    'sequence': 'X' * n_atoms
                },
                'atom': {
                    'atom_number': list(range(n_atoms)),
                    'atom_type': ['C'] * n_atoms,
                    'residue_number': list(range(n_atoms)),
                    'residue_type': ['X'] * n_atoms,
                    'x': coords[:, 0].tolist(),
                    'y': coords[:, 1].tolist(),
                    'z': coords[:, 2].tolist(),
                    'SASA': [-1] * n_atoms
                },
                'residue': {
                    'residue_number': list(range(n_atoms)),
                    'residue_type': ['X'] * n_atoms,
                    'x': coords[:, 0].tolist(),
                    'y': coords[:, 1].tolist(),
                    'z': coords[:, 2].tolist(),
                    'SASA': [-1] * n_atoms,
                    'RSA': [-1] * n_atoms
                }
            }

            return protein
        except Exception as e:
            if self.verbosity > 0:
                print(f"Failed to parse {path}: {e}")
            return None
