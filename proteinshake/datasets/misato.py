import os
import glob
import requests
from joblib import Parallel, delayed

from proteinshake.datasets import Dataset
from proteinshake.utils import download_url, unzip_file, progressbar


class ZenodoProteinDataset(Dataset):
    """
    Downloads PDB structures from Zenodo record 7711953 and parses them.

    Only files ending in `.pdb` or `.pdb.gz` are considered.
    """

    def __init__(self, zenodo_id=7711953, **kwargs):
        self.zenodo_id = zenodo_id
        super().__init__(**kwargs)

    def get_raw_files(self):
        return glob.glob(f'{self.root}/raw/files/*.npy')

    def get_id_from_filename(self, filename):
        return os.path.basename(filename).split('.')[0]

    def parse_pdb(self, path):
        try:
            pdbid = self.get_id_from_filename(path)

            coords = np.load(path)  # assume shape (N, 3)
            n_atoms = coords.shape[0]

            protein = {
                'protein': {
                    'ID': pdbid,
                    'sequence': 'X' * n_atoms  # dummy sequence
                },
                'atom': {
                    'atom_number': list(range(n_atoms)),
                    'atom_type': ['C'] * n_atoms,  # dummy atom type
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
