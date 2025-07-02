# -*- coding: utf-8 -*-
import os
import h5py
from proteinshake.datasets.dataset import Dataset
from proteinshake.utils import download_url, progressbar

class MisatoProteinLigandDataset(Dataset):
    """Protein-ligand dataset from the Misato MD.hdf5 file on Zenodo.
    
    The dataset stores atom and residue-level structural info for each protein-ligand complex.
    """
    name = "misato_protein_ligand"
    description = "Protein-ligand structures from Misato MD.hdf5"

    def get_raw_files(self):
        return [f"{self.root}/raw/MD.hdf5"]

    def download(self):
        url = "https://zenodo.org/record/7711953/files/MD.hdf5"
        out_path = os.path.join(self.root, 'raw', 'files', 'MD.hdf5')

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not os.path.exists(out_path):
        download_url(url, out_path, description="Downloading Misato MD dataset", verbosity=self.verbosity)

    
    if not os.path.exists(out_path):
        download_url(url, out_path, description="Downloading Misato MD dataset", verbosity=self.verbosity)


    def get_id_from_filename(self, filename):
        return filename.split(".")[0]

    def parse(self):
        path = f"{self.root}/raw/MD.hdf5"
        with h5py.File(path, "r") as f:
            for protein_id in progressbar(list(f.keys())[:self.limit], desc="Parsing Misato proteins", verbosity=self.verbosity):
                group = f[protein_id]
                protein = {
                    "protein": {
                        "ID": protein_id,
                    },
                    "residue": {
                        "residue_number": list(range(len(group["residue_coords"]))),
                        "x": group["residue_coords"][:, 0].tolist(),
                        "y": group["residue_coords"][:, 1].tolist(),
                        "z": group["residue_coords"][:, 2].tolist(),
                    },
                    "atom": {
                        "atom_number": list(range(len(group["atom_coords"]))),
                        "x": group["atom_coords"][:, 0].tolist(),
                        "y": group["atom_coords"][:, 1].tolist(),
                        "z": group["atom_coords"][:, 2].tolist(),
                        "atom_type": [
                            a.decode("utf-8") if isinstance(a, bytes) else str(a)
                            for a in group["atom_types"][:]
                        ],
                    }
                }
                self.save_protein(protein)
