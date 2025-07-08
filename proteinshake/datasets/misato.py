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
        return glob.glob(f'{self.root}/raw/files/*.pdb')

    def get_id_from_filename(self, filename):
        return os.path.basename(filename).split('.')[0].lower()

    def download(self):
        os.makedirs(f'{self.root}/raw/files', exist_ok=True)

        if self.verbosity > 0:
            print(f"Querying Zenodo record {self.zenodo_id}...")

        zenodo_api = f'https://zenodo.org/api/records/{self.zenodo_id}'
        r = requests.get(zenodo_api)
        r.raise_for_status()
        record = r.json()

        files = [
            f for f in record['files']
            if f['key'].endswith('.pdb') or f['key'].endswith('.pdb.gz')
        ]

        if self.limit:
            files = files[:self.limit]

        def fetch_file(f):
            url = f['links']['self']
            out_path = os.path.join(self.root, 'raw', 'files', os.path.basename(f['key']))
            download_url(url, os.path.dirname(out_path), verbosity=self.verbosity)
            if out_path.endswith('.gz'):
                unzip_file(out_path)

        Parallel(n_jobs=self.n_jobs)(
            delayed(fetch_file)(f) for f in progressbar(files, desc='Downloading proteins from Zenodo', verbosity=self.verbosity)
        )
