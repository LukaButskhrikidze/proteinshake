import os
import requests
from proteinshake.utils import download_url, unzip_file, progressbar
from joblib import Parallel, delayed

class ZenodoProteinDataset(Dataset):
    def __init__(self, zenodo_id=7711953, **kwargs):
        self.zenodo_id = zenodo_id
        super().__init__(**kwargs)

    def get_raw_files(self):
        return glob.glob(f'{self.root}/raw/files/*.pdb')

    def get_id_from_filename(self, filename):
        return os.path.basename(filename).split('.')[0].lower()

    def download(self):
        os.makedirs(f'{self.root}/raw/files', exist_ok=True)

        zenodo_api = f'https://zenodo.org/api/records/{self.zenodo_id}'
        r = requests.get(zenodo_api)
        record = r.json()

        # Filter for protein files (e.g., .pdb or .pdb.gz)
        files = [f for f in record['files'] if f['key'].endswith('.pdb') or f['key'].endswith('.pdb.gz')]

        # Optional: apply self.limit for testing
        if self.limit:
            files = files[:self.limit]

        def fetch_file(f):
            url = f['links']['self']
            out_path = os.path.join(self.root, 'raw', 'files', os.path.basename(f['key']))
            download_url(url, os.path.dirname(out_path), verbosity=self.verbosity)
            if out_path.endswith('.gz'):
                unzip_file(out_path)

        Parallel(n_jobs=self.n_jobs)(delayed(fetch_file)(f) for f in progressbar(files, desc='Downloading proteins from Zenodo', verbosity=self.verbosity))
