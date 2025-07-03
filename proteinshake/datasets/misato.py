import requests, glob, json, os, tarfile
import pandas as pd
from joblib import Parallel, delayed

from proteinshake.datasets import Dataset
from proteinshake.utils import download_url, unzip_file, error, warning, progressbar


class MisatoProteinLigandDataset(Dataset):
    """ Protein-ligand complexes from the MISATO dataset.
    
    MISATO is a comprehensive dataset of protein-ligand interactions with molecular dynamics simulations
    and quantum mechanical calculations. This dataset contains protein-ligand complexes with their
    corresponding binding affinity data.
    
    .. admonition:: Please cite
    
        Skalic, M., Sabbadin, D., Sattarov, B., Sciabola, S., & De Fabritiis, G. (2023). 
        MISATO: machine learning dataset of protein-ligand complexes for structure-based drug discovery.
        
    .. admonition:: Source
    
        Raw data was obtained from `Zenodo MISATO dataset <https://zenodo.org/records/7711953>`_.
    
    Parameters
    ----------
    subset: str, default 'train'
        Which subset of the MISATO dataset to use ('train', 'test', or 'val').
    include_md: bool, default False
        Whether to include molecular dynamics trajectory data.
    """

    def __init__(self, subset='train', include_md=False, **kwargs):
        self.subset = subset
        self.include_md = include_md
        self.zenodo_base_url = 'https://zenodo.org/records/7711953/files'
        super().__init__(**kwargs)

    def get_raw_files(self):
        """Returns list of PDB files from the downloaded MISATO dataset."""
        return glob.glob(f'{self.root}/raw/files/**/*.pdb', recursive=True)

    def get_id_from_filename(self, filename):
        """Extract protein ID from filename."""
        # MISATO files typically have format like: protein_id_ligand_id.pdb
        basename = os.path.basename(filename)
        return basename.replace('.pdb', '').split('_')[0]

    def download(self):
        """Downloads the MISATO dataset from Zenodo."""
        
        # Define the files to download based on subset
        files_to_download = []
        
        if self.subset == 'train':
            files_to_download.extend([
                'train_set.tar.gz',
                'train_set_no_lig.tar.gz'
            ])
        elif self.subset == 'test':
            files_to_download.extend([
                'test_set.tar.gz',
                'test_set_no_lig.tar.gz'
            ])
        elif self.subset == 'val':
            files_to_download.extend([
                'val_set.tar.gz',
                'val_set_no_lig.tar.gz'
            ])
        else:
            error(f'Unknown subset: {self.subset}. Must be one of: train, test, val', 
                  verbosity=self.verbosity)
            return

        # Download MD trajectories if requested
        if self.include_md:
            files_to_download.extend([
                f'MD_{self.subset}_set.tar.gz',
                f'QM_{self.subset}_set.tar.gz'
            ])

        # Download metadata
        files_to_download.extend([
            'misato_db.csv',
            'README.md'
        ])

        # Download files
        for filename in progressbar(files_to_download, desc='Downloading MISATO files', verbosity=self.verbosity):
            try:
                download_url(f'{self.zenodo_base_url}/{filename}', 
                           f'{self.root}/raw/', 
                           verbosity=self.verbosity)
            except Exception as e:
                warning(f'Failed to download {filename}: {e}', verbosity=self.verbosity)

        # Extract tar.gz files
        for filename in files_to_download:
            if filename.endswith('.tar.gz'):
                tar_path = f'{self.root}/raw/{filename}'
                if os.path.exists(tar_path):
                    self._extract_tar_gz(tar_path)

    def _extract_tar_gz(self, tar_path):
        """Extract tar.gz file to the files directory."""
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=f'{self.root}/raw/files/')
            if self.verbosity > 1:
                print(f'Extracted {os.path.basename(tar_path)}')
        except Exception as e:
            warning(f'Failed to extract {tar_path}: {e}', verbosity=self.verbosity)

    def add_protein_attributes(self, protein_dict):
        """Add MISATO-specific attributes to protein dictionary."""
        # Load metadata if available
        metadata_path = f'{self.root}/raw/misato_db.csv'
        if os.path.exists(metadata_path):
            try:
                metadata = pd.read_csv(metadata_path)
                protein_id = protein_dict['protein']['ID']
                
                # Find matching entry in metadata
                matching_rows = metadata[metadata['ID'].str.contains(protein_id, na=False)]
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    
                    # Add binding affinity if available
                    if 'affinity' in row:
                        protein_dict['protein']['binding_affinity'] = row['affinity']
                    
                    # Add ligand information if available
                    if 'ligand_smiles' in row:
                        protein_dict['protein']['ligand_smiles'] = row['ligand_smiles']
                    
                    # Add other available metadata
                    for col in ['resolution', 'r_work', 'r_free', 'method']:
                        if col in row and pd.notna(row[col]):
                            protein_dict['protein'][col] = row[col]
                            
            except Exception as e:
                warning(f'Failed to load metadata for {protein_dict["protein"]["ID"]}: {e}', 
                       verbosity=self.verbosity)
        
        return protein_dict

    def get_ligand_files(self):
        """Get ligand structure files if available."""
        return glob.glob(f'{self.root}/raw/files/**/*.sdf', recursive=True)

    def get_md_files(self):
        """Get molecular dynamics trajectory files if available."""
        if not self.include_md:
            return []
        return glob.glob(f'{self.root}/raw/files/**/*.xtc', recursive=True)
        
