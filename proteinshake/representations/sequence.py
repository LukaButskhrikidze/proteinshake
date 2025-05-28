import os
from tqdm import tqdm
import numpy as np
import torch

from proteinshake.utils import tokenize


class Sequence:
    """ Linear sequence representation of a protein (1D amino acid string).

    Parameters
    ----------
    protein : dict
        A protein object containing the sequence under `protein['protein']['sequence']`.
    """

    def __init__(self, protein):
        self.protein_dict = protein
        self.sequence = protein['protein']['sequence']
        self.tokens = tokenize(self.sequence, resolution='residue')  # make it explicitly a NumPy array
        self.data = self.tokens  # compatibility with other representations


class SequenceDataset:
    """ Dataset of proteins represented as linear sequences.

    Parameters
    ----------
    proteins : generator
        A generator of protein objects.
    size : int
        Number of proteins in the dataset.
    path : str
        Path to save the processed dataset.
    """

    def __init__(self, proteins, root, name, verbosity=2):
        self.verbosity = verbosity
        self.path = f'{root}/processed/sequence/{name}'
        self.sequences = list(Sequence(protein) for protein in tqdm(proteins, total=len(proteins)))
        self.size = len(self.sequences)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            'sequence': torch.tensor(seq.tokens, dtype=torch.long),
            'id': idx,
        }

    def strings(self):
        """ Returns all sequences as plain amino acid strings. """
        return [seq.sequence for seq in self.sequences]

    def numpy(self):
        """ Returns all tokenized sequences as a list of NumPy arrays. """
        return [seq.tokens for seq in self.sequences]
