import logging
import collections
import numpy as np
import re
from typing import Tuple, Dict

class FCGR():
    def __init__(self, sequence: str, k: int):

        self.sequence = sequence.strip().upper()
        self.k = k

        # check if the sequence is valid and does not contain any character apart from [ATGCN]
        if not self._is_valid_seq():
            raise ValueError("Invalid Sequence Passed")

    def k_mer_count(self) -> Dict[str, int]:
        k_mer_dict = collections.defaultdict(int)
        skipped_kmers = 0  # skip k_mer if it contains N

        for i in range(len(self.sequence) - (self.k - 1)):
            k_mer = self.sequence[i : i + self.k]

            if all(nt in "ATGC" for nt in k_mer):
                k_mer_dict[k_mer] += 1
            else:
                skipped_kmers += 1
        # Log skipped kmers if any
        if skipped_kmers > 0:
            logging.warning(f"Total number of Skipped kmers: {skipped_kmers}")
        return k_mer_dict

    def k_mer_frequency(self):
        k_mer_dict = self.k_mer_count()
        frequencies = collections.defaultdict(float)

        total_k_mers = len(self.sequence) - (self.k - 1)
        valid_k_mers = len(k_mer_dict)
        if valid_k_mers <= 0:
            raise ValueError("Sequence bypassed validation, has 0 valid k_mer")

        for key, value in k_mer_dict.items():
            frequencies[key] = value / total_k_mers  # not with valid k_mers (attention)
        return frequencies

    def kmer2coordinate(self, kmer: str) -> Tuple[int, int] | Tuple[None, None]:
        coordinates = {
            "A": (0, 0),
            "C": (1, 0),
            "G": (1, 1),
            "T": (0, 1)
        }

        x, y = 0.5, 0.5
        for nucleotide in kmer:
            if nucleotide not in coordinates:
                print(f"Invalid nucleotide: {nucleotide}")
                return None, None

            corner_x, corner_y = coordinates[nucleotide]
            x = (x + corner_x) / 2
            y = (y + corner_y) / 2

        size = 2 ** self.k
        matrix_x = int(x * size)
        matrix_y = int(y * size)

        matrix_x = min(matrix_x, size - 1)
        matrix_y = min(matrix_y, size - 1)

        return matrix_x, matrix_y

    def generate_fcgr(self) -> np.ndarray:
        size = 2 ** self.k
        fcgr = np.zeros((size, size), dtype=float)

        frequencies = self.k_mer_frequency()
        if not frequencies:
            raise ValueError("Couldn't get frequencies in generate_fcgr()")

        for kmer, freq in frequencies.items():
            x, y = self.kmer2coordinate(kmer)
            if x is not None and y is not None:
                fcgr[y, x] += freq

        return fcgr

    def _is_valid_seq(self) -> bool:
        if not self.sequence:
            return False
        if 'N' in self.sequence:
            logging.warning(f"Sequence Contains N: {self.sequence}")
            logging.warning(f"Total number of N in sequence: {self.sequence.count('N')}")
            logging.error(f"Sequence: {self.sequence}")
        return bool(re.fullmatch(r"[ATGCN]*", self.sequence))
    
    
if __name__ == "__main__":
   print("Checking FCGR for ATGCGGTCGT")
   print(FCGR("ATGCGGTCGT", 3).generate_fcgr())