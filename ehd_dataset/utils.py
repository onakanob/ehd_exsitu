import numpy as np
from scipy.stats import pearsonr

def correlate_dfs(A, A_key, B, B_key, offset):
    Aidx = A.index
    Bidx = B.index + offset
    Bidx = Bidx[np.isin(Bidx, B.index)]

    Aidx = Aidx[np.isin(Aidx, Bidx - offset)]
    Bidx = Bidx[np.isin(Bidx, Aidx + offset)]

    if len(Aidx) < 2:
        corr = None
    else:
        if A_key:
            A = A[A_key]
        if B_key:
            B = B[B_key]
        corr, _ = pearsonr(A.loc[Aidx], B.loc[Bidx])
    return corr, Aidx, Bidx


def cell_to_array(string):
    split_string = string[1:-1].split()
    return np.array(split_string).astype(float)


def parse_volt_strings(string):
    string = string[:-1]
    return float(string)
