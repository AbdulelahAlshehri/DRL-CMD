from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs


def diversity_score(mol, target, radius=2, nBits=2048,
                    useChirality=True):
    """
    Average pairwise Tanimoto distance between the Morgan fingerprints of the molecules
    """

    x = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius,
                                                       nBits=nBits,
                                                       useChirality=useChirality)
    target = rdMolDescriptors.GetMorganFingerprintAsBitVect(target,
                                                            radius=radius,
                                                            nBits=nBits,
                                                            useChirality=useChirality)
    return DataStructs.TanimotoSimilarity(x, target)
