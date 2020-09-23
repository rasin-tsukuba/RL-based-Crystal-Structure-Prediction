#
# calculation of synthetic accessibility score 
# as described in:
# 
# Estimation of Synthetic Accessibility Score of 
# Drug-like Molecules based on Molecular Complexity 
# and Fragment Contributions

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pickle
from rdkit.six import iteritems

import math

import os.path as op

_fscores = None

def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename
    if name == 'fpscores':
        name = op.join(op.dirname(__file__), name)

    _fscores = pickle.load(gzip.open('%s.pkl.gz' % name))
    
    outDict = {}
    for i in _fscores:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    
    # Returns the number of spiro atoms (atoms shared 
    # between rings that share exactly one atom)
    # 螺状
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)

    # Returns the number of bridgehead atoms (atoms 
    # shared between rings that share at least two bonds)
    # 桥状
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    
    return nBridgehead, nSpiro


def processMols(mols):
    print('smiles\t Name\t sa_score')
    for i, m in enumerate(mols):
        if m is None:
            continue

    s = calculateScore(m)

    smiles = Chem.MolToSmiles(m)
    print(smiles + '\t' + m.GetProp('_Name') + '\t%3f' %s)


def calculateScore(m):
    # fragment score
    if _fscores is None:
        readFragmentScores

    # Returns a Morgan fingerprint for a molecule
    fp = rdMolDescriptors.GetMorganFingerprint(m,
    # ↓ 2 is the *radius* of the circular fingerprint
                                                2)
    # returns a dictionary of the nonzero elements
    fps = fp.GetNonzeroElements()
    print(fps)
    score1 = 0.
    nf = 0

    for bitId, v in iteritems(fps):
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4 * v)
    
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    # 手性中心
    nChiralCenters = len(Chem.FindMolChiralCenters(
        m,
        includeUnassigned=True
    ))

    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    # 大环化合物
    nMacrocycles = 0

    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms ** 1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    # This differs from the paper, which defines:
    #
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # 
    # This form generates better results when 2 or 
    # more macrocycles are present
    macrocyclePenalty = 0.
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise

    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.

    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0    

    return sascore

if __name__ == '__main__':
    import sys, time

    t1 =time.time()
    readFragmentScores("fpscores")
    t2 = time.time()

    suppl = Chem.SmilesMolSupplier(sys.argv[1])
    t3 = time.time()
    processMols(suppl)
    t4 = time.time()

    print('Reading took %.2f seconds. Calculating took %.2f seconds' 
            % ((t2 - t1), (t4 - t3)),
            file=sys.stderr)

