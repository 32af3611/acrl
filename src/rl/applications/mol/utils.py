"""Tools for manipulating graphs and converting from atom and pair features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shlex
import subprocess
import sys
import uuid
from sys import exit

import numpy as np
import pandas as pd
import scipy.spatial as scsp
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import RDConfig
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.rl.applications.mol import hyp
from src.rl.applications.mol.model.model_definitions import triple_predict

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer


def get_largest_ring_size(molecule):
    """Calculates the largest ring size in the molecule.
    Refactored from
    https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py
    Args:
      molecule: Chem.Mol. A molecule.
    Returns:
      Integer. The largest ring size.
    """
    cycle_list = molecule.GetRingInfo().AtomRings()
    if cycle_list:
        cycle_length = max([len(j) for j in cycle_list])
    else:
        cycle_length = 0
    return cycle_length


def penalized_logp(molecule):
    """Calculates the penalized logP of a molecule.
    Refactored from
    https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py
    See Junction Tree Variational Autoencoder for Molecular Graph Generation
    https://arxiv.org/pdf/1802.04364.pdf
    Section 3.2
    Penalized logP is defined as:
     y(m) = logP(m) - SA(m) - cycle(m)
     y(m) is the penalized logP,
     logP(m) is the logP of a molecule,
     SA(m) is the synthetic accessibility score,
     cycle(m) is the largest ring size minus by six in the molecule.
    Args:
      molecule: Chem.Mol. A molecule.
    Returns:
      Float. The penalized logP value.
    """
    log_p = Descriptors.MolLogP(molecule)
    sas_score = sascorer.calculateScore(molecule)
    largest_ring_size = get_largest_ring_size(molecule)
    cycle_score = max(largest_ring_size - 6, 0)
    return log_p - sas_score - cycle_score


def get_fingerprint(smiles, fingerprint_length, fingerprint_radius):
    """Get Morgan Fingerprint of a specific SMILES string.

  Args:
    smiles: String. The SMILES string of the molecule.
    hparams: tf.contrib.training.HParams. Hyper parameters.

  Returns:
    np.array. shape = [hparams.fingerprint_length]. The Morgan fingerprint.
  """
    if smiles is None:
        return np.zeros((hyp.fingerprint_length,))
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return np.zeros((hyp.fingerprint_length,))
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, hyp.fingerprint_radius, hyp.fingerprint_length
    )
    arr = np.zeros((1,))
    # ConvertToNumpyArray takes ~ 0.19 ms, while
    # np.asarray takes ~ 4.69 ms
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


def atom_valences(atom_types):
    """Creates a list of valences corresponding to atom_types.

  Note that this is not a count of valence electrons, but a count of the
  maximum number of bonds each element will make. For example, passing
  atom_types ['C', 'H', 'O'] will return [4, 1, 2].

  Args:
    atom_types: List of string atom types, e.g. ['C', 'H', 'O'].

  Returns:
    List of integer atom valences.
  """
    periodic_table = Chem.GetPeriodicTable()
    return [
        max(list(periodic_table.GetValenceList(atom_type))) for atom_type in atom_types
    ]


def get_scaffold(mol):
    """Computes the Bemis-Murcko scaffold for a molecule.

  Args:
    mol: RDKit Mol.

  Returns:
    String scaffold SMILES.
  """
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=True)


def contains_scaffold(mol, scaffold):
    """Returns whether mol contains the given scaffold.

  NOTE: This is more advanced than simply computing scaffold equality (i.e.
  scaffold(mol_a) == scaffold(mol_b)). This method allows the target scaffold to
  be a subset of the (possibly larger) scaffold in mol.

  Args:
    mol: RDKit Mol.
    scaffold: String scaffold SMILES.

  Returns:
    Boolean whether scaffold is found in mol.
  """
    pattern = Chem.MolFromSmiles(scaffold)
    matches = mol.GetSubstructMatches(pattern)
    return bool(matches)


def get_largest_ring_size(molecule):
    """Calculates the largest ring size in the molecule.

  Refactored from
  https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py

  Args:
    molecule: Chem.Mol. A molecule.

  Returns:
    Integer. The largest ring size.
  """
    cycle_list = molecule.GetRingInfo().AtomRings()
    if cycle_list:
        cycle_length = max([len(j) for j in cycle_list])
    else:
        cycle_length = 0
    return cycle_length


def penalized_logp(molecule):
    """Calculates the penalized logP of a molecule.

  Refactored from
  https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py
  See Junction Tree Variational Autoencoder for Molecular Graph Generation
  https://arxiv.org/pdf/1802.04364.pdf
  Section 3.2
  Penalized logP is defined as:
   y(m) = logP(m) - SA(m) - cycle(m)
   y(m) is the penalized logP,
   logP(m) is the logP of a molecule,
   SA(m) is the synthetic accessibility score,
   cycle(m) is the largest ring size minus by six in the molecule.

  Args:
    molecule: Chem.Mol. A molecule.

  Returns:
    Float. The penalized logP value.

  """
    log_p = Descriptors.MolLogP(molecule)
    sas_score = sascorer.calculateScore(molecule)
    largest_ring_size = get_largest_ring_size(molecule)
    cycle_score = max(largest_ring_size - 6, 0)
    return log_p - sas_score - cycle_score


def bin_based_selection(smiles_list, index_list, n_points, n_bins, model1, model2, model3):
    """Implements points sampling based on standard deviations of individual bins of the distribution of points.

  Args:
    smiles_list : list of SMILES values
    index_list : their index numbers
    n_points : number of points to sample
    n_bins : number of bins of the histogram
    model1 : first model
    model2 : second model
    model3 : third model

  Returns:
    list of selected SMILES
  """
    df = pd.DataFrame()
    df["SMILES"] = smiles_list
    df["index"] = index_list
    preds = [triple_predict(Chem.MolFromSmiles(smiles), model1, model2, model3) for smiles in smiles_list]
    means = [np.mean(preds[i]) for i in range(len(smiles_list))]
    stds = [np.std(preds[i]) for i in range(len(smiles_list))]
    spacing = (np.max(means) - np.min(means)) / n_bins
    bins = np.arange(np.min(means), np.max(means), spacing)
    df["std"] = stds
    bin_indices = np.digitize(means, bins)
    df["bin"] = bin_indices
    it_list = []
    selected_std_list = []
    for bin_value in np.unique(bin_indices):
        n_points_per_bin = (n_points * len(df[df["bin"] == bin_value]) // len(df)) + 1
        df_1 = df[df["bin"] == bin_value].sort_values('std', ascending=False).head(n_points_per_bin)
        # it_list += df_1['SMILES'].values.tolist()
        it_list += df_1['index'].values.tolist()
        selected_std_list += df_1['std'].values.tolist()

    it_list = np.array(it_list)[np.argsort(-np.array(selected_std_list))][:n_points].tolist()

    return it_list


def get_coords_from_smiles(smiles, suffix, conversion_method):
    if conversion_method == "any":
        to_try = ["rdkit", "molconvert"]
    elif conversion_method == "rdkit":
        to_try = ["rdkit"]
    elif conversion_method == "molconvert":
        to_try = ["molconvert"]
    error = ""
    for m in to_try:
        if m == "molconvert":

            if which("molconvert") != None:

                coords, elements = get_coords_from_smiles_marvin(smiles, suffix)
                if coords is None or elements is None:
                    error += " molconvert_failed "
                    pass
                else:
                    if abs(np.max(coords.T[2]) - np.min(coords.T[2])) > 0.01:
                        print("   ---   conversion done with molconvert")
                        return (coords, elements)
                    else:
                        error += " molconvert_mol_flat "
                        pass
            else:
                error += " molconvert_not_available "

        if m == "rdkit":
            # print("use rdkit")
            coords, elements = get_coords_from_smiles_rdkit(smiles, suffix)
            if coords is None or elements is None:
                print(" rdkit_failed ")
                return None, None
            else:
                if abs(np.max(coords.T[2]) - np.min(coords.T[2])) > 0.01:
                    # print("   ---   conversion done with rdkit")
                    return coords, elements
                else:
                    print(" rdkit_failed, produced flat molecule! ")
                    return None, None


def get_coords_from_smiles_rdkit(smiles, suffix):
    try:
        m = Chem.MolFromSmiles(smiles)
    except:
        return (None, None)
        # print("could not convert %s to rdkit molecule. Exit!"%(smiles))
        # exit()
    try:
        m = Chem.AddHs(m)
    except:
        return (None, None)
        # print("ERROR: could not add hydrogen to rdkit molecule of %s. Exit!"%(smiles))
        # exit()
    try:
        # AllChem.EmbedMolecule(m)
        conformer_id = AllChem.EmbedMolecule(m, useRandomCoords=True)
        if conformer_id < 0:
            print(f'Could not embed molecule {smiles}.')
            return (None, None)
    except:
        return (None, None)
        # print("ERROR: could not calculate 3D coordinates from rdkit molecule %s. Exit!"%(smiles))
        # exit()
    try:
        block = Chem.MolToMolBlock(m)
        blocklines = block.split("\n")
        coords = []
        elements = []
        for line in blocklines[4:]:
            if len(line.split()) == 4:
                break
            elements.append(line.split()[3])
            coords.append([float(line.split()[0]), float(line.split()[1]), float(line.split()[2])])
        coords = np.array(coords)
        mean = np.mean(coords, axis=0)
        distances = scsp.distance.cdist([mean], coords)[0]
        if np.max(distances) < 0.1:
            return (None, None)

    except:
        return (None, None)

    return (coords, elements)


def get_coords_from_smiles_marvin(smiles, suffix):
    name = uuid.uuid4()

    if not os.path.exists("tempfiles%s" % (suffix)):
        try:
            os.makedirs("tempfiles%s" % (suffix))
        except:
            pass
    if not os.path.exists("input_structures%s" % (suffix)):
        try:
            os.makedirs("input_structures%s" % (suffix))
        except:
            pass

    outfile = open("tempfiles%s/%s.smi" % (suffix, name), "w")
    outfile.write("%s\n" % (smiles))
    outfile.close()

    path_here = os.getcwd()
    os.system(
        "molconvert -2 mrv:+H %s/tempfiles%s/%s.smi > tempfiles%s/%s.mrv" % (path_here, suffix, name, suffix, name))
    filename = "tempfiles%s/%s.mrv" % (suffix, name)
    if not os.path.exists(filename):
        os.system("rm tempfiles%s/%s.smi" % (suffix, name))
        return (None, None)
        # print("ERROR: could not convert %s to 2D (mrv) using marvin. Exit!"%(smiles))
        # exit()

    os.system(
        "molconvert -3 xyz %s/tempfiles%s/%s.mrv > input_structures%s/%s.xyz" % (path_here, suffix, name, suffix, name))
    filename = "input_structures%s/%s.xyz" % (suffix, name)
    if not os.path.exists(filename):
        os.system("rm tempfiles%s/%s.smi tempfiles%s/%s.mrv" % (suffix, name, suffix, name))
        return (None, None)
        # print("ERROR: could not convert %s to 3D (xyz) using marvin. Exit!"%(smiles))
        # exit()

    coords, elements = readXYZ(filename)
    if len(coords) == 0:
        os.system("rm tempfiles%s/%s.smi tempfiles%s/%s.mrv input_structures%s/%s.xyz" % (
            suffix, name, suffix, name, suffix, name))
        print("ERROR: could not convert %s to 3D (coords in empty) using marvin. Exit!" % (smiles))
        # return(None, None)
        # exit()
    os.system("rm tempfiles%s/%s.smi tempfiles%s/%s.mrv input_structures%s/%s.xyz" % (
        suffix, name, suffix, name, suffix, name))
    os.system("rm -r tempfiles%s" % (suffix))
    os.system("rm -r input_structures%s" % (suffix))
    return (coords, elements)


def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def readXYZ(filename):
    infile = open(filename, "r")
    coords = []
    elements = []
    lines = infile.readlines()
    if len(lines) < 3:
        exit("ERROR: no coordinates found in %s/%s" % (os.getcwd(), filename))
    for line in lines[2:]:
        elements.append(line.split()[0].capitalize())
        coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
    infile.close()
    coords = np.array(coords)
    return coords, elements


def exportXYZ(coords, elements, filename, mask=[]):
    if coords is None or elements is None:
        pass
    else:
        outfile = open(filename, "w")
        if len(mask) == 0:
            outfile.write("%i\n\n" % (len(elements)))
            for atomidx, atom in enumerate(coords):
                outfile.write("%s %f %f %f\n" % (elements[atomidx].capitalize(), atom[0], atom[1], atom[2]))
        else:
            outfile.write("%i\n\n" % (len(mask)))
            for atomidx in mask:
                atom = coords[atomidx]
                outfile.write("%s %f %f %f\n" % (elements[atomidx].capitalize(), atom[0], atom[1], atom[2]))
        outfile.close()


def xtb_calc(coords, elements, opt=False, grad=False, hess=False, charge=0, freeze=[]):
    if opt and grad:
        exit("opt and grad are exclusive")
    if hess and grad:
        exit("hess and grad are exclusive")

    if hess or grad:
        if len(freeze) != 0:
            print("WARNING: please test the combination of hess/grad and freeze carefully")

    rundir = "xtb_tmpdir_%s" % (uuid.uuid4())
    if not os.path.exists(rundir):
        os.makedirs(rundir)
    else:
        if len(os.listdir(rundir)) > 0:
            os.system("rm %s/*" % (rundir))

    startdir = os.getcwd()
    os.chdir(rundir)

    exportXYZ(coords, elements, "in.xyz")

    if len(freeze) > 0:
        outfile = open("xcontrol", "w")
        outfile.write("$fix\n")
        outfile.write(" atoms: ")
        for counter, i in enumerate(freeze):
            if (counter + 1) < len(freeze):
                outfile.write("%i," % (i + 1))
            else:
                outfile.write("%i\n" % (i + 1))
        # outfile.write("$gbsa\n solvent=toluene\n")
        outfile.close()
        add = " -I xcontrol "
    else:
        add = ""

    if not os.path.exists("in.xyz"):

        os.chdir(startdir)
        os.system("rm -r %s" % (rundir))
        results = {"energy": None, "HOMO": None, "LUMO": None,
                   "coords": None, "elements": None, "gradient": None, "hessian": None,
                   "vibspectrum": None, "reduced_masses": None}
        return results

    else:
        if charge == 0:
            if opt:
                if hess:
                    command = "xtb %s in.xyz --ohess" % (add)
                else:
                    command = "xtb %s in.xyz --opt" % (add)
            else:
                if grad:
                    command = "xtb %s in.xyz --grad" % (add)
                else:
                    command = "xtb %s in.xyz" % (add)

        else:
            if opt:
                if hess:
                    command = "xtb %s in.xyz --ohess --chrg %i" % (add, charge)
                else:
                    command = "xtb %s in.xyz --opt --chrg %i" % (add, charge)
            else:
                if grad:
                    command = "xtb %s in.xyz --grad --chrg %i" % (add, charge)
                else:
                    command = "xtb %s in.xyz --chrg %i" % (add, charge)

        os.environ["OMP_NUM_THREADS"] = "10"  # "%s"%(settings["OMP_NUM_THREADS"])
        os.environ["MKL_NUM_THREADS"] = "10"  # "%s"%(settings["MKL_NUM_THREADS"])

        args = shlex.split(command)

        mystdout = open("xtb.log", "a")
        process = subprocess.Popen(args, stdout=mystdout, stderr=subprocess.PIPE)
        out, err = process.communicate()
        mystdout.close()

        if opt:
            if not os.path.exists("xtbopt.xyz"):
                print("WARNING: xtb geometry optimization did not work")
                coords_new, elements_new = None, None
            else:
                coords_new, elements_new = readXYZ("xtbopt.xyz")
        else:
            coords_new, elements_new = None, None

        if grad:
            grad = read_xtb_grad()
        else:
            grad = None

        if hess:
            hess, vibspectrum, reduced_masses = read_xtb_hess()
        else:
            hess, vibspectrum, reduced_masses = None, None, None

        e = read_xtb_energy()
        HOMO, LUMO = read_xtb_homo_lumo()
        os.chdir(startdir)

        os.system("rm -r %s" % (rundir))

        results = {"energy": e, "HOMO": HOMO, "LUMO": LUMO,
                   "coords": coords_new, "elements": elements_new, "gradient": grad, "hessian": hess,
                   "vibspectrum": vibspectrum, "reduced_masses": reduced_masses}
        return (results)


def read_xtb_energy():
    if not os.path.exists("xtb.log"):
        return (None)
    energy = None
    for line in open("xtb.log"):
        if "| TOTAL ENERGY" in line:
            energy = float(line.split()[3])
    return (energy)


def read_xtb_homo_lumo():
    if not os.path.exists("xtb.log"):
        return (None, None)
    HOMO, LUMO = None, None
    for line in open("xtb.log"):
        if "(HOMO)" in line:
            HOMO = float(line.split()[len(line.split()) - 3])
        if "(LUMO)" in line:
            LUMO = float(line.split()[len(line.split()) - 3])
    return HOMO, LUMO


def read_xtb_grad():
    if not os.path.exists("gradient"):
        return (None)
    grad = []
    for line in open("gradient", "r"):
        if len(line.split()) == 3:
            grad.append([float(line.split()[0]), float(line.split()[1]), float(line.split()[2])])
    if len(grad) == 0:
        grad = None
    else:
        grad = np.array(grad)
    return (grad)


def read_xtb_hess():
    hess = None
    if not os.path.exists("hessian"):
        return (None, None, None)
    hess = []
    for line in open("hessian", "r"):
        if "hess" not in line:
            for x in line.split():
                hess.append(float(x))
    if len(hess) == 0:
        hess = None
    else:
        hess = np.array(hess)

    vibspectrum = None
    if not os.path.exists("vibspectrum"):
        return (None, None, None)
    vibspectrum = []
    read = False
    for line in open("vibspectrum", "r"):
        if "end" in line:
            read = False

        if read:
            if len(line.split()) == 5:
                vibspectrum.append(float(line.split()[1]))
            elif len(line.split()) == 6:
                vibspectrum.append(float(line.split()[2]))
            else:
                print("WARNING: weird line length: %s" % (line))
        if "RAMAN" in line:
            read = True

    reduced_masses = None
    if not os.path.exists("g98.out"):
        print("g98.out not found")
        return (None, None, None)
    reduced_masses = []
    read = False
    for line in open("g98.out", "r"):
        if "Red. masses" in line:
            for x in line.split()[3:]:
                try:
                    os.system("rm -r %s" % (rundir))
                    reduced_masses.append(float(x))
                except:
                    pass

    if len(vibspectrum) == 0:
        vibspectrum = None
        print("no vibspectrum found")
    else:
        vibspectrum = np.array(vibspectrum)

    if len(reduced_masses) == 0:
        reduced_masses = None
        print("no reduced masses found")
    else:
        reduced_masses = np.array(reduced_masses)

    return (hess, vibspectrum, reduced_masses)
