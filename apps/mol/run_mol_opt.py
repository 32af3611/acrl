import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from main_mol_opt import triple_run
from src.rl.applications.mol import hyp
from src.rl.applications.mol.agent_mol_opt import RewardMolecule
from src.rl.applications.mol.model.dqn import MolDQN
from src.rl.applications.mol.model.model_definitions import triple_predict

warnings.filterwarnings("ignore")
# python3 run_mol_opt.py -f 0 -p 320 -freq 500 -m "random"


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=int, required=True)
parser.add_argument('-p', '--points', nargs='?', const=800, type=int, default=800)
parser.add_argument('-freq', '--frequency', nargs='?', const=500, type=int, default=500)
parser.add_argument('-m', '--mode', nargs='?', const='random', type=str, default="random")
args = parser.parse_args()

model1 = MolDQN(hyp.fingerprint_length, 2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model1.load_state_dict(torch.load('src/rl/applications/mol/model/mol_opt/homo_lumo_model1.pth', map_location=device))
model1.to(device)
model1.eval()

model2 = MolDQN(hyp.fingerprint_length, 2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model2.load_state_dict(torch.load('src/rl/applications/mol/model/mol_opt/homo_lumo_model2.pth', map_location=device))
model2.to(device)
model2.eval()

model3 = MolDQN(hyp.fingerprint_length, 2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model3.load_state_dict(torch.load('src/rl/applications/mol/model/mol_opt/homo_lumo_model3.pth', map_location=device))
model3.to(device)
model3.eval()


def approx_reward(smiles, ref_homo, ref_lumo, model1_, model2_, model3_):
    molecule = Chem.MolFromSmiles(smiles)
    predictions = triple_predict(molecule, model1_, model2_, model3_)
    homo, lumo = np.mean(predictions, axis=0)
    diff = lumo - homo
    diff_ref = ref_lumo - ref_homo
    # logp = penalized_logp(molecule)
    # logp_ref = hyp.logp_ref
    penalty = np.abs(diff - diff_ref) * 27.2114 + (lumo - ref_lumo) * 27.2114
    # penalty += 0.1*(logp - logp_ref) ** 2
    return -penalty


class property(RewardMolecule):
    model1 = None
    model2 = None
    model3 = None
    ref_homo = None
    ref_lumo = None

    def _reward(self):
        return approx_reward(self._state, self.ref_homo, self.ref_lumo, self.model1, self.model2, self.model3)


TB_LOG_PATH = "outputs/mol/mol_opt/retrain"
if not os.path.exists(TB_LOG_PATH):
    os.makedirs(TB_LOG_PATH, exist_ok=True)

chk_dir = TB_LOG_PATH + "/checkpoints_" + str(args.file)
if not os.path.exists(chk_dir):
    os.makedirs(chk_dir, exist_ok=True)

# checkpoint = chk_dir + "/checkpoint_2500/"
episodes_list, original_smiles_list, SMILES_list, rewards_list = triple_run(property, chk_dir=chk_dir,
                                                                            start_from_chk=None,
                                                                            model1=model1, model2=model2,
                                                                            model3=model3,
                                                                            approx_reward=approx_reward,
                                                                            model_update=True,
                                                                            update_mode=args.mode,
                                                                            n_points=int(args.points),
                                                                            frequency=args.frequency,
                                                                            column="HOMO_LUMO")

df = pd.DataFrame()
df["episode"] = episodes_list
df["SMILES0"] = original_smiles_list
df["SMILES"] = SMILES_list
df["reward"] = rewards_list
df.to_csv(TB_LOG_PATH + "/" + str(args.file) + ".csv", index=False, header=True)
