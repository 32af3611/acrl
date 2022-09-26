import argparse
import os
import warnings

import pandas as pd
import torch
from rdkit import Chem

from main import run
from src.rl.applications.mol import hyp
from src.rl.applications.mol.agent import RewardMolecule
from src.rl.applications.mol.model.dqn import MolDQN
from src.rl.applications.mol.model.model_definitions import single_predict

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', nargs='?', const='QED', type=str, default="QED", required=True)  # ="QED" or "logP"
parser.add_argument('-f', '--file', type=int, required=True)
args = parser.parse_args()

model = MolDQN(hyp.fingerprint_length, 1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.task == "QED":
    model.load_state_dict(torch.load('src/rl/applications/mol/model/QED/qed_model.pth', map_location=device))
    TB_LOG_PATH = "outputs/mol/qed/one_model/update"
elif args.task == "logP":
    model.load_state_dict(torch.load('src/rl/applications/mol/model/logP/logP_model.pth', map_location=device))
    TB_LOG_PATH = "outputs/mol/logp/one_model/update"
else:
    exit('task should be either "QED" or "logP"')

model.to(device)
model.eval()


def approx_reward(smiles, model_):
    molecule = Chem.MolFromSmiles(smiles)
    return single_predict(molecule, model_)


class property(RewardMolecule):
    model = model

    def _reward(self):
        return approx_reward(self._state, self.model)


if not os.path.exists(TB_LOG_PATH):
    os.makedirs(TB_LOG_PATH, exist_ok=True)
episodes_list, SMILES_list, rewards_list, losses_list = run(property, model=model, approx_reward=approx_reward,
                                                            model_update=True, frequency=500, column=args.task)
df = pd.DataFrame()
df["episode"] = episodes_list
df["SMILES"] = SMILES_list
df["reward"] = rewards_list
df["loss"] = losses_list
df.to_csv(TB_LOG_PATH + "/" + str(args.file) + ".csv", index=False, header=True)
