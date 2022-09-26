import argparse
import os
import warnings

import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED

from main import run
from src.rl.applications.mol.agent import RewardMolecule
from src.rl.applications.mol.utils import penalized_logp

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', nargs='?', const='QED', type=str, default="QED", required=True)  # ="QED" or "logP"
parser.add_argument('-f', '--file', type=int, required=True)
args = parser.parse_args()


def real_reward(molecule):
    qed = QED.qed(molecule)
    logp = penalized_logp(molecule)
    if molecule is None:
        return 0.0
    return qed, logp


if args.task == "QED":
    task_index = 0
    TB_LOG_PATH = "outputs/mol/qed/real"
elif args.task == "logP":
    task_index = 1
    TB_LOG_PATH = "outputs/mol/logp/real"
else:
    exit('task should be either "QED" or "logP"')


class property(RewardMolecule):

    def _reward(self):
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return 0.0
        return real_reward(molecule)[task_index] * self.discount_factor ** (self.max_steps - self.num_steps_taken)


if not os.path.exists(TB_LOG_PATH):
    os.makedirs(TB_LOG_PATH, exist_ok=True)
episodes_list, SMILES_list, rewards_list, losses_list = run(property)
df = pd.DataFrame()
df["episode"] = episodes_list
df["SMILES"] = SMILES_list
df["reward"] = rewards_list
df["loss"] = losses_list
df.to_csv(TB_LOG_PATH + "/" + str(args.file) + ".csv", index=False, header=True)
