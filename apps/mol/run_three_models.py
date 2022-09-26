import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from main_three_models import triple_run
from src.rl.applications.mol.agent import RewardMolecule
from src.rl.applications.mol.model.dqn import MolDQN
from src.rl.applications.mol.model.model_definitions import triple_predict

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', nargs='?', const='QED', type=str, default="QED", required=True)  # ="QED" or "logP"
parser.add_argument('-f', '--file', type=int, required=True)
parser.add_argument('-p', '--points', nargs='?', const=320, type=int, default=320)
parser.add_argument('-freq', '--frequency', nargs='?', const=500, type=int, default=500)
parser.add_argument('-m', '--mode', nargs='?', const='random', type=str, default="random")
args = parser.parse_args()

model1, model2, model3 = MolDQN(2048, 1), MolDQN(2048, 1), MolDQN(2048, 1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.task == "QED":
    model1.load_state_dict(torch.load('src/rl/applications/mol/model/QED/qed_model.pth', map_location=device))
    model2.load_state_dict(torch.load('src/rl/applications/mol/model/QED/qed_model2.pth', map_location=device))
    model3.load_state_dict(torch.load('src/rl/applications/mol/model/QED/qed_model3.pth', map_location=device))
    TB_LOG_PATH = "outputs/mol/qed/three_models/" + args.mode + "_update/" + str(args.points)
elif args.task == "logP":
    model1.load_state_dict(torch.load('src/rl/applications/mol/model/logP/logP_model.pth', map_location=device))
    model2.load_state_dict(torch.load('src/rl/applications/mol/model/logP/logP_model2.pth', map_location=device))
    model3.load_state_dict(torch.load('src/rl/applications/mol/model/logP/logP_model3.pth', map_location=device))
    TB_LOG_PATH = "outputs/mol/logp/three_models/" + args.mode + "_update/" + str(args.points)
else:
    exit('task should be either "QED" or "logP"')
model1.to(device)
model1.eval()
model2.to(device)
model2.eval()
model3.to(device)
model3.eval()


def approx_reward(smiles, model1_, model2_, model3_):
    molecule = Chem.MolFromSmiles(smiles)
    return np.mean(triple_predict(molecule, model1_, model2_, model3_))


class property(RewardMolecule):
    model1 = None
    model2 = None
    model3 = None

    def _reward(self):
        return approx_reward(self._state, self.model1, self.model2, self.model3)


if not os.path.exists(TB_LOG_PATH):
    os.makedirs(TB_LOG_PATH, exist_ok=True)

chk_dir = TB_LOG_PATH + "/checkpoints_" + str(args.file)

if not os.path.exists(chk_dir):
    os.makedirs(chk_dir, exist_ok=True)
# checkpoint = chk_dir + "/checkpoint_2500/"
episodes_list, SMILES_list, rewards_list, losses_list = triple_run(property, chk_dir=chk_dir, start_from_chk=None,
                                                                   model1=model1, model2=model2, model3=model3,
                                                                   approx_reward=approx_reward,
                                                                   model_update=True,
                                                                   update_mode=args.mode, n_points=int(args.points),
                                                                   frequency=args.frequency, column=args.task)
df = pd.DataFrame()
df["episode"] = episodes_list
df["SMILES"] = SMILES_list
df["reward"] = rewards_list
df["loss"] = losses_list
df.to_csv(TB_LOG_PATH + "/" + str(args.file) + ".csv", index=False, header=True)
