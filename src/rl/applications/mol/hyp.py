import random

import pandas as pd
from rdkit import Chem

df = pd.read_csv('data/mol/raw_dataset.csv')
df.dropna(subset=["HOMO"], inplace=True)
mol9 = [df['SMILES'].iloc[i] for i in range(len(df)) if Chem.MolFromSmiles(df['SMILES'].iloc[i]).GetNumHeavyAtoms() == 9]
start_molecule = random.choice(mol9)
reference_homo = df[df['SMILES'] == start_molecule]['HOMO'].values[0]
reference_lumo = df[df['SMILES'] == start_molecule]['LUMO'].values[0]

epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 2000
optimizer = "Adam"
polyak = 0.995
atom_types = ["C", "O", "N"]
max_steps_per_episode = 5
allow_removal = True
allow_no_modification = True
allow_bonds_between_rings = False
allowed_ring_sizes = [3, 4, 5, 6]
replay_buffer_size = 1000000
learning_rate = 1e-4
gamma = 0.95
fingerprint_radius = 2
fingerprint_length = 2048
discount_factor = 0.9

logp_ref = 9.179097331276875
