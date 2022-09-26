USER=$1

rsync -azvW -P \
  --exclude=*.py \
  --exclude=*.pth \
  --exclude=*.md \
  --exclude=apps \
  --exclude=scripts \
  --exclude=tmp \
  --exclude=slurm \
  --exclude=data \
  --exclude=models/reward* \
  --exclude=models/agent* \
  "$USER"@horeka.scc.kit.edu:~/code/repos/rl-approx-rewards-pub/* \
  .
