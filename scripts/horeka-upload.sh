USER=$1

rsync -azvW -P \
  --exclude=.git \
  --exclude=.idea \
  --exclude=__pycache__ \
  --exclude=tmp \
  --exclude=outputs \
  --exclude=models \
  --exclude=plots \
  --exclude=resampling* \
  --exclude=*.pth \
  "$(pwd)" "$USER"@horeka.scc.kit.edu:~/code/repos
