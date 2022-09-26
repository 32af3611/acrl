# install rdkit
conda install -y -c conda-forge rdkit

# install baselines
pip install tensorflow==1.14
pip install "git+https://github.com/openai/baselines.git"

pip install numpy==1.21.6
pip install pandas==1.3.5
pip install torch==1.12.1
pip install scikit-learn==1.0.2
pip install protobuf==3.20.0
pip install imageio==2.21.1
pip install seaborn==0.11.2