conda create -n drl python=3.8 -y
conda activate drl

pip install -r requirements.txt
pip install tensorflow[and-cuda]==2.8.0

cp ~/miniconda3/envs/drl/lib/libpython3.8.so.1.0 /usr/lib/libpython3.8.so.1.0
python -m ipykernel install --user --name drl

git config user.name colinw
git config user.email colingwuyu@gmail.com
