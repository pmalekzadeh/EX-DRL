# Create the directories to place our activation and deacivation scripts in
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
conda install -c conda-forge cudnn -y
# Add commands to the scripts
printf 'export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}\nexport LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib/\n' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
printf 'export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}\nunset OLD_LD_LIBRARY_PATH\n' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

# Run the script once
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# pip install --upgrade pip
# pip install tensorflow==2.8.0
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
