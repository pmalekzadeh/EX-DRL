python --version
sudo apt-get update -y
sudo apt-get install python3.8 python3.8-dev python3.8-distutils libpython3.8-dev
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
sudo update-alternatives --config python3
python --version
sudo python3 get-pip.py
pip install -r requirements.txt
pip install tensorflow[and-cuda]==2.8.0