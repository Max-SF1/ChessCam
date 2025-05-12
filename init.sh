apt update
apt install gedit -y

#these three lines were added by chatGPT for matlab error. 

# Clean reinstall of matplotlib to avoid system/PIP conflicts
python3 -m pip install --upgrade pip
pip uninstall -y matplotlib
pip install matplotlib --no-cache-dir

pip install -r requirements.txt --ignore-installed