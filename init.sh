apt update
apt install gedit -y

#these three lines were added by chatGPT for matlab error. 
# pip uninstall -y opencv-python opencv-python-headless pyqt5
pip install -r requirements.txt --no-cache-dir --ignore-installed

# Clean reinstall of matplotlib to avoid system/PIP conflicts
python3 -m pip install --upgrade pip
pip uninstall -y matplotlib
pip install matplotlib --no-cache-dir