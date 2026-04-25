# add claude path 
export PATH="/workspace/tools/node20/bin:$PATH"
echo 'export PATH="/workspace/tools/node20/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# DexYCB path
export DEX_YCB_DIR=/workspace/dataset/

# tmux
apt update
apt install -y tmux

# utils
apt install -y unzip zip tar

# python demo/run_demo.py     --sdf-source gt_pv --method genhand     --cluster-method hdbscan --contact-threshold 0.05     --sample-idx 70 --robot Shadow     --device cuda:0 --sim-render DIRECTtar -xzf