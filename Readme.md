# Setup
To run the training + evaluation loop of METRA + SAC-Discrete

1. Make sure you have model/ and runs/ directory where the logs will be stored
2. Run
```
conda create -n <env_name>
conda activate <env_name>
pip install -r requirements.txt
python main.py
```

You can run METRA + SAC on Ant Environment by running `python ant.py` (and the same setup steps as above). You would also need mujoco on your machine and path.
