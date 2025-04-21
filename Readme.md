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

You can run METRA + SAC on Ant Environment by running `python ant.py` (and the same setup steps as above)

# Results
You can look at the tensorboard logs that support the argument in report. Specifically,
1. `runs/SAC-20_155305`: METRA + SAC implementation on ant environment
2. `runs/SAC-Discrete-18_183958`, `runs/SAC-Discrete-18_190806/`, `runs/SAC-Discrete-18_191832`: represent the sudden policy/entropy drop that cause unstable loss functions, prompting the adoption of a decaying temperature schedule
3. `runs/SAC-Discrete-20_114556`: Mean $\Delta \phi$ in original architecture
4. `runs/SAC-Discrete-20_165643`: Mean $\Delta \phi$ in ResNet backboned architecture
