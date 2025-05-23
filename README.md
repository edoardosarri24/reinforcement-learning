# Overview
This repository contains a project on Reinforcement Learning. The CartPole and LunarLander environments from Gymnasium were solved using the REINFORCE algorithm.

The code does not necessarily need to be executed. If you’re only interested in the results, you can simply refer to the report provided in the `relazione.pdf` file.

---

# Install
Clone the repository and install the requirements. This line were tested for osx-arm64.

```bash
git clone https://github.com/edoardosarri24/Reinforcement-learning.git
cd reinforcement-learning
conda env create -f environment.yml -n edoardosarri
conda activate edoardosarri
wandb login
```

# Run
Run the main training script with the following command:
```bash
python main.py --env cart_pole|lunar_lander [OPTIONS]
```

If you want to run the same experiments, execute the following command:
```bash
./script.sh
```

### Options
- `--env`: environment to solve (options: `cart_pole`, `lunar_lander`).
- `--run_name`: name of the andB run (dafault: `test`).
- `--baseline`: Type of baseline to use (options: `none`, `std`, `stateValue`; default: `none`).
- `--gamma`: Discount factor for future rewards (default: `0.99`).
- `--lr`: Learning rate for the optimizer (default: `1e-3`).
- `--episodes`: Number of training episodes (default: `500`).
- `--visualize`: Set flag to visualize the final agent (default: `False`).
- `--device`: (option: `cpu`, `mps`, `cuda`; default: `cpu`).
- `--sweep`: make sweep (only for cart_pole).

### Example
To train the agent without using a baseline and visualize the final agent, you can run:

```bash
python main.py --env cart_pole --episodes 1000 --visualize
```
