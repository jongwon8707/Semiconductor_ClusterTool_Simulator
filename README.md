# Deep Reinforcement Learning for Scheduling Semiconductor Cluster Tools in Varying Configurations

This repository provides the full source code and simulation environment used in the paper **“Deep Reinforcement Learning for Scheduling Semiconductor Cluster Tools in Varying Configurations.”**

The project includes:

- A realistic semiconductor cluster tool simulator (Vacuum & Atmospheric robots, chambers, wafer flow)
- A Gymnasium-compatible RL environment
- Deep Q-Network (DQN) agents for throughput optimization under varying system conditions

## 📂 Repository Structure

- **`EQP_Scheduler.py`**  
  Core simulation logic of the cluster tool.  
  Models wafer flow, robot behaviors (VTM/ATM), chambers, load locks, and processing sequences.

- **`EQP_Scheduler_env.py`**  
  Gymnasium wrapper for the simulator.  
  Defines the state space, action space, reward function, and episode logic.

- **`DQN_2.py`**  
  Main entry point.  
  Includes the DQN agent, training loop, target/policy networks, validation procedure, hyperparameter grid search, and comparative evaluation logic.

- **`requirements.txt`**  
  List of all dependencies.

- **`runs/`**  
  Auto-generated folder that stores:
  - Training logs (`.log`)
  - Model checkpoints (`.pt`)
  - Visualizations (`.png`)
  - Exported data (`.csv`)

## 🚀 Installation

### 1. Clone the repository

```bash
git clone [https://github.com/jongwon8707/Semiconductor_ClusterTool_Simulator.git](https://github.com/jongwon8707/Semiconductor_ClusterTool_Simulator.git)
cd Semiconductor_ClusterTool_Simulator
```

### 2. Install dependencies

Python **3.8+** is recommended.

```bash
pip install -r requirements.txt
```

## 💻 Usage

The experimental data used in the paper was generated using this simulation framework.  
You can reproduce the results by running the main script.

### 1. Train the Agent (Data Generation)

```bash
python DQN_2.py
```

**Default behavior (`is_training=True`):**

Outputs stored in `runs/`:

- **Logs:** `*.log` files with step/episode information  
- **Models:** best model saved as `*.pt`  
- **Charts:** reward curves, epsilon decay, loss curves (`*.png`)  
- **Data:** exported metrics (`*_chart_data.csv`, `*_validation_data.csv`)

### 2. Evaluation & Comparison (Paper Reproduction)

To reproduce the comparative analysis between DRL agents and rule-based heuristics (Random, Rule 1, Rule 2, Rule 3) as presented in the paper (e.g., boxplots):

1. Open `DQN_2.py`
2. Locate the `if __name__ == "__main__":` block at the bottom.
3. Uncomment or ensure the following line is active:

```python
dql.compare_and_plot(num_episodes=10)
```

4. Run the script:

```bash
python DQN_2.py
```

**Output:**  
This will generate `model_comparison.png` (or similar) in the `runs/` directory, visualizing the performance differences.

### 3. Hyperparameter Sensitivity Analysis

To run the grid search as described in the paper:

1. Open `DQN_2.py`
2. Scroll to the main block and uncomment:

```python
# dql.hyperparameter_grid_search(max_episodes=1000, validation_episodes=1)
```

3. Run again:

```bash
python DQN_2.py
```

### 4. Visualization (Real-Time Rendering)

To watch the agent controlling the equipment in real time:

```python
dql.run(
    trained_model_file='[MODEL_NAME_WITHOUT_EXTENSION]',
    is_training=False,
    render=True
)
```

Running the script will open a window visualizing:

- Load ports  
- ATM robot  
- Load locks  
- VTM robot  
- Six process chambers  
- Wafer IDs, states, and process times  

## 📝 Requirements

- gymnasium  
- torch  
- numpy  
- matplotlib  
- pygame  
- seaborn  
- scipy  
- pyyaml  

Install them via:

```bash
pip install -r requirements.txt
```

## 📧 Contact

For questions regarding the code or experimental data, please contact the author at:

**[kirora@korea.ac.kr]**
