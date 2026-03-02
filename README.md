# Smart Walkers

This repo holds the code for the experiments in the paper [Smart Walkers in Discrete Space](https://arxiv.org/abs/2601.22235).

## Installation
Installation instructions (for Linux):

1. Clone the repo and navigate to the project directory:

```bash
git clone https://github.com/gperi/smart-walkers.git
cd smart-walkers
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The scripts starting with `simulate_` run the simulations and save the data, while the scripts starting with `plot_` read the saved data and produce the plots.