# R-MDN-based Robot Localization

Comparing particle filter vs neural network for 2D robot localization.

## What it does

A robot moves in a 20×20 world with 8 landmarks. Both methods try to estimate the robot's position `(x, y, θ)` using noisy sensors and movement data.

**Result**: Particle filter wins. Neural networks have higher error rate as they use function approximators.

## Setup

```bash
git clone https://github.com/nikolask11/GRU-based-localization.git
cd GRU-based-localization
pip install -r requirements.txt
```

## Run

```bash
python robot_localization.py
```

This will train the neural network, run both methods, and save comparison plots.

## Methods

**Particle Filter**: Maintains multiple location hypotheses as weighted particles.

**R-MDN Network**: Predicts a 5 Component Gaussians that collaps onto a single trajectory

## Why neural network loses

Neural Networks rely on approximate functions that are learned. These always lose to set rules-based methods in environments where sensor and motion models are known.

## License

MIT
