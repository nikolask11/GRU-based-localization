# GRU-based Robot Localization

Comparing particle filter vs neural network for 2D robot localization.

## What it does

A robot moves in a 20×20 world with 8 landmarks. Both methods try to estimate the robot's position `(x, y, θ)` using noisy sensors and movement data.

**Result**: Particle filter wins. Neural networks struggle with global localization because they can't maintain multiple hypotheses.

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

**GRU Network**: Predicts a single location from sensor/movement history.

## Why neural network loses

Global localization is ambiguous early on. Particle filters can say "I don't know yet" by spreading particles. Neural networks must pick one spot immediately.

## License

MIT
