import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead_fractions = []
        current = x

        with torch.no_grad():
            for layer in model.children():
                current = layer(current)
                if isinstance(layer, nn.ReLU):
                    # current shape: (batch_size, n_neurons)
                    # a neuron is dead if it's 0 for every sample in the batch
                    dead_mask = (current == 0).all(dim=0)  # True for each dead neuron
                    dead_fraction = dead_mask.float().mean().item()
                    dead_fractions.append(round(dead_fraction, 4))

        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        # 1. any layer more than half dead
        if any(f > 0.5 for f in dead_fractions):
            return 'use_leaky_relu'
        
        # 2. first layer is significantly dead
        if dead_fractions[0] > 0.3:
            return 'reinitialize'
        
        # 3. dead fraction strictly increases with depth and last layer is bad
        strictly_increasing = all(
            dead_fractions[i] < dead_fractions[i+1]
            for i in range(len(dead_fractions)-1)
        )
        if strictly_increasing and dead_fractions[-1] > 0.1:
            return 'reduce_learning_rate'
        
        # 4 & 5. healthy
        return 'healthy'
