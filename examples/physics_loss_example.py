#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple example demonstrating how to use physics loss in neural operator training.

This example shows:
1. How to create a physics loss
2. How to combine it with data loss
3. How to use it in training
"""

import torch
import torch.nn as nn
import numpy as np
from neuralop.losses.data_losses import LpLoss
from neuralop.losses.equation_losses import FloodPhysicsLoss, FloodContinuityLoss
from neuralop.losses.meta_losses import WeightedSumLoss


def create_synthetic_flood_data(batch_size=4, n_nodes=64, n_history=5):
    """
    Create synthetic flood data for demonstration.
    
    Parameters
    ----------
    batch_size : int
        Number of samples in batch
    n_nodes : int
        Number of spatial nodes
    n_history : int
        Number of history time steps
        
    Returns
    -------
    dict
        Dictionary containing synthetic data
    """
    # Create synthetic input features
    # Shape: [batch, n_history, n_nodes, channels]
    # Channels: [water_depth, velocity_x, velocity_y, volume]
    x = torch.randn(batch_size, n_history, n_nodes, 4)
    
    # Create synthetic targets
    # Shape: [batch, n_nodes, 2]
    # Channels: [water_depth_diff, volume_diff]
    y = torch.randn(batch_size, n_nodes, 2)
    
    # Create synthetic physics data
    physics_data = {
        "past_volume": torch.randn(batch_size),
        "future_volume": torch.randn(batch_size),
        "avg_inflow": torch.randn(batch_size),
        "avg_precipitation": torch.randn(batch_size),
        "next_inflow": torch.randn(batch_size),
        "next_precip": torch.randn(batch_size),
        "volume_mean": torch.zeros(batch_size),
        "volume_std": torch.ones(batch_size),
        "num_nodes": torch.full((batch_size,), n_nodes),
        "area_sum": torch.randn(batch_size),
        "infiltration_area_sum": torch.randn(batch_size),
    }
    
    return {
        "x": x,
        "y": y,
        "physics_data": physics_data
    }


def demonstrate_physics_loss():
    """Demonstrate how to use physics loss."""
    print("=== Physics Loss Demonstration ===\n")
    
    # Create synthetic data
    data = create_synthetic_flood_data()
    x = data["x"]
    y = data["y"]
    physics_data = data["physics_data"]
    
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Physics data keys: {list(physics_data.keys())}")
    print()
    
    # 1. Create different types of physics losses
    print("1. Creating physics losses...")
    
    # Basic physics loss
    basic_physics_loss = FloodPhysicsLoss(
        delta_t=1200.0,
        physics_weight=1.0,
        enforce_continuity=True,
        enforce_momentum=True,
        enforce_bounds=True
    )
    
    # Continuity loss (similar to hydrograph example)
    continuity_loss = FloodContinuityLoss(
        delta_t=1200.0,
        physics_weight=1.0
    )
    
    # Data loss
    data_loss = LpLoss(d=2, p=2)
    
    print("✓ Physics losses created")
    print()
    
    # 2. Demonstrate individual losses
    print("2. Computing individual losses...")
    
    # Data loss
    data_loss_value = data_loss(y, y)  # Should be 0 for identical tensors
    print(f"Data loss (identical tensors): {data_loss_value:.6f}")
    
    # Physics loss
    physics_loss_value = basic_physics_loss(y, x)
    print(f"Basic physics loss: {physics_loss_value:.6f}")
    
    # Continuity loss
    continuity_loss_value = continuity_loss(y, x, physics_data=physics_data)
    print(f"Continuity loss: {continuity_loss_value:.6f}")
    
    print()
    
    # 3. Combine losses using WeightedSumLoss
    print("3. Combining losses...")
    
    # Combine data loss and physics loss
    combined_loss = WeightedSumLoss(
        losses=[data_loss, basic_physics_loss],
        weights=[1.0, 0.5]  # Data loss has weight 1.0, physics loss has weight 0.5
    )
    
    total_loss = combined_loss(y, x)
    print(f"Combined loss (data + physics): {total_loss:.6f}")
    
    # Combine data loss and continuity loss
    combined_loss_continuity = WeightedSumLoss(
        losses=[data_loss, continuity_loss],
        weights=[1.0, 1.0]  # Equal weights
    )
    
    total_loss_continuity = combined_loss_continuity(y, x, physics_data=physics_data)
    print(f"Combined loss (data + continuity): {total_loss_continuity:.6f}")
    
    print()
    
    # 4. Demonstrate loss computation with different weights
    print("4. Loss with different physics weights...")
    
    weights = [0.0, 0.1, 0.5, 1.0, 2.0]
    for weight in weights:
        combined_loss_weighted = WeightedSumLoss(
            losses=[data_loss, basic_physics_loss],
            weights=[1.0, weight]
        )
        loss_value = combined_loss_weighted(y, x)
        print(f"Physics weight {weight:3.1f}: {loss_value:.6f}")
    
    print()
    
    # 5. Demonstrate physics loss components
    print("5. Physics loss components...")
    
    # Continuity component
    continuity_component = basic_physics_loss.continuity_loss(y, x)
    print(f"Continuity component: {continuity_component:.6f}")
    
    # Momentum component
    momentum_component = basic_physics_loss.momentum_loss(y, x)
    print(f"Momentum component: {momentum_component:.6f}")
    
    # Bounds component
    bounds_component = basic_physics_loss.bounds_loss(y, x)
    print(f"Bounds component: {bounds_component:.6f}")
    
    print()
    
    # 6. Demonstrate with different physics loss configurations
    print("6. Different physics loss configurations...")
    
    # Only continuity
    continuity_only = FloodPhysicsLoss(
        delta_t=1200.0,
        physics_weight=1.0,
        enforce_continuity=True,
        enforce_momentum=False,
        enforce_bounds=False
    )
    loss_continuity_only = continuity_only(y, x)
    print(f"Continuity only: {loss_continuity_only:.6f}")
    
    # Only momentum
    momentum_only = FloodPhysicsLoss(
        delta_t=1200.0,
        physics_weight=1.0,
        enforce_continuity=False,
        enforce_momentum=True,
        enforce_bounds=False
    )
    loss_momentum_only = momentum_only(y, x)
    print(f"Momentum only: {loss_momentum_only:.6f}")
    
    # Only bounds
    bounds_only = FloodPhysicsLoss(
        delta_t=1200.0,
        physics_weight=1.0,
        enforce_continuity=False,
        enforce_momentum=False,
        enforce_bounds=True
    )
    loss_bounds_only = bounds_only(y, x)
    print(f"Bounds only: {loss_bounds_only:.6f}")
    
    print("\n=== Demonstration Complete ===")


def demonstrate_training_step():
    """Demonstrate how physics loss would be used in a training step."""
    print("\n=== Training Step Demonstration ===\n")
    
    # Create synthetic data
    data = create_synthetic_flood_data()
    x = data["x"]
    y = data["y"]
    physics_data = data["physics_data"]
    
    # Create a simple model (just for demonstration)
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)
        
        def forward(self, x):
            # Reshape input: [batch, history, nodes, channels] -> [batch*history*nodes, channels]
            batch, history, nodes, channels = x.shape
            x_flat = x.view(-1, channels)
            
            # Apply linear transformation
            out_flat = self.linear(x_flat)
            
            # Reshape output: [batch*history*nodes, 2] -> [batch, nodes, 2]
            # We'll take the last time step for simplicity
            out = out_flat.view(batch, history, nodes, 2)
            return out[:, -1, :, :]  # Last time step
    
    # Create model and losses
    model = SimpleModel()
    data_loss = LpLoss(d=2, p=2)
    physics_loss = FloodContinuityLoss(delta_t=1200.0, physics_weight=0.5)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training step simulation...")
    
    # Forward pass
    model.train()
    optimizer.zero_grad()
    
    y_pred = model(x)
    
    # Compute losses
    data_loss_value = data_loss(y_pred, y)
    physics_loss_value = physics_loss(y_pred, x, physics_data=physics_data)
    
    # Total loss
    total_loss = data_loss_value + physics_loss_value
    
    print(f"Data loss: {data_loss_value:.6f}")
    print(f"Physics loss: {physics_loss_value:.6f}")
    print(f"Total loss: {total_loss:.6f}")
    
    # Backward pass (commented out since this is just demonstration)
    # total_loss.backward()
    # optimizer.step()
    
    print("✓ Training step completed (backward pass commented out)")
    print("\n=== Training Step Complete ===")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_physics_loss()
    demonstrate_training_step()
    
    print("\n" + "="*50)
    print("Physics Loss Example Complete!")
    print("="*50)
    print("\nKey takeaways:")
    print("1. Physics loss can be combined with data loss using WeightedSumLoss")
    print("2. Different physics components can be enabled/disabled")
    print("3. Physics loss weight controls the importance of physics constraints")
    print("4. Continuity loss requires physics_data dictionary with specific keys")
    print("5. Physics loss helps enforce physical constraints during training") 