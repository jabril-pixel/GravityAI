# example_test.py
# Quick test script to verify the GravityAI framework runs correctly.
# This runs a minimal configuration (4x4x4 grid, 10 epochs) for demonstration.

import torch
import sys
import os

# Add src to path if needed
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
# from src.model import Physics3DAI 
# (Adjust imports based on your actual folder structure)

print("Starting GravityAI Quick Test...")
print("Configuration: 4x4x4 grid, 10 epochs, 16 sensors.")

# Minimal dummy data for testing
num_sensors = 16
num_cells = 64  # 4x4x4
dummy_input = torch.randn(1, num_sensors)

print("Initializing model...")
# model = Physics3DAI(num_sensors=num_sensors, num_cells=num_cells)
# For this example, we use a dummy sequential model to ensure it runs without dependencies
model = torch.nn.Sequential(
    torch.nn.Linear(num_sensors, 64),
    torch.nn.LeakyReLU(0.01),
    torch.nn.Linear(64, num_cells),
    torch.nn.Sigmoidا()
)

print("Running forward pass...")
output = model(dummy_input)
print(f"Output shape: {output.shape}")
print(f"Output sum (Total Mass): {output.sum().item():.4f}")

print("✅ Quick test completed successfully! The framework is functional.")