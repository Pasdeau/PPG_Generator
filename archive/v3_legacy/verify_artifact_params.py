#!/usr/bin/env python3
"""
Validation of artifact generation mechanism - Simplified
"""

import numpy as np

print("=" * 70)
print("Artifact Parameter Verification Guide")
print("=" * 70)

# Test 1: typ_artifact parameter
print("\n[1] typ_artifact = np.array([1, 1, 1, 1])")
print("-" * 70)

typ_artifact = np.array([1, 1, 1, 1])
prob = typ_artifact / np.sum(typ_artifact)
print(f"Normalized probabilities: {prob}")
print(f"  Type 0 (device displacement): {prob[0]:.0%}")
print(f"  Type 1 (forearm motion):       {prob[1]:.0%}")
print(f"  Type 2 (hand motion):          {prob[2]:.0%}")
print(f"  Type 3 (poor contact):         {prob[3]:.0%}")
print(f"\n[INFO] Conclusion: 4 artifact types appear randomly, each with probability {prob[0]:.0%}")

# Test 2: Other config examples
print("\n\n[2] Other typ_artifact configuration examples")
print("-" * 70)

configs = [
    ([1, 0, 0, 0], "Only Type 0 (device displacement)"),
    ([0, 1, 0, 0], "Only Type 1 (forearm motion)"),
    ([2, 1, 0, 0], "Type 0 is 2x more likely than Type 1"),
    ([1, 1, 1, 1], "All types equal"),
    ([3, 2, 2, 1], "Type 0 most common"),
]

for typ, desc in configs:
    typ_arr = np.array(typ)
    prob = typ_arr / np.sum(typ_arr) if np.sum(typ_arr) > 0 else typ_arr
    Active_types = [i for i, p in enumerate(prob) if p > 0]
    print(f"\n{desc}:")
    print(f"  typ_artifact = {typ}")
    print(f"  Probability distribution: {[f'{p:.0%}' for p in prob]}")

# Test 3: dur_mu0 and dur_mu parameters
print("\n\n[3] dur_mu0 and dur_mu parameter explanation")
print("-" * 70)

print("\nParameter meanings:")
print("  dur_mu0: Mean duration of artifact-free intervals (seconds)")
print("  dur_mu:  Mean duration of artifact intervals (seconds)")

print("\nAdjustment strategies:")
print("\n  Light artifacts (Clean mostly):")
print("    dur_mu0 = 15, dur_mu = 2")
print("    -> Artifacts appear approx every 15s, lasting ~2s")
print("    -> Artifact ratio: ~12%")

print("\n  Medium artifacts (Moderate):")
print("    dur_mu0 = 10, dur_mu = 5")
print("    -> Artifacts appear approx every 10s, lasting ~5s")
print("    -> Artifact ratio: ~33%")

print("\n  Heavy artifacts (Frequent):")
print("    dur_mu0 = 5, dur_mu = 10")
print("    -> Artifacts appear approx every 5s, lasting ~10s")
print("    -> Artifact ratio: ~67%")

print("\n  No artifacts (Completely clean):")
print("    add_artifacts = False")
print("    -> Generates no artifacts")

# Test 4: Artifact ratio estimation
print("\n\n[4] Artifact ratio estimation")
print("-" * 70)

configs = [
    (20, 2),
    (15, 3),
    (10, 5),
    (8, 6),
    (5, 10),
]

print(f"\n{'dur_mu0':>8} {'dur_mu':>7} {'Ratio':>10}")
print("-" * 30)
for mu0, mu in configs:
    ratio = mu / (mu0 + mu)
    print(f"{mu0:>8} {mu:>7} {ratio:>9.0%}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("\nCurrent config: typ_artifact = [1,1,1,1], dur_mu0=10, dur_mu=5")
print("\nEffect:")
print("  [INFO] 4 artifact types appear randomly, equal probability (25% each)")
print("  [INFO] Artifacts appear approx every 10 seconds")
print("  [INFO] Each artifact lasts approx 5 seconds")
print("  [INFO] Artifacts cover ~33% of total duration")
print("\nThis is a medium intensity artifact configuration!")
print("=" * 70)
