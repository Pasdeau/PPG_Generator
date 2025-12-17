#!/usr/bin/env python3
"""
验证伪影生成机制 - 简化版
"""

import numpy as np

print("=" * 70)
print("伪影参数验证说明")
print("=" * 70)

# 测试1: typ_artifact 参数
print("\n【1】typ_artifact = np.array([1, 1, 1, 1])")
print("-" * 70)

typ_artifact = np.array([1, 1, 1, 1])
prob = typ_artifact / np.sum(typ_artifact)
print(f"归一化概率: {prob}")
print(f"  类型0 (device displacement): {prob[0]:.0%}")
print(f"  类型1 (forearm motion):       {prob[1]:.0%}")
print(f"  类型2 (hand motion):          {prob[2]:.0%}")
print(f"  类型3 (poor contact):         {prob[3]:.0%}")
print(f"\n✅ 结论: 4种伪影会随机出现，每种概率 {prob[0]:.0%}")

# 测试2: 其他配置示例
print("\n\n【2】其他 typ_artifact 配置示例")
print("-" * 70)

configs = [
    ([1, 0, 0, 0], "只有类型0 (device displacement)"),
    ([0, 1, 0, 0], "只有类型1 (forearm motion)"),
    ([2, 1, 0, 0], "类型0出现概率2倍于类型1"),
    ([1, 1, 1, 1], "所有类型均等"),
    ([3, 2, 2, 1], "类型0最常见"),
]

for typ, desc in configs:
    typ_arr = np.array(typ)
    prob = typ_arr / np.sum(typ_arr) if np.sum(typ_arr) > 0 else typ_arr
    Active_types = [i for i, p in enumerate(prob) if p > 0]
    print(f"\n{desc}:")
    print(f"  typ_artifact = {typ}")
    print(f"  概率分布: {[f'{p:.0%}' for p in prob]}")

# 测试3: dur_mu0 和 dur_mu 参数
print("\n\n【3】dur_mu0 和 dur_mu 参数说明")
print("-" * 70)

print("\n参数含义:")
print("  dur_mu0: 无伪影段的平均持续时间(秒)")
print("  dur_mu:  有伪影段的平均持续时间(秒)")

print("\n调整策略:")
print("\n  轻度伪影 (信号大部分时间干净):")
print("    dur_mu0 = 15, dur_mu = 2")
print("    → 平均每15秒出现1次伪影，每次持续约2秒")
print("    → 伪影占比: ~12%")

print("\n  中度伪影 (适度伪影):")
print("    dur_mu0 = 10, dur_mu = 5")
print("    → 平均每10秒出现1次伪影，每次持续约5秒")
print("    → 伪影占比: ~33%")

print("\n  严重伪影 (频繁出现):")
print("    dur_mu0 = 5, dur_mu = 10")
print("    → 平均每5秒出现1次伪影，每次持续约10秒")
print("    → 伪影占比: ~67%")

print("\n  无伪影 (完全干净):")
print("    add_artifacts = False")
print("    → 不生成任何伪影")

# 计算不同配置的伪影占比
print("\n\n【4】伪影占比估算")
print("-" * 70)

configs = [
    (20, 2),
    (15, 3),
    (10, 5),
    (8, 6),
    (5, 10),
]

print(f"\n{'dur_mu0':>8} {'dur_mu':>7} {'伪影占比':>10}")
print("-" * 30)
for mu0, mu in configs:
    ratio = mu / (mu0 + mu)
    print(f"{mu0:>8} {mu:>7} {ratio:>9.0%}")

print("\n" + "=" * 70)
print("总结")
print("=" * 70)
print("\n当前配置: typ_artifact = [1,1,1,1], dur_mu0=10, dur_mu=5")
print("\n效果:")
print("  ✅ 4种伪影随机出现，概率均等(各25%)")
print("  ✅ 平均每10秒出现伪影")
print("  ✅ 每次伪影持续约5秒")
print("  ✅ 伪影占总时长约33%")
print("\n这是一个中等强度的伪影配置！")
print("=" * 70)
