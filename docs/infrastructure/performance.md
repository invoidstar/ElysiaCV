# 性能指南

> 状态：Active（长期有效） | 最后更新：2026-10

本文档记录 ElysiaCV 的训练性能优化策略，均来自 ElysiaRobot 的已验证实践。

---

## DDP 配置

```yaml
engine:
  distributed:
    ddp_find_unused_parameters: true    # 检测未使用参数（新模型建议开启）
    ddp_broadcast_buffers: false        # 不广播 BN buffers（PyTorch 已处理）
    ddp_bucket_cap_mb: 64               # 大 bucket 减少通信轮次
    ddp_gradient_as_bucket_view: true   # 零拷贝梯度视图（省显存）
    ddp_static_graph: true              # 跳过每步图遍历（Backward p50 +45.6%）
```

**注意**：如果模型有动态分支（如某些 Transformer 的条件层），必须设置 `ddp_static_graph: false`。

---

## 梯度累积

```
micro_0    micro_1    micro_2    micro_3
(no_sync)  (no_sync)  (no_sync)  (sync + clip + step)
```

- 仅最后 micro-step 触发 `all-reduce` + `optimizer.step()`
- 前向 loss 自动归一化：`scaled_loss = loss / grad_accum_steps`
- 梯度裁剪只执行一次（在 step 之前）

```yaml
engine:
  optimization:
    gradient_accumulation_steps: 4
    grad_clip_norm: 10.0
    optimizer_zero_grad_set_to_none: true    # 避免分配零张量
```

---

## DataLoader 最佳实践

| 参数 | 推荐值 | 原因 |
|------|--------|------|
| `num_workers` | 2-4 | 8 卡验证：worker 从 2→4 收益递减，过多无益 |
| `prefetch_factor` | 2 | 适度预取，`prefetch × workers × batch` 会放大内存占用 |
| `pin_memory` | `true` | 非阻塞 H2D 传输 |
| `persistent_workers` | `true` | 避免 worker 反复初始化（CV 图片场景收益显著） |
| `multiprocessing_context` | `"spawn"` | 防止 CUDA 初始化后 fork 导致死锁 |

```yaml
data:
  train:
    num_workers: 4
    prefetch_factor: 2
    pin_memory: true
    persistent_workers: true
```

---

## 指标计算

```python
# ❌ 错误：每步对每个指标单独 .item() 和 all_reduce
for name, metric_tensor in metrics.items():
    reduced = all_reduce(metric_tensor)  # 每个指标一次通信
    log(name, reduced.item())

# ✅ 正确：批量向量 all_reduce，.item() 只在日志边界执行
metrics_vector = torch.stack(list(metrics.values()))
reduced = all_reduce_vector(metrics_vector)     # 一次通信
log(reduced)  # 仅在 log 边界 detach → python
```

这样将 400+ 次通信减少为 1 次。

---

## CUDA 后端

```python
# 训练开始前自动配置
torch.backends.cudnn.benchmark = True          # 卷积自动调优
torch.backends.cuda.matmul.allow_tf32 = True   # TF32 加速
torch.set_float32_matmul_precision("high")
```

`configs/_base_/default_runtime.yaml` 中已预设，无需手动配置。

---

## torch.compile

```yaml
engine:
  compile:
    enabled: false     # 默认关闭
    mode: "default"
```

| 场景 | 建议 |
|------|------|
| debug / smoke test | 关闭（`false`） |
| 单卡 < 5000 step | 关闭（编译开销 > 收益） |
| 长训 > 20000 step | 开启（1.5-1.8x 稳态加速） |

**代价**：首个 micro batch 编译约 5 分钟。

---

## 常见性能陷阱

| # | 陷阱 | 影响 | 解决 |
|---|------|------|------|
| 1 | CUDA init 后 fork DataLoader worker | 死锁 | `multiprocessing_context="spawn"` |
| 2 | `persistent_workers` + shutdown → native abort | SIGABRT | 依赖进程组退出，不强制 shutdown |
| 3 | `static_graph=true` 遇到动态分支 | 梯度错误 | 关闭 static_graph |
| 4 | `prefetch_factor × workers × world_size` 内存爆炸 | OOM | 默认 prefetch=2, workers=2 |
| 5 | 梯度累积时 loss 直接 backward 不除累积步数 | 收敛问题 | 框架层自动归一化 `loss / accum` |
| 6 | 热路径中 `.item()` 同步 CUDA→CPU | 严重变慢 | 梯度范数只在 log 边界 item() |