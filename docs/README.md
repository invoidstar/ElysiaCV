# ElysiaCV 文档

> 状态：Active（长期有效） | 最后更新：2026-10

---

## 文档目录

| 目录 | 职责 | 入口 |
|------|------|------|
| `status/` | 当前工程状态、功能清单、路线图 | [status/current.md](status/current.md) |
| `guides/` | 用户操作指南（训练/评测/配置/扩展） | [guides/training.md](guides/training.md) |
| `infrastructure/` | 长期架构合同、性能优化依据 | [infrastructure/architecture.md](infrastructure/architecture.md) |
| `assets/` | 静态资源（架构图、截图） | — |

---

## 常用入口

| 做什么 | 去哪里 |
|--------|--------|
| 第一次使用，想跑一个训练 | `README.md` → 快速开始 → [训练指南](guides/training.md) |
| 新增模型 | [扩展指南](guides/extension.md) |
| 自定义配置 | [配置指南](guides/config.md) |
| 理解架构 | [架构合同](infrastructure/architecture.md) |
| 查看支持的功能 | [当前状态](status/current.md) |
| 了解性能优化 | [性能指南](infrastructure/performance.md) |

---

## 文档维护规则

1. **状态类文档单一事实来源**：当前能力清单只写在 `status/current.md`。
2. **指南只写用户视角**：不暴露内部实现细节，以可执行命令为核心。
3. **架构文档写合同而非实现**：写模块职责边界、依赖关系、数据流方向，不写具体代码。
4. **通用指南不写绝对路径**：所有路径使用 `${VAR}` 或相对路径。
5. **命令可直接执行**：所有 CLI 示例必须先验证可运行再写入文档。
6. **变更记录同步**：新增/删除能力时同步更新 `status/current.md`。