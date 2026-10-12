# AGENTS.md — docs/

> 本目录存放项目文档。修改代码后，务必同步更新相关文档。

---

## 目录约定

```
docs/
├── guides/              # 使用指南
│   ├── config.md        #   配置系统说明
│   ├── extension.md     #   扩展新模型/数据集指南
│   └── training.md      #   训练命令参考
├── infrastructure/      # 基础设施文档（长期有效）
│   ├── architecture.md  #   模块职责边界（权威参考）
│   └── performance.md   #   性能优化记录
├── status/
│   └── current.md       #   当前开发状态（每次功能变更后更新）
└── assets/readme/       # README 中使用的 SVG 资源
```

---

## 更新规则

| 改了哪里 | 需要更新哪些文档 |
|----------|-----------------|
| 添加新模型 | `extension.md` + `status/current.md` |
| 修改 CLI 接口 | `training.md` + `status/current.md` |
| 修改模块依赖 | `infrastructure/architecture.md` |
| 修复 bug | `status/current.md`（更新已知问题） |
| 修改配置格式 | `guides/config.md` |

---

## `status/current.md` 维护约定

- 每次重要改动后更新"最近核对"日期
- "已有实现"列出已验证可工作的模块
- "尚未闭环"列出已知缺口和占位实现
- 使用以下标记：
  - ✅ 已闭环
  - ⚠️ 部分完成
  - ❌ 未实现