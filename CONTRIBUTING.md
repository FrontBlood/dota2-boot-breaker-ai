# 贡献指南

感谢关注本项目。由于本项目涉及游戏自动化和本地输入模拟，贡献时请优先保证行为边界清晰、文档准确、不会引入对抗平台规则的能力。

## 可以提交的内容

欢迎提交：

- 视觉识别稳定性改进；
- 不同分辨率、DPI、显示模式下的兼容性修复；
- 输入释放、安全退出、日志脱敏和发布流程改进；
- 单元测试、回归测试和文档修正；
- 本地模拟器或无用户数据的最小复现样本。

不接受：

- 读取、修改或注入 Dota 2 进程；
- 规避反作弊、隐藏进程、驱动级绕过或封包修改；
- 自动化滥用策略、账号收益最大化或规避封禁指导；
- 包含用户昵称、头像、聊天、日志、截图等未脱敏数据的提交。

## 开发环境

```powershell
python -m pip install -r requirements.txt
python -m pip install pytest
```

常用检查：

```powershell
python -m pytest
python -m compileall boot_breaker lockpick_ai tests
```

## 提交前检查

提交前请确认：

- `git status --short` 中没有 `release/`、`build/`、`dist/`、`boot_runs/`、`runs/`；
- 没有提交 `.env`、日志、数据库、zip、exe 或用户截图；
- 文档描述与实际默认行为一致；
- 新增配置项在 README 或相关说明中有解释；
- 程序异常退出时会释放按键。

## 问题反馈

反馈运行问题时，请尽量提供：

- 操作系统版本；
- Dota 2 显示模式和分辨率；
- 是否使用管理员权限；
- 监控窗口中 `ARMED/LOCKED`、`action`、`boot`、`paddle` 状态；
- 最新 telemetry 的必要片段。

请先打码昵称、头像、聊天内容和本机路径中的真实用户名。