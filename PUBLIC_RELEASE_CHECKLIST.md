# 公开发布检查记录

本文件记录本项目 GitHub 化前的公开范围、排除内容和待确认事项。它用于私有远端复核和后续公开前追溯。

## 目标公开内容

计划进入源码仓库：

- `boot_breaker/`：破牢之靴主控制器；
- `lockpick_ai/`：早期撬锁控制器与共用工具；
- `tests/`：单元测试；
- `boot_config.json`、`config.example.json`、`config.json`：本地默认配置；
- `boot_collision_model.json`：挡板碰撞标定模型；
- `requirements.txt`；
- `build_release.ps1`；
- `release_launchers/`；
- 根目录启动 `.bat`；
- README、隐私说明、安全说明、发布说明。

## 不进入源码仓库

以下内容必须保持未跟踪：

- `release/`：对外 zip、exe 和 PyInstaller 运行库；
- `build/`、`dist/`：构建产物；
- `boot_runs/`、`runs/`：遥测、分数样本和用户调试数据；
- 用户反馈截图、录像、聊天记录；
- 虚拟环境、缓存、数据库、日志、压缩包、可执行文件。

## 已执行检查

- 当前仓库尚无提交历史，适合从审核后的源码建立第一条干净历史；
- `.gitignore` 已排除运行日志、遥测、构建产物和发布包；
- 未在跟踪候选文件中发现明显 token、API key、证书或本机绝对路径；
- `release/`、`build/`、`dist/`、`boot_runs/` 当前由 `.gitignore` 排除；
- README 已调整为外部读者首页，并说明数据保存位置、权限要求、风险边界和 MIT 许可证。

## 待确认事项

- 已采用 MIT License，许可证正文见 `LICENSE`；
- 当前环境未安装 `pytest`，需要在安装测试依赖后运行完整测试；
- 推送到 GitHub 私有仓库后，需要在网页端复核 README、文件列表和提交历史；
- 若后续要公开发布 zip，应通过 GitHub Releases 上传，不应提交到 Git 历史。

## 推荐最终门槛

公开前执行：

```powershell
git status --short
rg -n -i "token|api[_-]?key|secret|password|credential|bearer|authorization|private[_-]?key|C:\\|Users\\|\.env|steamid|webhook" . --glob '!release/**' --glob '!build/**' --glob '!dist/**' --glob '!boot_runs/**' --glob '!.git/**'
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest
python -m compileall boot_breaker lockpick_ai tests
```

然后先推送到私有远端，确认无误后再考虑 Public。