# Dota 2 破牢之靴视觉控制器

> 面向 Dota 2 活动小游戏“破牢之靴”的实验性屏幕视觉控制器。程序只读取屏幕像素并在用户明确解锁后模拟键盘输入，不读取或修改 Dota 2 进程内存。

## 项目状态

当前仓库以“破牢之靴”为主项目，早期“撬锁”控制器仍保留在 `lockpick_ai/` 中，作为同一套视觉控制框架的历史模块和参考实现。

本项目适合用于学习以下内容：

- Windows 屏幕捕获与窗口客户区定位；
- OpenCV 颜色分割、连通域分析和帧差运动检测；
- 目标身份确认、短时轨迹回归和反弹落点预测；
- 安全释放键盘输入、热键控制和发布包制作。

## 重要风险说明

使用自动输入工具可能违反游戏、活动或平台规则，并可能带来账号、活动资格或其他风险。使用者应自行确认相关规则并承担后果。

本项目不提供“安全过检测”“规避反作弊”或任何对抗平台规则的功能。仓库中的代码不包含驱动、注入、内存读写、封包修改或系统启动项修改。

## 功能概览

破牢之靴控制器会：

1. 自动寻找可见的 `dota2.exe` 窗口；
2. 以 Dota 2 客户区为坐标系捕获小游戏画面；
3. 在画面中自动对齐小游戏内容边框；
4. 识别红色挡板、中央发射提示、发射虚线、实时分数和飞行靴子；
5. 使用颜色、运动和靴子弧形几何证据确认靴子身份；
6. 通过短时轨迹估计 `vx/vy`，预测靴子落到挡板接触线时的横坐标；
7. 在用户按 `F8` 解锁后模拟 `A/D` 和空格键。

更详细的策略、视觉规则和调试字段见 [BOOT_BREAKER.md](BOOT_BREAKER.md)。

## 运行条件

- Windows 10/11 64 位；
- Python 3.11+；
- Dota 2 使用 16:9 画面更稳；
- 推荐窗口模式或无边框窗口；
- 输入法建议切换到英文；
- 若 Dota 2 以管理员权限运行，控制器也需要同等权限。

## 安装依赖

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` 只包含运行所需依赖：

- `numpy`
- `opencv-python`
- `mss`

测试需要额外安装 `pytest`。

## 启动方式

开发环境下可使用根目录批处理：

```text
start_boot_observe.bat        # 观察模式，不发送按键
start_boot_control.bat        # 控制模式，带监控窗口
start_boot_control_fast.bat   # 控制模式，不显示监控窗口
```

也可以从终端启动：

```powershell
python -m boot_breaker
python -m boot_breaker --control
python -m boot_breaker --control --no-window
python -m boot_breaker --smoke-test
```

启动后默认保持锁定。切回 Dota 2 后按 `F8` 才会开始自动控制。

快捷键：

- `F8`：解锁/锁定自动控制；
- `F9`：暂停并释放方向键；
- `Esc`：退出并释放方向键。

## 数据保存位置

程序默认会在本地写入：

- `boot_runs/telemetry_*.jsonl`：每帧识别、决策、分数和窗口状态；
- `boot_runs/score_samples/`：分数变化时的小型分数区域样本；
- `runs/`：早期撬锁模块的遥测输出。

这些文件可能间接包含用户游戏画面、昵称、分数或本机运行环境信息。它们已被 `.gitignore` 排除，不应提交到公开仓库或发布包。

详见 [PRIVACY.md](PRIVACY.md)。

## 构建发布包

发布包使用 PyInstaller 生成。先安装构建依赖：

```powershell
python -m pip install pyinstaller
```

然后运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_release.ps1 -Version 1.0.0
```

输出位于 `release/`，该目录不会进入 Git。给外部用户下载的 zip 应通过 GitHub Releases 发布，而不是提交到仓库历史中。

发布包内只保留面向用户的启动器和使用指南：

- `启动控制器_带监控.bat`
- `启动控制器_无监控.bat`
- `使用指南.md`
- `Dota2-Boot-Breaker-AI.exe`
- `_internal/`

## 测试

```powershell
python -m pytest
python -m compileall boot_breaker lockpick_ai tests
```

如果当前环境没有安装 `pytest`，先执行：

```powershell
python -m pip install pytest
```

## 仓库公开检查

公开前请至少确认：

- `release/`、`build/`、`dist/`、`boot_runs/` 未被 Git 跟踪；
- 没有 telemetry、用户截图、发布 zip、exe 或本机私有文件进入提交历史；
- README、`RELEASE_README.md` 和实际程序默认行为一致；
- 已选择并加入明确许可证；
- 已在私有远端复核 GitHub 页面显示内容。

当前发布审计记录见 [PUBLIC_RELEASE_CHECKLIST.md](PUBLIC_RELEASE_CHECKLIST.md)。

## 许可证

本项目采用 [MIT License](LICENSE)。

## 安全反馈

如果发现误打包敏感文件、异常系统行为或其他安全问题，请先按 [SECURITY.md](SECURITY.md) 中的方式私下报告，不要直接公开日志、截图或用户数据。