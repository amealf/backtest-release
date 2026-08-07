# 当前任务清单

## 用途

使用稳定任务标识、可验证验收标准与简明交接状态记录用户声明的工作。不能只根据打开的应用、分支或文件推断任务。

## 当前登记

| task_id | 标题 | 状态 | 目标 | 验收 | 更新时间 |
| --- | --- | --- | --- | --- | --- |
| `V43-P0-EXECUTION` | V4.3 成交真实性修复 | 已完成 | 避免信号平仓在无真实成交时成交，同时保留已确认的 pending-entry 行为。 | 开仓／平仓回归测试通过；V4.3 身份独立。 | 2026-08-01 |
| `V43-PORTABLE-RUNTIME` | 仓库相对运行输入 | 已完成 | 去除活跃 D:/F: 运行依赖，同时保留来源记录。 | 运行 manifest 闭合本地数据／模板；活跃代码从当前文件夹解析。 | 2026-08-01 |
| `V43-DUAL-BASELINE` | 双 baseline 采样策略 | 已完成 | 保持 `all_window` 为默认，并增加可选择的 `exclude_marked`，使用独立身份与排名。 | 中性准备标记、runner／manifest 策略绑定、固定模板策略控件与回归测试。 | 2026-08-02 |
| `SHARED-SCENARIO-MANAGER` | 跨版本场景管理器 | 已完成 | 复用指定选定器，让各版本共用一个场景库。 | 保存采用不会复用的连续 `情景N` 名称；当前场景以两行按钮展示并可点击载入；归档／恢复继续有效；默认 15 秒及 1／5／15／30／60／120 分钟周期、下方上传入口通过响应式 QA；原始选定器与历史快照保持不变。 | 2026-08-02 |
| `V43-NEXT-CAMPAIGN` | V4.3 max-W 有界多轮 campaign | 已完成 | 在四个独立视图下运行宽广、可审计的多轮 `all_window` 探索与终局交互细化。 | Round 1 smoke、504 坐标 Round 2、72 坐标 Round 3、不可变原始／交付／QA 闭合、精确反连接、终局决定、无 Round 4、无参数接受。 | 2026-08-02 |
| `V43-ROUND4-BOUNDARY` | 用户授权的 Round 4 边界续轮 | 已完成 | 在不变的四视图下，使用三个彼此独立的 24 点 `all_window`／`rolling_tr_sum`／S480 块检验三个终局领导者的未解边界。 | 72 坐标计划冻结；精确反连接与 compute-only 闭合；不可变原始证据、稳定固定模板交付、canonical why／run／observed／next 记录，且无参数接受。 | 2026-08-02 |
| `V43-ROUND5-TERMINAL` | 终局双块微型细化 | 已完成 | 只检验通过 Round 4 续轮规则的情景一与无限制总收益边界；两个笔均收益视图保持停止。 | 24 坐标计划冻结；精确反连接与 3／12／4096 compute-only 闭合；不可变原始证据与稳定交付；canonical 终局解释；无 Round 6、无参数接受。 | 2026-08-02 |

## 交接状态

- Round 1 是实现／smoke probe，原始与交付均已闭合：12 个坐标、1,196 笔交易、12 个 lazy chunk、四个 review worker，阶段与 union 各通过 160 个浏览器状态。
- `all_window` 保持默认，`exclude_marked` 为可选。排除 synthetic 或 real-only 策略仍属于未批准的研究范围。
- Round 1 计算时来源 manifest SHA-256 保持为 `78fcea5452769fe14f531c4e3036fe7ba3b2513092a0ad1e43a5b7d495c949ec`。当前 gap-contract 来源 manifest 已闭合为 SHA-256 `306bf83ca50dab913fc7d8f081ec249dc05e03529ced7ce1c6dc58addab3f946`。
- 四个主视图只使用完整总收益与笔均收益。Gap-excluded return 只是显示用 gap 依赖审计，不能过滤或排名资格、续轮或候选。
- Round 2 原始证据按 504 个坐标、42 个 batch 与 73,306 笔交易完成不可变闭合，fingerprint 为 `307d2d8adbec3d7f0915b19ce51db65edd58d86111d491b21a91a7e060ed8095`；completion SHA-256 为 `863eb92646e5ae78d6d2bd72f6c9e5d24acfb311b069e17c83c4fd94894eb2cc`。固定模板交付证据 SHA-256 为 `af17ff903b66877bcebae41d1bc7e1e6f041847b091945a573f17c8df2059160`。
- Round 3 已按 72 个坐标、37,618 笔交易和交付证据 SHA-256 `7b8f1450c9b53b84ad5d5d2e3ddd51fa6d84147624b41b4135cbe9d29262bf6f` 完成终局闭合。累计结果包含三个阶段、588 个坐标与 112,120 笔交易。
- 最终视图专属候选为：情景一总收益 E30/BH720/TRW6/K1.5/W1/M2；无限制总收益 E120/BH360/TRW6/K0.75/W1/M0.25；两个笔均收益视图 E200/BH720/TRW24/K0.75/W192/M10。它们只属于相互依赖的样本内候选；没有参数被接受。
- Round 4 已稳定闭合：72 坐标／73,373 笔交易；四个阶段累计 660／185,493；交付 QA 证据 SHA-256 `0d5d3187de8a2a1297bbb35f1de220930d34ed30c117906b936506cd404b7fb8`。
- Round 5 已按 24 坐标／46,732 笔交易完成终局闭合；五个阶段累计交付为 684 坐标／232,225 笔交易。不可变证据 SHA-256 为 `ae53e778944aa3fd4ed41627655fbaeb07aa863632110a0073728bba11b784c3`；交付 QA 证据 SHA-256 为 `6d3cc38cfe98d16d2b095990f2f6a8c85d69c230b9b873bae61d5c414c6751a0`。不授权 Round 6。
- 纠正后的 Round 1 交付 QA artifact SHA-256 为 `d0f88bc6a07c0ec0b5f8a298998aeee900a5a8a47b3dc9c581202bde14703b7c`；原始文件保持不可变。
- V4.2 文件与结果保持在活跃 V4.3 边界之外。
- 任何场景使用请求都应打开 `D:\Code\backtest-release\shared_tools\scenario_manager\index.html`，再把已审阅定义绑定到目标版本的计划。

## 状态词汇

使用「计划中」「进行中」「等待输入」「已阻塞」「已完成」或「已存档」。阻塞项需要说明原因与所需外部变化。

## 维护规则

用户声明、完成、阻塞、恢复任务或改变任务范围时更新。工作完成或发生重要变化时，在本文保留简洁任务状态，并把结果、证据、限制与交接路径记录到 `WORK_PROGRESS.en.md`。被替代的细节可以移入存档，同时保留可追溯指针。
