# V4.4 策略协议

## 用途

本文定义与品种无关的交易协议。品种数据、交易时段、成本、gap 规则、低活跃阈值和事件日期归入品种配置或实验清单。当前策略的数据颗粒度协议为每个执行原子对应一根已经完成的 15 秒 bar；迁移目标必须使用相同颗粒度。

## 开仓协议

- 寻找有序高点锚 H，并从 H 计算 E 窗口下跌。
- 使用 BH 内已经完成的 TR 原子并按 TRW 分组，形成开仓基准。
- 开仓阈值为 `max(K × baseline, absolute floor)`。
- 开仓资格要求 `baseline`、`drop` 和 `threshold` 都是有限值，同时满足 `baseline > 0`、`drop > 0`、`threshold > 0` 与 `drop >= threshold`。正值边界相等仍可开仓；零 baseline、零 drop 或零 threshold 不能开仓。
- calculated-threshold 信号按阈值成交；真实 bar 的 open 穿过阈值时使用 open。
- 非真实 bar 形成的信号会保留，并在 120 根连续候选 bar 内按第一根真实成交 bar 的 open 开仓。

## 平仓协议

- W 使用 1 到 W 的可用前缀，计算 `open[start] - low[end]`。
- W 起点不早于 H、连续交易段起点或 `end-W+1`。
- M 把本次持仓已经完成的最大 W 下跌转换为回撤阈值。
- 旧理论线按 `open >= trigger` 退出；否则按 `high >= trigger` 在理论线退出；相等也退出。
- 同 bar 出现严格新低时，使用 close 确认并按该 close 平仓。
- S 判断下跌停止推进。非真实 bar 的平仓等待下一根真实成交 bar 的 open。
- 样本结束仍有持仓时，按样本结束 bar 的 close 强制平仓。

## 通用状态机

flat reset、calculated-threshold fill、synthetic bar、pending entry、pending exit 与确认后低活跃门禁都属于策略核心。低成交量连续段处于待确认状态时，对策略没有影响：原子可以进入 baseline，也可以开仓；确认前恢复时，待确认状态取消。连续第 120 根低成交量 15 秒原子完成时，过滤正式成立：从第一根低成交量原子到确认原子全部退出此后每次 baseline 计算，未成交开仓委托取消，并从确认原子起停止新开仓，直到最后一根低成交量原子结束。第一根正常成交量原子结束门禁，并立即允许新的 baseline 采样与开仓判断。已有持仓继续使用原有平仓规则。该规则只使用确认时已经完成的数据，不读取未来状态。

## 品种边界

策略核心不包含品种名称、交易所、币种、合约乘数、手续费、滑点、情景日期或交易时钟。它会绑定共享的 15 秒执行颗粒度与正值开仓资格。可执行 JSON 权威为 `research_variants\short_momentum_net_drop_rebound_v4_4\contracts\STRATEGY_CONTRACT_V4_4.json`。
