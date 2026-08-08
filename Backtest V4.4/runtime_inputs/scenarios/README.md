# 行情与场景目录

`market_catalog.json` 记录可以在行情选择器中打开的固定行情分段。每个分段绑定品种、评估包、数据文件、时区和截取区间。增加 Chinext 2026、Nasdaq 2026 等行情时，在这里添加一项，再运行行情选择器生成程序。

`scenario_catalog.json` 是场景的机器可读记录。一个场景只能引用一个行情分段，内部可以保存一段或多段选区。场景资格使用固定规则：每段选区内恰好开仓一次、区间内没有平仓、持仓越过区间结束，并在之后以回撤或速度条件退出；多段选区全部合格时，参数才符合该场景。

行情选择器把 `scenario_catalog.json` 当作一个多场景目录管理。载入按钮导入整份目录，不代表打开某一项行情；保存按钮会通过 Chrome 的保存窗口写出包含全部场景的完整 JSON。新场景的 ID 自动取下一个未使用的 `scenario_N`，名称默认取下一个未使用的「新场景N」，名称仍可修改。保存完成后页面会准备下一项新场景草稿。覆盖本目录里的同名文件后，场景应用脚本即可读取新结果；若页面内置目录还没有刷新，可以使用「载入场景目录」导入刚保存的 JSON。

框选模式只记录时间范围。拖出的选区会自动铺满图表上下边界，保存到场景目录的也是全高时间带，不使用价格轴上的二维矩形作为资格边界。

运行：

```powershell
python "tools\build_v4_41_scenario_manager.py"
python "tools\apply_v4_41_scenario.py" --scenario-id scenario_1
python "tools\apply_v4_41_scenario.py" --all
```

生成的行情选择器位于 `results\market_scenario_manager\index.html`。场景排序页位于 `results\scenario_analysis\<scenario_id>\index.html`，页面结构、筛选、排序和逐笔入口沿用当前总入口。
