(() => {
  "use strict";

  const root = document.documentElement;
  const themeButton = document.getElementById("theme-toggle");
  const state = { data: null };
  const $ = (id) => document.getElementById(id);
  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
  }[character]));
  const number = (value, digits = 4) => value === null || value === undefined || Number.isNaN(Number(value))
    ? "—"
    : Number(value).toLocaleString("zh-CN", { maximumFractionDigits: digits });
  const percent = (value, digits = 2) => value === null || value === undefined || Number.isNaN(Number(value))
    ? "—"
    : `${(Number(value) * 100).toFixed(digits)}%`;
  const signed = (value, formatter = number) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return "—";
    return `${numeric > 0 ? "+" : ""}${formatter(numeric)}`;
  };
  const methodLabels = { rolling_tr_sum: "滚动 TR 总和" };
  const baselinePolicyLabels = { all_window: "全部", exclude_marked: "排除标记" };
  const objectiveLabels = {
    absolute_return: "总收益",
    average_trade: "笔均收益",
    gap_excluded_return: "排除缺口收益",
    low_drawdown: "低回撤",
    event_01: "event_01",
    event_02: "event_02",
    dual_event_and: "双事件 AND",
    short_drop_3_15m: "3–15 分钟短跌",
  };

  function applyTheme(theme, persist = true) {
    const resolved = theme === "dark" ? "dark" : "light";
    root.dataset.theme = resolved;
    themeButton.textContent = resolved === "dark" ? "◑" : "◐";
    themeButton.setAttribute("aria-pressed", String(resolved === "dark"));
    if (persist) {
      try { localStorage.setItem("backtest-research-hub-theme", resolved); } catch (error) {}
    }
  }

  function loadScript(source) {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = source;
      script.async = true;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error(`无法载入 ${source}`));
      document.head.appendChild(script);
    });
  }

  function tradeRoute(comboId) {
    return `trade_analysis/index.html?combo_id=${encodeURIComponent(comboId)}`;
  }

  function parameterText(row) {
    return `E${number(row.e)} · BH${number(row.bh)} · TRW${number(row.trw)} · K${number(row.k)} · W${number(row.w)} · M${number(row.m)}`;
  }

  function renderPreparation(data) {
    const facts = [
      ["准备身份", `<span class="mono">${escapeHtml(data.identity.prepared_identity)}</span>`],
      ["15 秒原子", `${number(data.preparation.atom_count)} 根`],
      ["正式排除", `${number(data.preparation.baseline_excluded_atom_count)} 根 · ${number(data.preparation.baseline_excluded_minutes)} 分钟`],
      ["pending／恢复重入", `${number(data.preparation.buffer_reinserted_atom_count)} 根 pending／恢复原子在确认后重入`],
      ["恢复确认", `${number(data.preparation.recovery_confirmation_count)} 次`],
      ["审计入口", `<a href="${escapeHtml(data.preparation.audit_url)}" target="_blank" rel="noreferrer">打开 V4.4 低成交量与机制审计</a>`],
    ];
    $("preparation-facts").classList.remove("loading-block");
    $("preparation-facts").innerHTML = facts.map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`).join("");
  }

  function renderRepair(data) {
    const counts = data.counts;
    $("repair-summary").classList.remove("loading-block");
    $("repair-summary").innerHTML = [
      `<span><strong>${number(counts.validation_coordinates)}</strong> 个坐标</span>`,
      `<span><strong>${number(counts.validation_trades)}</strong> 笔 V4.4 交易</span>`,
      `<span><strong>${number(counts.validation_waited_entries)}</strong> 笔等待真实成交</span>`,
      `<span>最长等待 <strong>${number(counts.validation_maximum_wait)}</strong> 根</span>`,
      `<span>结果：<strong>未接受参数</strong></span>`,
    ].join("");
    const rows = data.validation;
    $("repair-table").innerHTML = `<table>
      <thead><tr><th>方法／参数</th><th>V4 笔数</th><th>V4.4 笔数</th><th>笔数变化</th><th>V4 收益</th><th>V4.4 收益</th><th>收益变化</th><th>回撤变化</th><th>等待笔数</th><th>最长等待</th></tr></thead>
      <tbody>${rows.map((row) => {
        const combo = row.v4_4_combo_id;
        const parameters = `E${number(row.e_v4_4)} · BH${number(row.bh_v4_4)} · TRW${number(row.trw_v4_4)} · K${number(row.k_v4_4)} · W${number(row.w_v4_4)} · M${number(row.m_v4_4)}`;
        const returnDelta = Number(row.return_delta_v4_4_minus_v4);
        const drawdownDelta = Number(row.max_drawdown_abs_delta_v4_4_minus_v4);
        return `<tr>
          <td class="text-cell" title="${escapeHtml(combo)}"><strong>${escapeHtml(methodLabels[row.method_v4_4] || row.method_v4_4)}</strong><br><span class="value-muted">${escapeHtml(parameters)}</span></td>
          <td>${number(row.v4_train_trade_count)}</td><td>${number(row.train_trade_count)}</td><td>${signed(row.trade_count_delta_v4_4_minus_v4)}</td>
          <td>${percent(row.v4_train_return)}</td><td>${percent(row.train_return)}</td>
          <td class="${returnDelta >= 0 ? "value-good" : "value-bad"}">${signed(returnDelta, percent)}</td>
          <td class="${drawdownDelta <= 0 ? "value-good" : "value-bad"}">${signed(drawdownDelta, percent)}</td>
          <td>${number(row.v4_4_waited_entry_trade_count)}</td><td>${number(row.v4_4_max_entry_wait_bar_count)}</td>
        </tr>`;
      }).join("")}</tbody>
    </table>`;
  }

  function renderExploration(data) {
    const counts = data.counts;
    const improvedReturn = data.comparison.filter((row) => Number(row.return_delta) > 0).length;
    const improvedGap = data.comparison.filter((row) => Number(row.gap_return_delta) > 0).length;
    const improvedDrawdown = data.comparison.filter((row) => Number(row.drawdown_delta) < 0).length;
    $("exploration-summary").classList.remove("loading-block");
    $("exploration-summary").innerHTML = [
      `<span><strong>${number(counts.promising_coordinates)}</strong> 个新计算坐标</span>`,
      `<span><strong>${number(counts.promising_trades)}</strong> 笔交易</span>`,
      `<span><strong>${number(counts.promising_waited_entries)}</strong> 笔等待真实成交</span>`,
      `<span>收益改善 <strong>${number(improvedReturn)} / 24</strong></span>`,
      `<span>排除缺口收益改善 <strong>${number(improvedGap)} / 24</strong></span>`,
      `<span>绝对回撤改善 <strong>${number(improvedDrawdown)} / 24</strong></span>`,
    ].join("");
    const items = data.objective_summary.map((row) => ({
      label: `${methodLabels[row.method] || row.method} · ${objectiveLabels[row.selection_objective] || row.selection_objective}`,
      copy: `2 个坐标 · V4.4 平均收益 ${percent(row.v4_4_return_mean)} · 平均变化 ${signed(row.return_delta_mean, percent)} · 等待 ${number(row.waited_entry_trade_count)} 笔`,
    }));
    $("objective-summary").classList.remove("loading-block");
    $("objective-summary").innerHTML = items.map((item) => `<div class="evidence-item"><strong>${escapeHtml(item.label)}</strong><span>${escapeHtml(item.copy)}</span></div>`).join("");
  }

  function renderRankings() {
    const data = state.data;
    if (!data) return;
    const method = $("ranking-method").value;
    const baselinePolicy = $("ranking-baseline-policy").value;
    const objective = $("ranking-objective").value;
    const minimum = Number($("ranking-minimum").value);
    const comboMap = new Map(data.combos.map((row) => [row.combo_id, row]));
    const rankingRows = data.rankings
      .filter((row) => row.objective === objective && Number(row.minimum_trade_count) === minimum)
      .map((row) => ({ ...row, combo: comboMap.get(row.combo_id) }))
      .filter((row) => row.combo && row.combo.method === method && row.combo.baseline_sampling_policy === baselinePolicy)
      .sort((left, right) => left.rank - right.rank);
    $("ranking-status").textContent = `${methodLabels[method]} · ${baselinePolicyLabels[baselinePolicy]} · ${objectiveLabels[objective]} · ${minimum ? `至少 ${minimum} 笔` : "不限笔数"} · ${rankingRows.length} 个坐标`;
    if (!rankingRows.length) {
      $("ranking-table").innerHTML = `<div class="empty-state"><h3>当前组合形成真实空集</h3><p>该事件资格在本次 24 坐标 V4.4 population 中没有成员；闭合结果保持不变。</p></div>`;
      return;
    }
    $("ranking-table").innerHTML = `<table>
      <thead><tr><th>排名</th><th>参数</th><th>交易数</th><th>总收益</th><th>排除缺口收益</th><th>笔均</th><th>回撤</th><th>缺口笔数</th><th>synthetic 信号</th><th>事件</th></tr></thead>
      <tbody>${rankingRows.map(({ rank, combo }) => `<tr>
        <td><a class="inspect-button" href="${tradeRoute(combo.combo_id)}">查看 #${number(rank)}</a></td>
        <td class="text-cell"><span class="value-muted">${escapeHtml(parameterText(combo))}</span></td>
        <td>${number(combo.train_trade_count)}</td><td>${percent(combo.train_return)}</td><td>${percent(combo.train_return_excluding_gap_spanning_trades)}</td>
        <td>${percent(combo.train_avg_trade, 3)}</td><td>${percent(combo.train_max_drawdown_abs)}</td><td>${number(combo.gap_spanning_trade_count)}</td>
        <td>${number(combo.synthetic_signal_trade_count)}</td><td>${combo.event_01_qualified ? "event_01" : ""}${combo.event_01_qualified && combo.event_02_qualified ? " + " : ""}${combo.event_02_qualified ? "event_02" : ""}${!combo.event_01_qualified && !combo.event_02_qualified ? "—" : ""}</td>
      </tr>`).join("")}</tbody>
    </table>`;
  }

  function reasonText(reasons) {
    return String(reasons || "").split("|").filter(Boolean).map((reason) => {
      const [method, objective] = reason.split(":");
      return `${methodLabels[method] || method}／${objectiveLabels[objective] || objective}`;
    }).join("；");
  }

  function renderShortlist(data) {
    $("shortlist-table").innerHTML = `<table>
      <thead><tr><th>交易入口</th><th>方法／参数</th><th>入选原因</th><th>交易数</th><th>总收益</th><th>排除缺口收益</th><th>回撤</th><th>缺口笔数</th></tr></thead>
      <tbody>${data.shortlist.map((row) => `<tr>
        <td><a class="inspect-button" href="${tradeRoute(row.combo_id)}">逐笔查看</a></td>
        <td class="text-cell">${escapeHtml(methodLabels[row.method] || row.method)}<br><span class="value-muted">${escapeHtml(parameterText(row))}</span></td>
        <td class="text-cell combo-cell" title="${escapeHtml(reasonText(row.shortlist_reasons))}">${escapeHtml(reasonText(row.shortlist_reasons))}</td>
        <td>${number(row.train_trade_count)}</td><td>${percent(row.train_return)}</td><td>${percent(row.train_return_excluding_gap_spanning_trades)}</td>
        <td>${percent(row.train_max_drawdown_abs)}</td><td>${number(row.gap_spanning_trade_count)}</td>
      </tr>`).join("")}</tbody>
    </table>`;
  }

  function renderProvenance(data) {
    const rows = [
      ["strategy_id", data.identity.strategy_id],
      ["result_semantics_id", data.identity.result_semantics_id],
      ["source SHA-256", data.identity.source_sha256],
      ["preparation manifest", data.identity.preparation_manifest_sha256],
      ["8-coordinate manifest", data.identity.validation_manifest_sha256],
      ["24-coordinate manifest", data.identity.promising_manifest_sha256],
    ];
    $("provenance-list").classList.remove("loading-block");
    $("provenance-list").innerHTML = rows.map(([label, value]) => `<div class="provenance-row"><span>${escapeHtml(label)}</span><code>${escapeHtml(value)}</code></div>`).join("");
  }

  function render(data) {
    state.data = data;
    renderPreparation(data);
    renderRepair(data);
    renderExploration(data);
    renderRankings();
    renderShortlist(data);
    renderProvenance(data);
    window.V41_HOME_APP = {
      dataLoaded: true,
      coordinateCount: data.combos.length,
      rankingRowCount: data.rankings.length,
      shortlistCount: data.shortlist.length,
    };
  }

  function fail(error) {
    const message = `资料载入失败：${error.message}`;
    ["repair-table", "ranking-table", "shortlist-table"].forEach((id) => {
      const node = $(id);
      if (node) node.innerHTML = `<div class="empty-state"><h3>V4.4 资料未载入</h3><p>${escapeHtml(message)}</p></div>`;
    });
    window.V41_HOME_APP = { dataLoaded: false, error: error.message };
  }

  applyTheme(root.dataset.theme, false);
  themeButton.addEventListener("click", () => applyTheme(root.dataset.theme === "dark" ? "light" : "dark"));
  ["ranking-method", "ranking-baseline-policy", "ranking-objective", "ranking-minimum"].forEach((id) => $(id).addEventListener("change", renderRankings));
  const schedule = window.requestIdleCallback || ((callback) => setTimeout(callback, 0));
  schedule(() => loadScript("assets/home_data.js")
    .then(() => {
      if (!window.V41_HOME_DATA) throw new Error("home_data.js 没有发布 V41_HOME_DATA");
      render(window.V41_HOME_DATA);
    })
    .catch(fail));
})();
