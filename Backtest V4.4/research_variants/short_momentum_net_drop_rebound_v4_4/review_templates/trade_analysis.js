(() => {
  "use strict";

  const PAGE_SIZE = 100;
  const root = document.documentElement;
  const darkThemeButton = document.getElementById("darkThemeBtn");
  const lightThemeButton = document.getElementById("lightThemeBtn");
  const $ = (id) => document.getElementById(id);
  const state = {
    catalog: null,
    comboMap: new Map(),
    filtered: [],
    page: 1,
    selectedCatalog: null,
    selectedDetail: null,
    selectedChunk: null,
    chartMode: "entry",
    loadedChunkKeys: new Set(),
  };
  const filterIds = ["combo", "method", "baseline-policy", "e", "bh", "trw", "k", "w", "m", "speed", "wait", "exit", "gap", "synthetic"];
  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
  }[character]));
  const number = (value, digits = 4) => value === null || value === undefined || Number.isNaN(Number(value))
    ? "—"
    : Number(value).toLocaleString("zh-CN", { maximumFractionDigits: digits });
  const price = (value) => value === null || value === undefined || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  const percent = (value, digits = 3) => value === null || value === undefined || Number.isNaN(Number(value)) ? "—" : `${(Number(value) * 100).toFixed(digits)}%`;
  const yesNo = (value) => value ? "是" : "否";
  const methodLabels = { rolling_tr_sum: "滚动 TR 总和" };
  const baselinePolicyLabels = { all_window: "全部", exclude_marked: "排除标记" };
  const exitLabels = { rebound_threshold: "回撤阈值", downside_speed_below_threshold: "速度限制", segment_end: "区间结束" };

  function applyTheme(theme, persist = true) {
    const resolved = theme === "dark" ? "dark" : "light";
    root.dataset.theme = resolved;
    darkThemeButton.setAttribute("aria-pressed", String(resolved === "dark"));
    lightThemeButton.setAttribute("aria-pressed", String(resolved === "light"));
    if (persist) {
      try { localStorage.setItem("short-drop-value-theme", resolved); } catch (error) {}
    }
    if (state.selectedDetail) requestAnimationFrame(drawChart);
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

  function parameterText(row) {
    const speed = row.speed_window_bars === null || row.speed_window_bars === undefined ? "" : ` · S${number(row.speed_window_bars)}`;
    return `E${number(row.e)} · BH${number(row.bh)} · TRW${number(row.trw)} · K${number(row.k)} · W${number(row.w)} · M${number(row.m)}${speed}`;
  }

  function shortCombo(row) {
    const suffix = String(row.combo_id || "").slice(-10);
    return `${methodLabels[row.method] || row.method} · ${parameterText(row)} · ${suffix}`;
  }

  function updateAppState(extra = {}) {
    window.V41_TRADE_APP = {
      catalogLoaded: Boolean(state.catalog),
      catalogTradeCount: state.catalog?.trades?.length || 0,
      comboCount: state.catalog?.combos?.length || 0,
      filteredCount: state.filtered.length,
      loadedChunkKeys: [...state.loadedChunkKeys],
      selectedTradeId: state.selectedDetail?.id || "",
      selectedComboId: state.selectedDetail?.combo_id || "",
      selectedActualEntryReal: Boolean(state.selectedDetail?.actual_entry_real),
      chartMode: state.chartMode,
      ...extra,
    };
  }

  function addOptions(select, values, label = (value) => value) {
    const existing = select.options[0]?.outerHTML || "";
    select.innerHTML = existing + values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(label(value))}</option>`).join("");
  }

  function uniqueNumeric(field) {
    return [...new Set(state.catalog.trades.map((row) => row[field]))].sort((left, right) => Number(left) - Number(right));
  }

  function populateFilters() {
    const combos = [...state.catalog.combos].sort((left, right) => String(left.method).localeCompare(String(right.method)) || Number(left.e) - Number(right.e) || String(left.combo_id).localeCompare(String(right.combo_id)));
    addOptions($("filter-combo"), combos.map((row) => row.combo_id), (value) => shortCombo(state.comboMap.get(value)));
    ["e", "bh", "trw", "k", "w", "m"].forEach((field) => addOptions($("filter-" + field), uniqueNumeric(field), number));
    addOptions($("filter-speed"), uniqueNumeric("speed_window_bars"), (value) => `${number(value)} bar / ${number(Number(value) / 4)} 分钟`);
    addOptions($("filter-exit"), [...new Set(state.catalog.trades.map((row) => row.exit_reason))].sort(), (value) => exitLabels[value] || value);
  }

  function currentFilter(field) {
    return $("filter-" + field).value;
  }

  function matchesFilters(row) {
    if (currentFilter("combo") && row.combo_id !== currentFilter("combo")) return false;
    if (currentFilter("method") && row.method !== currentFilter("method")) return false;
    if (currentFilter("baseline-policy") && row.baseline_sampling_policy !== currentFilter("baseline-policy")) return false;
    for (const field of ["e", "bh", "trw", "k", "w", "m"]) {
      if (currentFilter(field) && String(row[field]) !== currentFilter(field)) return false;
    }
    if (currentFilter("speed") && String(row.speed_window_bars) !== currentFilter("speed")) return false;
    if (currentFilter("wait") === "waited" && !row.waited) return false;
    if (currentFilter("wait") === "immediate" && row.waited) return false;
    if (currentFilter("exit") && row.exit_reason !== currentFilter("exit")) return false;
    if (currentFilter("gap") === "yes" && !row.crosses_gap) return false;
    if (currentFilter("gap") === "no" && row.crosses_gap) return false;
    if (currentFilter("synthetic") === "yes" && !row.synthetic_signal) return false;
    if (currentFilter("synthetic") === "no" && row.synthetic_signal) return false;
    const query = $("filter-search").value.trim().toLowerCase();
    if (query) {
      const haystack = `${row.combo_id} ${row.signal_time} ${row.entry_time} ${row.exit_time} ${parameterText(row)}`.toLowerCase();
      if (!haystack.includes(query)) return false;
    }
    return true;
  }

  function applyFilters({ preservePage = false, selectFirst = false } = {}) {
    state.filtered = state.catalog.trades.filter(matchesFilters).sort((left, right) => String(left.entry_time).localeCompare(String(right.entry_time)) || String(left.combo_id).localeCompare(String(right.combo_id)));
    if (!preservePage) state.page = 1;
    const pageCount = Math.max(1, Math.ceil(state.filtered.length / PAGE_SIZE));
    state.page = Math.min(state.page, pageCount);
    const activeCount = filterIds.filter((field) => currentFilter(field)).length + Number(Boolean($("filter-search").value.trim()));
    $("filter-summary").textContent = `${activeCount ? `${activeCount} 项条件 · ` : ""}${number(state.filtered.length)} / ${number(state.catalog.trade_count)} 笔`;
    $("results-status").textContent = `筛选后 ${number(state.filtered.length)} / ${number(state.catalog.trade_count)} 笔；每页最多 ${PAGE_SIZE} 行。`;
    renderTable();
    updateAppState();
    if (selectFirst && state.filtered.length) selectTrade(state.filtered[0]);
  }

  function renderTable() {
    const pageCount = Math.max(1, Math.ceil(state.filtered.length / PAGE_SIZE));
    const start = (state.page - 1) * PAGE_SIZE;
    const rows = state.filtered.slice(start, start + PAGE_SIZE);
    $("page-status").textContent = `${state.page} / ${pageCount}`;
    $("page-previous").disabled = state.page <= 1;
    $("page-next").disabled = state.page >= pageCount;
    if (!rows.length) {
      $("trade-table").innerHTML = `<div class="empty-state"><h3>当前筛选形成真实空集</h3><p>可调整 combo、参数、速度窗口、等待成交、退出原因、跨缺口或 synthetic 信号条件。</p></div>`;
      return;
    }
    $("trade-table").innerHTML = `<table>
      <thead><tr><th>查看</th><th>#</th><th>combo／参数</th><th>信号时间</th><th>实际开仓</th><th>等待</th><th>退出时间</th><th>退出原因</th><th>收益</th><th>跨缺口</th><th>synthetic 信号</th><th>真实开仓柱</th></tr></thead>
      <tbody>${rows.map((row, index) => `<tr data-trade-id="${escapeHtml(row.id)}" aria-selected="${state.selectedDetail?.id === row.id}">
        <td><button class="inspect-button" type="button" data-inspect="${escapeHtml(row.id)}">证据</button></td>
        <td>${number(start + index + 1)}</td>
        <td class="text-cell combo-cell" title="${escapeHtml(row.combo_id)}">${escapeHtml(methodLabels[row.method] || row.method)}<br><span class="value-muted">${escapeHtml(parameterText(row))}</span></td>
        <td>${escapeHtml(row.signal_time)}</td><td>${escapeHtml(row.entry_time)}<br><span class="value-muted">${price(row.entry_price)}</span></td>
        <td class="${row.waited ? "value-bad" : "value-muted"}">${row.waited ? `${number(row.wait_bars)} 根` : "0"}</td>
        <td>${escapeHtml(row.exit_time)}</td><td>${escapeHtml(exitLabels[row.exit_reason] || row.exit_reason)}</td>
        <td class="${Number(row.return) >= 0 ? "value-good" : "value-bad"}">${percent(row.return)}</td>
        <td>${yesNo(row.crosses_gap)}</td><td>${row.synthetic_signal ? `${number(row.synthetic_signal_bar_count)} 根` : "否"}</td>
        <td class="${row.actual_entry_real ? "value-good" : "value-bad"}">${row.actual_entry_real ? "通过" : "异常"}</td>
      </tr>`).join("")}</tbody>
    </table>`;
    document.querySelectorAll("[data-inspect]").forEach((button) => button.addEventListener("click", () => {
      const record = state.catalog.trades.find((row) => row.id === button.dataset.inspect);
      if (record) selectTrade(record);
    }));
  }

  async function loadChunk(comboKey) {
    if (window.V41_TRADE_CHUNKS?.[comboKey]) return window.V41_TRADE_CHUNKS[comboKey];
    const combo = state.catalog.combos.find((row) => row.key === comboKey);
    if (!combo) throw new Error(`交易目录缺少 combo 分块 ${comboKey}`);
    $("chunk-status").textContent = "正在载入单个 combo 分块…";
    await loadScript(combo.chunk);
    const chunk = window.V41_TRADE_CHUNKS?.[comboKey];
    if (!chunk) throw new Error(`combo 分块没有发布资料：${comboKey}`);
    state.loadedChunkKeys.add(comboKey);
    return chunk;
  }

  async function selectTrade(record, { scroll = true } = {}) {
    try {
      const chunk = await loadChunk(record.combo_key);
      const detail = chunk.trades.find((row) => row.id === record.id);
      if (!detail) throw new Error(`combo 分块缺少交易 ${record.id}`);
      state.selectedCatalog = record;
      state.selectedDetail = detail;
      state.selectedChunk = chunk;
      state.chartMode = new URLSearchParams(location.search).get("reason") === "exit" ? "exit" : "entry";
      $("detail-empty").hidden = true;
      $("detail-content").hidden = false;
      $("chunk-status").textContent = `已载入 1 / ${number(state.catalog.combo_count)} 个 combo 分块`;
      renderDetail();
      renderTable();
      const url = new URL(location.href);
      url.searchParams.set("combo_id", record.combo_id);
      url.searchParams.set("trade", String(detail.entry_index));
      url.searchParams.set("reason", state.chartMode);
      history.replaceState(null, "", url);
      if (scroll) $("trade-detail").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
      updateAppState();
    } catch (error) {
      $("chunk-status").textContent = `载入失败：${error.message}`;
      updateAppState({ error: error.message });
    }
  }

  function proofRows(detail) {
    const actual = detail.actual_entry_evidence;
    const initial = detail.initial_entry_evidence;
    const invalidReasons = [];
    if (initial.s) invalidReasons.push("synthetic");
    if (Number(initial.v) <= 0) invalidReasons.push("volume=0");
    if (Number(initial.n) <= 0) invalidReasons.push("trade count=0");
    return [
      ["实际时间", actual.t],
      ["OHLC", `${price(actual.o)} / ${price(actual.h)} / ${price(actual.l)} / ${price(actual.c)}`],
      ["volume／成交数", `${number(actual.v)} / ${number(actual.n)}`],
      ["synthetic", actual.s ? "是" : "否"],
      ["成交来源", detail.entry_fill_source],
      ["等待", detail.waited ? `${number(detail.entry_wait_bar_count)} 根；初始候选 ${initial.t}` : "初始候选即真实成交柱"],
      ["初始候选状态", detail.waited ? invalidReasons.join(" · ") || "未满足真实成交柱合同" : "通过"],
      ["开仓价格", `${price(detail.entry_price_before_slippage)} → ${price(detail.entry_price)}（slippage ${price(detail.entry_slippage)}）`],
    ];
  }

  function factList(target, rows) {
    $(target).innerHTML = rows.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
  }

  function renderTimeline(detail) {
    const items = [
      ["信号完成", detail.signal_time, `H ${detail.h_time} · drop ${price(detail.entry_drop_value)}`],
      ["初始候选", detail.initial_entry_time, detail.waited ? "候选柱无真实成交，继续等待" : "候选柱满足真实成交合同"],
      ["实际开仓", detail.entry_time, `${price(detail.entry_price)} · ${detail.entry_fill_source}`],
      ["退出", detail.exit_time, `${exitLabels[detail.exit_reason] || detail.exit_reason} · ${percent(detail.return)}`],
    ];
    $("trade-timeline").innerHTML = items.map(([label, time, copy]) => `<li><span class="timeline-label">${escapeHtml(label)}</span><span class="timeline-time">${escapeHtml(time)}</span><span class="timeline-copy">${escapeHtml(copy)}</span></li>`).join("");
  }

  function renderDetail() {
    const detail = state.selectedDetail;
    const combo = state.selectedChunk.combo;
    $("detail-position").textContent = `${methodLabels[detail.method] || detail.method} · combo 内第 ${number(detail.sequence)} 笔`;
    $("detail-title").textContent = `${detail.entry_time} 开仓 · ${percent(detail.return)} 收益`;
    $("entry-proof-state").textContent = detail.actual_entry_real ? "真实成交柱 · 通过" : "真实成交柱 · 异常";
    $("entry-proof-state").className = `state-label ${detail.actual_entry_real ? "state-label-good" : "state-label-bad"}`;
    factList("entry-proof-list", proofRows(detail));
    renderTimeline(detail);
    factList("signal-facts", [
      ["方法", methodLabels[detail.method] || detail.method],
      ["基准采样", baselinePolicyLabels[detail.baseline_sampling_policy] || detail.baseline_sampling_policy],
      ["参数", parameterText(detail)],
      ["signal / H", `${detail.signal_time} / ${detail.h_time}`],
      ["开仓基准", price(detail.entry_baseline_value)],
      ["信号跌幅", price(detail.entry_drop_value)],
      ["单根贡献", percent(detail.signal_single_bar_drop_share)],
      ["synthetic 信号 bar", `${number(detail.signal_synthetic_empty_bar_count)} 根`],
    ]);
    factList("exit-facts", [
      ["退出原因", exitLabels[detail.exit_reason] || detail.exit_reason],
      ["退出价格", price(detail.exit_price)],
      ["交易收益", percent(detail.return)],
      ["active low", `${price(detail.active_low)} · index ${number(detail.active_low_index)}`],
      ["回撤阈值", price(detail.rebound_threshold)],
      ["冻结 W 净跌幅", price(detail.rebound_net_drop)],
      ["S 速度窗口", detail.speed_window_bars == null ? "—" : `${number(detail.speed_window_bars)} bar / ${number(Number(detail.speed_window_bars) / 4)} 分钟`],
      ["速度参考", detail.speed_reference_time ? `${detail.speed_reference_time} · low ${price(detail.speed_reference_low)}` : "—"],
      ["当前低点／延伸", detail.speed_current_low == null ? "—" : `${price(detail.speed_current_low)} / ${price(detail.speed_extension)}`],
      ["速度检查／成交", detail.speed_check_price == null ? "—" : `${price(detail.speed_check_price)} · ${detail.speed_check_price_basis} / ${detail.exit_price_basis}`],
      ["跨真实缺口", yesNo(Boolean(detail.position_crosses_real_gap))],
      ["连续段", `${number(detail.entry_continuous_segment_id)} → ${number(detail.exit_continuous_segment_id)}`],
    ]);
    factList("baseline-facts", [
      ["filter_id", detail.baseline_filter_id],
      ["历史范围", `${number(detail.baseline_history_start_index)} → ${number(detail.baseline_history_end_index)}`],
      ["有效原子", `${number(detail.baseline_eligible_atom_count)} / 物理跨度 ${number(detail.baseline_physical_span_bars)}`],
      ["排除原子", number(detail.baseline_excluded_atom_count)],
      ["pending 原子", number(detail.baseline_pending_atom_count)],
      ["正式排除原子", number(detail.baseline_confirmed_excluded_atom_count)],
      ["combo", combo.combo_id],
    ]);
    $("chart-entry").setAttribute("aria-pressed", "true");
    $("chart-exit").setAttribute("aria-pressed", "false");
    drawChart();
  }

  function chartColors() {
    const style = getComputedStyle(root);
    const value = (name) => style.getPropertyValue(name).trim();
    const first = (...names) => names.map(value).find(Boolean) || "#ffffff";
    return {
      ink: first("--text", "--ink"), body: first("--body", "--text"), muted: first("--muted"),
      grid: value("--chart-grid") || (root.dataset.theme === "light" ? "rgba(15,35,72,.10)" : "rgba(255,255,255,.075)"),
      background: first("--chart-bg", "--panel-soft", "--surface"),
      up: first("--up", "--green"), down: first("--down", "--red"), signal: first("--signal", "--orange"),
      entry: first("--entry", "--green"), exit: first("--exit", "--purple"), line: first("--line-strong", "--line"),
    };
  }

  function barsForRange(range) {
    const [start, end] = range.map(Number);
    return state.selectedChunk.bars.filter((bar) => Number(bar.i) >= start && Number(bar.i) <= end);
  }

  function drawVerticalMarker(context, x, top, bottom, color, label, align = "left") {
    context.save();
    context.strokeStyle = color;
    context.lineWidth = 1.5;
    context.setLineDash([5, 4]);
    context.beginPath();
    context.moveTo(x, top);
    context.lineTo(x, bottom);
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = color;
    context.font = "600 11px system-ui";
    context.textAlign = align;
    context.fillText(label, align === "left" ? x + 4 : x - 4, top + 12);
    context.restore();
  }

  function drawChart() {
    if (!state.selectedDetail || !state.selectedChunk) return;
    const canvas = $("evidence-chart");
    const box = canvas.getBoundingClientRect();
    const width = Math.max(320, Math.round(box.width));
    const height = Math.max(300, Math.round(box.height));
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    const colors = chartColors();
    context.fillStyle = colors.background;
    context.fillRect(0, 0, width, height);
    const detail = state.selectedDetail;
    const range = state.chartMode === "entry" ? detail.entry_bar_range : detail.exit_bar_range;
    const bars = barsForRange(range);
    if (!bars.length) return;
    const padding = { left: 58, right: 18, top: 25, bottom: 48 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const lows = bars.map((bar) => Number(bar.l));
    const highs = bars.map((bar) => Number(bar.h));
    let minimum = Math.min(...lows);
    let maximum = Math.max(...highs);
    const rawSpan = Math.max(0.05, maximum - minimum);
    minimum -= rawSpan * 0.08;
    maximum += rawSpan * 0.08;
    const scaleY = (value) => padding.top + (maximum - Number(value)) / (maximum - minimum) * plotHeight;
    const step = plotWidth / Math.max(1, bars.length);
    const scaleX = (position) => padding.left + step * (position + 0.5);
    context.font = "11px system-ui";
    context.textBaseline = "middle";
    for (let tick = 0; tick <= 5; tick += 1) {
      const y = padding.top + plotHeight * tick / 5;
      const value = maximum - (maximum - minimum) * tick / 5;
      context.strokeStyle = colors.grid;
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(padding.left, y);
      context.lineTo(width - padding.right, y);
      context.stroke();
      context.fillStyle = colors.muted;
      context.textAlign = "right";
      context.fillText(price(value), padding.left - 7, y);
    }
    const candleWidth = Math.max(2, Math.min(11, step * 0.58));
    bars.forEach((bar, position) => {
      const x = scaleX(position);
      const open = Number(bar.o), close = Number(bar.c), high = Number(bar.h), low = Number(bar.l);
      const color = close >= open ? colors.up : colors.down;
      context.strokeStyle = color;
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(x, scaleY(high));
      context.lineTo(x, scaleY(low));
      context.stroke();
      const y = Math.min(scaleY(open), scaleY(close));
      const bodyHeight = Math.max(1.5, Math.abs(scaleY(open) - scaleY(close)));
      context.fillStyle = bar.s ? colors.line : color;
      context.fillRect(x - candleWidth / 2, y, candleWidth, bodyHeight);
    });
    const positionOf = (index) => bars.findIndex((bar) => Number(bar.i) === Number(index));
    if (state.chartMode === "entry") {
      const signalPosition = positionOf(detail.signal_index);
      const initialPosition = positionOf(detail.initial_entry_index);
      const entryPosition = positionOf(detail.entry_index);
      if (signalPosition >= 0) drawVerticalMarker(context, scaleX(signalPosition), padding.top, padding.top + plotHeight, colors.signal, "信号");
      if (initialPosition >= 0 && Number(detail.initial_entry_index) !== Number(detail.entry_index)) drawVerticalMarker(context, scaleX(initialPosition), padding.top, padding.top + plotHeight, colors.line, "初始", "right");
      if (entryPosition >= 0) drawVerticalMarker(context, scaleX(entryPosition), padding.top, padding.top + plotHeight, colors.entry, "实际开仓", "right");
      $("chart-caption").textContent = `${detail.signal_time} → ${detail.entry_time} · 等待 ${number(detail.entry_wait_bar_count)} 根`;
      $("chart-accessible-summary").textContent = `开仓证据：信号时间 ${detail.signal_time}，实际开仓时间 ${detail.entry_time}，实际开仓柱 synthetic 为否，volume ${number(detail.actual_entry_evidence.v)}，trade count ${number(detail.actual_entry_evidence.n)}。`;
    } else {
      const exitPosition = positionOf(detail.exit_index);
      if (exitPosition >= 0) drawVerticalMarker(context, scaleX(exitPosition), padding.top, padding.top + plotHeight, colors.exit, "退出", "right");
      $("chart-caption").textContent = `${detail.exit_time} · ${exitLabels[detail.exit_reason] || detail.exit_reason} · ${percent(detail.return)}`;
      $("chart-accessible-summary").textContent = `平仓证据：退出时间 ${detail.exit_time}，退出原因 ${exitLabels[detail.exit_reason] || detail.exit_reason}，退出价格 ${price(detail.exit_price)}。`;
    }
    context.fillStyle = colors.muted;
    context.textAlign = "left";
    context.textBaseline = "top";
    context.fillText(String(bars[0].t).slice(5, 16), padding.left, height - padding.bottom + 12);
    context.textAlign = "right";
    context.fillText(String(bars[bars.length - 1].t).slice(5, 16), width - padding.right, height - padding.bottom + 12);
  }

  function setChartMode(mode) {
    state.chartMode = mode === "exit" ? "exit" : "entry";
    $("chart-entry").setAttribute("aria-pressed", String(state.chartMode === "entry"));
    $("chart-exit").setAttribute("aria-pressed", String(state.chartMode === "exit"));
    $("chart-entry").classList.toggle("active", state.chartMode === "entry");
    $("chart-exit").classList.toggle("active", state.chartMode === "exit");
    drawChart();
    if (state.selectedDetail) {
      const url = new URL(location.href);
      url.searchParams.set("reason", state.chartMode);
      history.replaceState(null, "", url);
    }
    updateAppState();
  }

  function setDrawerOpen(open) {
    const drawer = $("controlsDrawer");
    const collapse = $("controlsCollapse");
    const opener = $("controlsOpen");
    drawer.dataset.open = String(open);
    drawer.setAttribute("aria-hidden", String(!open));
    collapse.setAttribute("aria-expanded", String(open));
    opener.setAttribute("aria-expanded", String(open));
    opener.hidden = open;
  }

  function resetFilters() {
    filterIds.forEach((field) => { $("filter-" + field).value = ""; });
    $("filter-search").value = "";
    applyFilters({ selectFirst: true });
  }

  function applyQuerySelection() {
    const params = new URLSearchParams(location.search);
    const requestedCombo = params.get("combo_id") || "";
    const requestedTrade = params.get("trade") || "";
    if (requestedCombo && state.comboMap.has(requestedCombo)) $("filter-combo").value = requestedCombo;
    applyFilters();
    let selected = state.filtered[0];
    if (requestedTrade) {
      selected = state.filtered.find((row) => String(row.id).endsWith(`-${requestedTrade}`)) || selected;
    }
    if (selected) selectTrade(selected, { scroll: false });
  }

  function fail(error) {
    $("startup-resource-status").textContent = `目录载入失败：${error.message}`;
    $("results-status").textContent = "交易目录不可用";
    $("trade-table").innerHTML = `<div class="empty-state"><h3>V4.4 交易目录未载入</h3><p>${escapeHtml(error.message)}</p></div>`;
    updateAppState({ error: error.message });
  }

  function initialize(catalog) {
    if (!catalog || !Array.isArray(catalog.trades) || !Array.isArray(catalog.combos) || catalog.trade_count !== catalog.trades.length || catalog.combo_count !== catalog.combos.length) throw new Error("交易目录闭合计数与目录内容不一致");
    if (catalog.trades.some((row) => !row.combo_id.startsWith("v4_4_") || !row.actual_entry_real)) throw new Error("交易目录包含非 V4.4 身份或未通过的开仓柱");
    state.catalog = catalog;
    state.comboMap = new Map(catalog.combos.map((row) => [row.combo_id, row]));
    populateFilters();
    $("startup-resource-status").textContent = `目录已载入 · combo 详情 0 / ${number(catalog.combo_count)}`;
    filterIds.forEach((field) => $("filter-" + field).addEventListener("change", () => applyFilters({ selectFirst: true })));
    $("filter-search").addEventListener("input", () => applyFilters({ selectFirst: false }));
    $("reset-filters").addEventListener("click", resetFilters);
    $("page-previous").addEventListener("click", () => { state.page -= 1; applyFilters({ preservePage: true }); });
    $("page-next").addEventListener("click", () => { state.page += 1; applyFilters({ preservePage: true }); });
    $("chart-entry").addEventListener("click", () => setChartMode("entry"));
    $("chart-exit").addEventListener("click", () => setChartMode("exit"));
    applyQuerySelection();
    updateAppState();
  }

  applyTheme(root.dataset.theme, false);
  darkThemeButton.addEventListener("click", () => applyTheme("dark"));
  lightThemeButton.addEventListener("click", () => applyTheme("light"));
  $("controlsCollapse").addEventListener("click", () => setDrawerOpen(false));
  $("controlsOpen").addEventListener("click", () => setDrawerOpen(true));
  setDrawerOpen(true);
  if ("ResizeObserver" in window) new ResizeObserver(() => { if (state.selectedDetail) drawChart(); }).observe($("evidence-chart"));
  else window.addEventListener("resize", () => { if (state.selectedDetail) drawChart(); });
  loadScript(`${window.V41_TRADE_ASSET_PREFIX || "../assets"}/trade_catalog.js`)
    .then(() => initialize(window.V41_TRADE_CATALOG))
    .catch(fail);
})();
