import fs from "node:fs";
import fsPromises from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import os from "node:os";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);

function loadPlaywright() {
  try { return require("playwright"); } catch {}
  const runtimeRoot = path.join(
    os.homedir(), ".cache", "codex-runtimes", "codex-primary-runtime",
    "dependencies", "node", "node_modules",
  );
  const direct = path.join(runtimeRoot, "playwright");
  if (fs.existsSync(direct)) return require(direct);
  throw new Error("Playwright is unavailable");
}

const analysisRoot = path.resolve(process.argv[2] || "");
const tradeOnly = process.argv.includes("--trade-only");
if (!analysisRoot || !fs.existsSync(analysisRoot)) {
  throw new Error("usage: node qa_v4_4_scenario_3_stage.mjs <analysis-root>");
}
const mainPath = path.join(analysisRoot, "index.html");
const tradePath = path.join(analysisRoot, "trade_review", "index.html");
const scenarioPath = path.join(analysisRoot, "scenario_requirements", "index.html");
const manifestPath = path.join(analysisRoot, "analysis_manifest.json");
const reviewManifestPath = path.join(analysisRoot, "trade_review", "trade_review_manifest.json");
for (const target of [mainPath, tradePath, scenarioPath, manifestPath, reviewManifestPath]) {
  if (!fs.existsSync(target)) throw new Error(`required stage resource is missing: ${target}`);
}

const sha256 = (target) => crypto.createHash("sha256").update(fs.readFileSync(target)).digest("hex");
const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const reviewManifest = JSON.parse(fs.readFileSync(reviewManifestPath, "utf8"));
const qaRoot = path.join(analysisRoot, "qa", "scenario_3_stage");
await fsPromises.mkdir(qaRoot, { recursive: true });
const runtimeErrors = [];
const externalRequests = [];
const chunkRequests = [];

function bindDiagnostics(page) {
  page.on("pageerror", (error) => runtimeErrors.push(String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") runtimeErrors.push(message.text());
  });
  page.on("request", (request) => {
    const url = request.url();
    if (/\/(?:chunks|trade_chunks|v3_native_trades_js)\/.*\.js(?:\?|$)/.test(url)) chunkRequests.push(url);
    if (!url.startsWith("file:") && !url.startsWith("data:") && !url.startsWith("blob:")) {
      externalRequests.push(url);
    }
  });
}

async function pageAudit(page, selectors) {
  return page.evaluate((requiredSelectors) => {
    const missing = requiredSelectors.filter((selector) => !document.querySelector(selector));
    const replacements = (document.body.innerText.match(/�/g) || []).length;
    const width = window.innerWidth;
    return {
      width,
      documentWidth: document.documentElement.scrollWidth,
      bodyWidth: document.body.scrollWidth,
      pageOverflow: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) > width + 1,
      replacements,
      missing,
      title: document.title,
    };
  }, selectors);
}

function assertAudit(audit, label) {
  if (audit.pageOverflow || audit.replacements || audit.missing.length) {
    throw new Error(`${label} visual audit failed: ${JSON.stringify(audit)}`);
  }
}

const { chromium } = loadPlaywright();
const browser = await chromium.launch({ headless: true, args: ["--allow-file-access-from-files"] });
try {
  const main = await browser.newPage({ viewport: { width: 1440, height: 950 } });
  bindDiagnostics(main);
  await main.goto(pathToFileURL(mainPath).href, { waitUntil: "load", timeout: 180000 });
  await main.waitForFunction(() => window.V4_ANALYSIS_DATA && document.querySelector("#table"), { timeout: 180000 });
  const mainData = await main.evaluate(() => ({
    coordinateCount: window.V4_ANALYSIS_DATA.coordinateCount,
    tradeCount: window.V4_ANALYSIS_DATA.tradeCount,
    scenario3QualifiedCount: window.V4_ANALYSIS_DATA.scenario3QualifiedCount,
    speedWindows: window.V4_ANALYSIS_DATA.speedWindows,
    baselineSamplingPolicies: window.V4_ANALYSIS_DATA.baselineSamplingPolicies,
    baselinePolicyControls: [...document.querySelectorAll("[data-baseline-policy]")].map((node) => node.dataset.baselinePolicy),
    speedControls: [...document.querySelectorAll("[data-speed]")].map((node) => node.dataset.speed),
    speedControlLabels: [...document.querySelectorAll("[data-speed]")].map((node) => node.textContent.trim()),
    metricControls: [...document.querySelectorAll("[data-metric]")].map((node) => node.dataset.metric),
    returnViewControls: [...document.querySelectorAll("[data-return-view]")].map((node) => node.dataset.returnView),
    activeReturnViewControls: [...document.querySelectorAll("[data-return-view].active")].map((node) => node.dataset.returnView),
    countControls: [...document.querySelectorAll("[data-count]")].map((node) => node.dataset.count),
    highReturnViews: window.V4_ANALYSIS_DATA.highReturnViews,
    highReturnViewControls: [...document.querySelectorAll("[data-high-return-view]")].map((node) => node.dataset.highReturnView),
    scenario3DataCount: window.V4_ANALYSIS_DATA.rows.filter((row) => Boolean(row.scenario_3_qualified)).length,
    templateSha: window.V4_ANALYSIS_DATA.templateProvenance.sha256,
    tradeLinks: [...document.querySelectorAll("a.rank-link")].map((node) => node.href),
    scenarioRequirementsRoute: window.V4_ANALYSIS_DATA.scenarioRequirementsRoute,
    costModel: window.V4_ANALYSIS_DATA.costModel,
  }));
  if (
    mainData.coordinateCount !== manifest.coordinate_count
    || mainData.tradeCount !== manifest.trade_count
    || mainData.scenario3QualifiedCount !== manifest.scenario_3_qualified_coordinate_count
    || mainData.scenario3DataCount !== manifest.scenario_3_qualified_coordinate_count
  ) {
    throw new Error(`main closure mismatch: ${JSON.stringify(mainData)}`);
  }
  const expandedControlHeight = await main.locator("#controls").evaluate((node) => node.getBoundingClientRect().height);
  await main.locator("#control-toggle").click();
  const collapsedControls = await main.evaluate(() => {
    const panel = document.querySelector("#controls");
    const heading = document.querySelector("#control-title").getBoundingClientRect();
    const toggle = document.querySelector("#control-toggle");
    const toggleRect = toggle.getBoundingClientRect();
    return {
      collapsed: panel.classList.contains("is-collapsed"),
      expanded: toggle.getAttribute("aria-expanded"),
      visibleText: toggle.innerText.trim(),
      accessibleLabel: toggle.getAttribute("aria-label"),
      iconPath: toggle.querySelector("path")?.getAttribute("d"),
      iconTransform: getComputedStyle(toggle.querySelector(".control-toggle-icon")).transform,
      gridDisplay: getComputedStyle(document.querySelector("#control-grid")).display,
      statusDisplay: getComputedStyle(document.querySelector("#status")).display,
      height: panel.getBoundingClientRect().height,
      oneRow: Math.abs(heading.top - toggleRect.top) < 10,
    };
  });
  if (
    !collapsedControls.collapsed
    || collapsedControls.expanded !== "false"
    || collapsedControls.visibleText !== ""
    || collapsedControls.accessibleLabel !== "Expand filters and sorting"
    || collapsedControls.iconPath !== "m6 9 6 6 6-6"
    || collapsedControls.iconTransform !== "none"
    || collapsedControls.gridDisplay !== "none"
    || collapsedControls.statusDisplay !== "none"
    || collapsedControls.height >= expandedControlHeight
    || !collapsedControls.oneRow
  ) {
    throw new Error(`main control collapse failed: ${JSON.stringify({ expandedControlHeight, collapsedControls })}`);
  }
  await main.locator("#control-toggle").click();
  const expandedControls = await main.evaluate(() => ({
    collapsed: document.querySelector("#controls").classList.contains("is-collapsed"),
    expanded: document.querySelector("#control-toggle").getAttribute("aria-expanded"),
    gridDisplay: getComputedStyle(document.querySelector("#control-grid")).display,
    iconTransform: getComputedStyle(document.querySelector(".control-toggle-icon")).transform,
  }));
  if (
    expandedControls.collapsed
    || expandedControls.expanded !== "true"
    || expandedControls.gridDisplay === "none"
    || expandedControls.iconTransform === "none"
  ) {
    throw new Error(`main control expand failed: ${JSON.stringify(expandedControls)}`);
  }
  const expectedSpeedControls = ["all", "lt5", "5_15", "15_30", "30_60", "60_120", "gte120"];
  const expectedSpeedControlLabels = ["全部", "＜5 分钟", "5–＜15 分钟", "15–＜30 分钟", "30–＜60 分钟", "60–＜120 分钟", "≥120 分钟"];
  if (
    JSON.stringify(mainData.speedControls) !== JSON.stringify(expectedSpeedControls)
    || JSON.stringify(mainData.speedControlLabels) !== JSON.stringify(expectedSpeedControlLabels)
  ) {
    throw new Error(`main speed controls do not match stage data: ${JSON.stringify(mainData)}`);
  }
  if (JSON.stringify(mainData.baselinePolicyControls) !== JSON.stringify(mainData.baselineSamplingPolicies)) {
    throw new Error(`baseline policy controls do not match stage data: ${JSON.stringify(mainData)}`);
  }
  const expectedHighReturnViews = [
    ["scenario_1_qualified_total_return", "scenario_1", 0, "total_return", "train_cost_adjusted_return"],
    ["unrestricted_total_return", "all", 0, "total_return", "train_cost_adjusted_return"],
    ["unrestricted_average_return_ge10", "all", 10, "average_trade", "train_cost_adjusted_avg_trade"],
    ["unrestricted_average_return_ge20", "all", 20, "average_trade", "train_cost_adjusted_avg_trade"],
  ];
  const declaredHighReturnViews = mainData.highReturnViews.map((view) => [view.id, view.scenario_filter, view.minimum_trade_count, view.metric, view.metric_key]);
  if (
    JSON.stringify(declaredHighReturnViews) !== JSON.stringify(expectedHighReturnViews)
    || JSON.stringify(mainData.highReturnViewControls) !== JSON.stringify(expectedHighReturnViews.map(([id]) => id))
    || !mainData.metricControls.includes("total_return")
    || !mainData.metricControls.includes("average_trade")
    || JSON.stringify(mainData.metricControls) !== JSON.stringify(["total_return", "average_trade"])
    || JSON.stringify(mainData.returnViewControls) !== JSON.stringify(["cost_adjusted", "gross"])
    || JSON.stringify(mainData.activeReturnViewControls) !== JSON.stringify(["cost_adjusted"])
    || Number(mainData.costModel?.round_trip_slippage_bps) !== 2
    || Number(mainData.costModel?.round_trip_commission_usd) !== 6
    || Number(mainData.costModel?.round_trip_total_cost_bps) !== Number(manifest.cost_model?.round_trip_total_cost_bps)
    || !mainData.countControls.includes("ge10")
    || !mainData.countControls.includes("ge20")
  ) {
    throw new Error(`four-view controls are incomplete: ${JSON.stringify(mainData)}`);
  }

  let stateCount = 0;
  if (!tradeOnly) {
    const methods = await main.locator("#method option").evaluateAll((nodes) => nodes.map((node) => node.value));
    for (const method of methods) {
      await main.selectOption("#method", method);
      for (const baselinePolicy of mainData.baselineSamplingPolicies) {
        await main.locator(`[data-baseline-policy="${baselinePolicy}"]`).click();
      for (const scenario of ["all", "scenario_1", "scenario_2", "scenario_3"]) {
        await main.locator(`[data-scenario="${scenario}"]`).click();
        for (const speed of expectedSpeedControls) {
          await main.locator(`[data-speed="${speed}"]`).click();
          for (const metric of ["total_return", "average_trade"]) {
            await main.locator(`[data-metric="${metric}"]`).click();
            for (const count of mainData.countControls) {
              await main.locator(`[data-count="${count}"]`).click();
              const current = await main.evaluate(() => ({
                empty: Boolean(document.querySelector("#table .empty")),
                rows: document.querySelectorAll("#table tbody tr").length,
                comboIds: [...document.querySelectorAll("#table a.rank-link")].map((node) => node.dataset.comboId),
                status: document.querySelector("#status")?.innerText || "",
              }));
              if (!current.empty && current.rows === 0) throw new Error(`invalid main state: ${JSON.stringify({ method, scenario, speed, metric, count, current })}`);
              if (scenario === "scenario_3") {
                const expectedComboIds = await main.evaluate(({ currentMethod, currentBaselinePolicy, currentSpeed, currentMetric, currentCount }) => {
                  const metricKeys = {
                    total_return: "train_cost_adjusted_return",
                    average_trade: "train_cost_adjusted_avg_trade",
                  };
                  const speedRanges = {
                    lt5: [null, 5],
                    "5_15": [5, 15],
                    "15_30": [15, 30],
                    "30_60": [30, 60],
                    "60_120": [60, 120],
                    gte120: [120, null],
                  };
                  const minimumTrades = Number(currentCount.replace("ge", "")) || 0;
                  const metricKey = metricKeys[currentMetric];
                  const direction = -1;
                  return window.V4_ANALYSIS_DATA.rows
                    .filter((row) => (
                      row.method === currentMethod
                      && row.baseline_sampling_policy === currentBaselinePolicy
                      && Boolean(row.scenario_3_qualified)
                      && (
                        currentSpeed === "all"
                        || (
                          speedRanges[currentSpeed]
                          && (speedRanges[currentSpeed][0] == null || Number(row.speed_window_bars) / 4 >= speedRanges[currentSpeed][0])
                          && (speedRanges[currentSpeed][1] == null || Number(row.speed_window_bars) / 4 < speedRanges[currentSpeed][1])
                        )
                      )
                      && Number(row.train_trade_count) >= minimumTrades
                      && Number.isFinite(Number(row[metricKey]))
                    ))
                    .sort((a, b) => direction * (Number(a[metricKey]) - Number(b[metricKey])) || String(a.combo_id).localeCompare(String(b.combo_id)))
                    .map((row) => row.combo_id);
                }, { currentMethod: method, currentBaselinePolicy: baselinePolicy, currentSpeed: speed, currentMetric: metric, currentCount: count });
                if (
                  current.empty !== (expectedComboIds.length === 0)
                  || JSON.stringify(current.comboIds) !== JSON.stringify(expectedComboIds)
                ) {
                  throw new Error(`Scenario 3 manifest/data reconciliation failed: ${JSON.stringify({ method, baselinePolicy, speed, metric, count, current, expectedComboIds })}`);
                }
              }
              stateCount += 1;
            }
          }
        }
      }
      }
    }
  }
  const highReturnViewStates = {};
  await main.locator('[data-scenario="all"]').click();
  await main.locator('[data-speed="all"]').click();
  for (const [viewId, viewScenario, minimumTradeCount, viewMetric] of expectedHighReturnViews) {
    await main.locator(`[data-high-return-view="${viewId}"]`).click();
    const state = await main.evaluate(() => ({
      activeViewIds: [...document.querySelectorAll("[data-high-return-view].active")].map((node) => node.dataset.highReturnView),
      activeScenarioIds: [...document.querySelectorAll("[data-scenario].active")].map((node) => node.dataset.scenario),
      activeMetricIds: [...document.querySelectorAll("[data-metric].active")].map((node) => node.dataset.metric),
      activeCountIds: [...document.querySelectorAll("[data-count].active")].map((node) => node.dataset.count),
      comboIds: [...document.querySelectorAll("#table a.rank-link")].map((node) => node.dataset.comboId),
      status: document.querySelector("#status")?.innerText || "",
    }));
    if (
      JSON.stringify(state.activeViewIds) !== JSON.stringify([viewId])
      || JSON.stringify(state.activeScenarioIds) !== JSON.stringify([viewScenario])
      || JSON.stringify(state.activeMetricIds) !== JSON.stringify([viewMetric])
      || JSON.stringify(state.activeCountIds) !== JSON.stringify([minimumTradeCount ? `ge${minimumTradeCount}` : "all"])
    ) {
      throw new Error(`four-view selection failed: ${JSON.stringify({ viewId, state })}`);
    }
    highReturnViewStates[viewId] = state;
  }
  await main.locator('[data-return-view="cost_adjusted"]').click();
  const costAdjustedDisplayState = await main.evaluate(() => ({
    activeReturnViewIds: [...document.querySelectorAll("[data-return-view].active")].map((node) => node.dataset.returnView),
    comboIds: [...document.querySelectorAll("#table a.rank-link")].map((node) => node.dataset.comboId),
    rankHeader: document.querySelector("#table thead th:first-child")?.innerText || "",
    tableText: document.querySelector("#table")?.innerText || "",
    status: document.querySelector("#status")?.innerText || "",
  }));
  if (
    JSON.stringify(costAdjustedDisplayState.activeReturnViewIds) !== JSON.stringify(["cost_adjusted"])
    || !costAdjustedDisplayState.rankHeader.includes("成本后排名")
    || !costAdjustedDisplayState.tableText.includes("成本后总收益")
    || !costAdjustedDisplayState.status.includes("默认采用手续费／滑点后口径")
  ) {
    throw new Error(`cost-adjusted display toggle failed: ${JSON.stringify(costAdjustedDisplayState)}`);
  }
  await main.locator('[data-return-view="gross"]').click();
  const grossDisplayState = await main.evaluate(() => {
    const currentMetric = document.querySelector("[data-metric].active")?.dataset.metric;
    const currentScenario = document.querySelector("[data-scenario].active")?.dataset.scenario;
    const currentCount = document.querySelector("[data-count].active")?.dataset.count || "all";
    const minimumTrades = Number(currentCount.replace("ge", "")) || 0;
    const metricKey = currentMetric === "average_trade" ? "train_avg_trade" : "train_return";
    const expectedComboIds = window.V4_ANALYSIS_DATA.rows
      .filter((row) => (
        row.method === document.querySelector("#method")?.value
        && row.baseline_sampling_policy === document.querySelector("[data-baseline-policy].active")?.dataset.baselinePolicy
        && (currentScenario === "all" || Boolean(row[`${currentScenario}_qualified`]))
        && Number(row.train_trade_count) >= minimumTrades
        && Number.isFinite(Number(row[metricKey]))
      ))
      .sort((a, b) => Number(b[metricKey]) - Number(a[metricKey]) || String(a.combo_id).localeCompare(String(b.combo_id)))
      .map((row) => row.combo_id);
    return {
      activeReturnViewIds: [...document.querySelectorAll("[data-return-view].active")].map((node) => node.dataset.returnView),
      comboIds: [...document.querySelectorAll("#table a.rank-link")].map((node) => node.dataset.comboId),
      expectedComboIds,
      rankHeader: document.querySelector("#table thead th:first-child")?.innerText || "",
      tableText: document.querySelector("#table")?.innerText || "",
      status: document.querySelector("#status")?.innerText || "",
    };
  });
  if (
    JSON.stringify(grossDisplayState.activeReturnViewIds) !== JSON.stringify(["gross"])
    || JSON.stringify(grossDisplayState.comboIds) !== JSON.stringify(grossDisplayState.expectedComboIds)
    || !grossDisplayState.rankHeader.includes("毛收益排名")
    || !grossDisplayState.tableText.includes("毛总收益")
    || !grossDisplayState.status.includes("排序与显示均采用 无手续费／滑点")
  ) {
    throw new Error(`gross display/ranking toggle failed: ${JSON.stringify(grossDisplayState)}`);
  }
  await main.locator('[data-return-view="cost_adjusted"]').click();
  await main.locator('[data-scenario="all"]').click();
  await main.locator('[data-speed="all"]').click();
  await main.locator('[data-metric="total_return"]').click();
  await main.locator('[data-count="ge10"]').click();
  await main.selectOption("#method", "rolling_tr_sum");
  const firstRoute = await main.locator("a.rank-link").first().getAttribute("href");
  if (!firstRoute || !firstRoute.includes("trade_review/index.html?combo_id=")) throw new Error(`main trade route is invalid: ${firstRoute}`);
  const scenarioRoute = await main.locator("#scenario-requirements-link").getAttribute("href");
  if (!scenarioRoute || !scenarioRoute.includes("scenario_requirements/index.html?scenario=all")) {
    throw new Error(`main scenario-requirements route is invalid: ${scenarioRoute}`);
  }

  await main.locator("#theme").click();
  const mainDesktop = await pageAudit(main, [".page-head", "#cards", "#table", "#contract-table"]);
  assertAudit(mainDesktop, "main desktop");
  const mainDesktopPath = path.join(qaRoot, "main_dark_1440x950.png");
  await main.screenshot({ path: mainDesktopPath, fullPage: true });
  await main.setViewportSize({ width: 390, height: 844 });
  const mainMobile = await pageAudit(main, [".page-head", "#cards", "#table", "#contract-table"]);
  assertAudit(mainMobile, "main mobile");
  const mainMobilePath = path.join(qaRoot, "main_mobile_390x844.png");
  await main.screenshot({ path: mainMobilePath, fullPage: true });

  const scenario = await browser.newPage({ viewport: { width: 1440, height: 950 } });
  bindDiagnostics(scenario);
  await scenario.goto(new URL(scenarioRoute, pathToFileURL(mainPath).href).href, { waitUntil: "load", timeout: 180000 });
  await scenario.waitForFunction(() => (
    window.V4_4_SCENARIO_REQUIREMENTS_READY
    && document.querySelector("#chart")?._fullLayout
  ), { timeout: 180000 });
  const scenarioStates = {};
  for (const scenarioId of ["all", "scenario_1", "scenario_2", "scenario_3"]) {
    await scenario.locator(`[data-scenario="${scenarioId}"]`).click();
    await scenario.waitForFunction((expected) => (
      new URL(location.href).searchParams.get("scenario") === expected
    ), scenarioId);
    scenarioStates[scenarioId] = await scenario.evaluate(() => ({
      rows: document.querySelectorAll("#segment-body tr").length,
      shapes: document.querySelector("#chart")?.layout?.shapes?.length || 0,
      traceTypes: (document.querySelector("#chart")?.data || []).map((trace) => trace.type),
      query: new URL(location.href).searchParams.get("scenario"),
    }));
  }
  if (
    scenarioStates.all.rows !== 3
    || scenarioStates.scenario_1.rows !== 1
    || scenarioStates.scenario_2.rows !== 1
    || scenarioStates.scenario_3.rows !== 3
    || Object.values(scenarioStates).some((state) => !state.traceTypes.includes("candlestick") || state.rows !== state.shapes)
  ) {
    throw new Error(`scenario requirements state mismatch: ${JSON.stringify(scenarioStates)}`);
  }
  const scenarioDesktop = await pageAudit(scenario, [".top", "#scenario-tabs", ".rule-panel", "#chart", "#segment-body"]);
  assertAudit(scenarioDesktop, "scenario desktop");
  const scenarioDesktopPath = path.join(qaRoot, "scenario_requirements_1440x950.png");
  await scenario.screenshot({ path: scenarioDesktopPath, fullPage: true });
  await scenario.setViewportSize({ width: 390, height: 844 });
  const scenarioMobile = await pageAudit(scenario, [".top", "#scenario-tabs", ".rule-panel", "#chart", "#segment-body"]);
  assertAudit(scenarioMobile, "scenario mobile");
  const scenarioMobilePath = path.join(qaRoot, "scenario_requirements_mobile_390x844.png");
  await scenario.screenshot({ path: scenarioMobilePath, fullPage: true });

  const tradeUrl = new URL(firstRoute, pathToFileURL(mainPath).href);
  tradeUrl.searchParams.set("reason", "exit");
  tradeUrl.searchParams.set("research_contract_id", manifest.campaign_id);
  const requestedComboId = tradeUrl.searchParams.get("combo_id");
  const trade = await browser.newPage({ viewport: { width: 1440, height: 950 } });
  bindDiagnostics(trade);
  await trade.goto(tradeUrl.href, { waitUntil: "load", timeout: 180000 });
  await trade.waitForFunction(() => (
    window.Plotly
    && window.PROCESS_PAYLOAD?.features?.datetime?.length
    && window.ALL_RESULTS_TRADE_EXPLAIN_CATALOG?.rows?.length
    && document.querySelector("#chart")?._fullLayout
    && document.querySelector("#tradeSelect")?.options?.length
  ), { timeout: 180000 });
  const tradeState = await trade.evaluate(() => ({
    catalogComboCount: window.ALL_RESULTS_TRADE_EXPLAIN_CATALOG.rows.length,
    sourceBarCount: window.PROCESS_PAYLOAD.features.datetime.length,
    selectedComboId: new URL(location.href).searchParams.get("combo_id"),
    selectedResearchContractId: new URL(location.href).searchParams.get("research_contract_id"),
    selectedTradeCount: document.querySelector("#tradeSelect").options.length,
    chartTraceTypes: document.querySelector("#chart").data.map((trace) => trace.type || "scatter"),
    chartTitle: document.querySelector("#chart").layout?.title?.text || "",
    chartShapeDashValues: Array.from(document.querySelector("#chart").layout?.shapes || [])
      .filter((shape) => Number(shape.line?.width || 0) > 0)
      .map((shape) => shape.line?.dash || "solid"),
    chartAnnotationTexts: Array.from(document.querySelector("#chart").layout?.annotations || [])
      .map((annotation) => String(annotation.text || "")),
    reasonHeading: document.querySelector("#reasonHeading")?.innerText || "",
    exitPanelVisible: !document.querySelector("#exitReasonPanel")?.hidden,
    selectionSummary: document.querySelector("#selectionSummary")?.innerText || "",
  }));
  const closure = reviewManifest.closure;
  if (
    tradeState.catalogComboCount !== manifest.coordinate_count
    || tradeState.sourceBarCount !== closure.source_ohlc_bar_count
    || tradeState.selectedComboId !== requestedComboId
    || tradeState.selectedResearchContractId !== manifest.campaign_id
    || !tradeState.chartTraceTypes.includes("candlestick")
    || tradeState.chartShapeDashValues.some((dash) => dash !== "solid")
    || tradeState.chartAnnotationTexts.some((text) => text.startsWith("理论线=") || text.includes("· 实际成交="))
    || tradeState.chartAnnotationTexts
      .filter((text) => text.startsWith("L="))
      .some((text) => !/^L=-?\d+(?:\.\d+)?$/.test(text))
    || !tradeState.exitPanelVisible
    || tradeState.reasonHeading !== "平仓理由"
  ) {
    throw new Error(`historical-template trade state mismatch: ${JSON.stringify(tradeState)}`);
  }
  if (closure.verified_real_entry_bar_count !== manifest.trade_count) {
    throw new Error(`trade source verification mismatch: ${JSON.stringify(closure)}`);
  }
  await trade.locator("#entryReasonBtn").click();
  await trade.waitForFunction(() => document.querySelector("#reasonHeading")?.innerText === "开仓理由");
  const rangeRectangleState = await trade.evaluate(() => {
    const shapes = Array.from(document.querySelector("#chart")?.layout?.shapes || [])
      .filter((shape) => shape.type === "rect" && shape.xref === "x" && shape.yref === "paper");
    return {
      count: shapes.length,
      lineWidths: shapes.map((shape) => Number(shape.line?.width || 0)),
      fillcolors: shapes.map((shape) => String(shape.fillcolor || "")),
    };
  });
  if (
    rangeRectangleState.count < 1
    || rangeRectangleState.lineWidths.some((width) => width !== 0)
    || rangeRectangleState.fillcolors.some((color) => (
      !color || color.replace(/\s+/g, "").toLowerCase() === "rgba(0,0,0,0)"
    ))
  ) {
    throw new Error(`range rectangles must use fill only with no border: ${JSON.stringify(rangeRectangleState)}`);
  }
  const baseShapeCount = await trade.evaluate(() => document.querySelector("#chart")?.layout?.shapes?.length || 0);
  await trade.locator('[data-highlight="drop_points"]').hover();
  await trade.waitForFunction((baseCount) => (
    (document.querySelector("#chart")?.layout?.shapes?.length || 0) >= baseCount + 2
  ), baseShapeCount);
  const dropPointHighlightState = await trade.evaluate((baseCount) => {
    const allShapes = Array.from(document.querySelector("#chart")?.layout?.shapes || []);
    const shapes = allShapes.slice(baseCount);
    return {
      shapeCount: shapes.length,
      allShapeDashValues: allShapes
        .filter((shape) => Number(shape.line?.width || 0) > 0)
        .map((shape) => shape.line?.dash || "solid"),
      shapes: shapes.map((shape) => ({
        type: shape.type,
        fillcolor: shape.fillcolor,
        lineColor: shape.line?.color || "",
      })),
    };
  }, baseShapeCount);
  if (
    dropPointHighlightState.shapeCount !== 2
    || dropPointHighlightState.allShapeDashValues.some((dash) => dash !== "solid")
    || dropPointHighlightState.shapes.some((shape) => (
      shape.type !== "circle"
      || String(shape.fillcolor || "").replace(/\s+/g, "").toLowerCase() !== "rgba(0,0,0,0)"
      || !shape.lineColor
    ))
  ) {
    throw new Error(`entry-reason drop-point highlight must be hollow: ${JSON.stringify(dropPointHighlightState)}`);
  }
  await trade.locator("#reasonHeading").hover();
  const routedParams = await trade.evaluate(() => Object.fromEntries(new URL(location.href).searchParams.entries()));
  if (routedParams.reason !== "entry" || routedParams.research_contract_id !== manifest.campaign_id) {
    throw new Error(`trade query compatibility failed: ${JSON.stringify(routedParams)}`);
  }
  await trade.locator("#localBtn").click();
  await trade.waitForFunction(() => document.querySelector("#localBtn")?.classList.contains("active"));
  await trade.locator("#fullBtn").click();
  await trade.locator("#lightThemeBtn").click();
  await trade.locator("#controlsCollapse").click();
  await trade.waitForFunction(() => !document.querySelector("#controlsOpen")?.hidden);
  const parametersButtonState = await trade.evaluate(() => {
    const button = document.querySelector("#controlsOpen");
    const rect = button?.getBoundingClientRect();
    return {
      hidden: button?.hidden,
      computedTop: button ? getComputedStyle(button).top : "",
      top: rect?.top ?? null,
    };
  });
  if (
    parametersButtonState.hidden !== false
    || parametersButtonState.computedTop !== "92px"
    || Math.abs(Number(parametersButtonState.top) - 92) > 0.5
  ) {
    throw new Error(`parameters button vertical offset mismatch: ${JSON.stringify(parametersButtonState)}`);
  }
  await trade.locator("#controlsOpen").click();
  await trade.waitForFunction(() => document.querySelector("#controlsDrawer")?.dataset.open === "true");
  const tradeDesktop = await pageAudit(trade, [".view-switch", ".controls-drawer", ".detail-layout", "#chart", ".reason-panel"]);
  assertAudit(tradeDesktop, "trade desktop");
  const tradeDesktopPath = path.join(qaRoot, "trade_light_1440x950.png");
  await trade.screenshot({ path: tradeDesktopPath, fullPage: true });
  await trade.setViewportSize({ width: 390, height: 844 });
  await trade.locator("#controlsCollapse").click();
  await trade.waitForTimeout(200);
  const tradeMobile = await pageAudit(trade, [".view-switch", ".detail-layout", "#chart", ".reason-panel"]);
  assertAudit(tradeMobile, "trade mobile");
  const tradeMobilePath = path.join(qaRoot, "trade_mobile_390x844.png");
  await trade.screenshot({ path: tradeMobilePath, fullPage: true });

  if (runtimeErrors.length || externalRequests.length) {
    throw new Error(`browser diagnostics failed: ${JSON.stringify({ runtimeErrors, externalRequests })}`);
  }
  const result = {
    status: "passed",
    qa_mode: tradeOnly ? "focused_trade_template_repair" : "full_stage",
    state_count: stateCount,
    analysis_manifest_sha256: sha256(manifestPath),
    trade_review_manifest_sha256: sha256(reviewManifestPath),
    main_entry_sha256: sha256(mainPath),
    trade_entry_sha256: sha256(tradePath),
    scenario_requirements_entry_sha256: sha256(scenarioPath),
    main_data: mainData,
    high_return_view_states: highReturnViewStates,
    cost_adjusted_display_state: costAdjustedDisplayState,
    gross_display_state: grossDisplayState,
    scenario_states: scenarioStates,
    trade_state: tradeState,
    drop_point_highlight_state: dropPointHighlightState,
    parameters_button_state: parametersButtonState,
    routed_query_params: routedParams,
    unique_chunk_request_count: [...new Set(chunkRequests)].length,
    runtime_errors: runtimeErrors,
    external_requests: externalRequests,
    audits: { mainDesktop, mainMobile, scenarioDesktop, scenarioMobile, tradeDesktop, tradeMobile },
    screenshots: [mainDesktopPath, mainMobilePath, scenarioDesktopPath, scenarioMobilePath, tradeDesktopPath, tradeMobilePath],
  };
  const qaPath = path.join(qaRoot, "qa_result.json");
  await fsPromises.writeFile(qaPath, `${JSON.stringify(result, null, 2)}\n`, "utf8");
  process.stdout.write(JSON.stringify({ status: result.status, qaPath, states: stateCount, screenshots: result.screenshots }));
} finally {
  await browser.close();
}
