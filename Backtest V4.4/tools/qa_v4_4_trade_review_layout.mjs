import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { chromium } from "playwright";


const projectRoot = path.resolve(import.meta.dirname, "..");
const runRoot = path.join(
  projectRoot,
  "results",
  "cross_instrument_comparison",
  "runs",
  "k200_train_test_si__combined_350_v56_20260807",
);
const testEntry = path.join(runRoot, "trade_review_k200_test", "index.html");
const trainingEntry = path.join(
  projectRoot,
  "results",
  "all_completed_union_analysis",
  "trade_review",
  "index.html",
);
const comboId =
  "v4_4_rolling_tr_sum_bpall_window_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s332_rx1_e320_bh240_trw20_k1p4_w6_m4p5_7d214509a0";
const outputRoot = path.join(
  projectRoot,
  "project_management",
  "screenshots",
  "trade_review_layout_20260808",
);
fs.mkdirSync(outputRoot, { recursive: true });

function entryUrl(filePath, researchContractId) {
  const url = new URL(pathToFileURL(filePath).href);
  url.searchParams.set("combo_id", comboId);
  url.searchParams.set("research_contract_id", researchContractId);
  return url.href;
}

async function waitForReview(page) {
  await page.waitForSelector("#chart .main-svg", { timeout: 45_000 });
  await page.waitForFunction(
    () => document.querySelector("#tradeSelect")?.options.length > 0,
    null,
    { timeout: 45_000 },
  );
  await page.waitForTimeout(250);
}

async function waitForPeerButton(page) {
  await page.waitForFunction(
    () => !document.querySelector("#peerReviewLink")?.hidden,
    null,
    { timeout: 45_000 },
  );
}

async function waitForPeerFrame(page) {
  for (let attempt = 0; attempt < 90; attempt += 1) {
    const frame = page
      .frames()
      .find((candidate) => candidate !== page.mainFrame() && candidate.url().includes("embedded=1"));
    if (frame) {
      await waitForReview(frame);
      return frame;
    }
    await page.waitForTimeout(100);
  }
  throw new Error("paired review frame did not load");
}

async function state(page) {
  return page.evaluate(() => {
    const chart = document.getElementById("chart");
    const peer = document.getElementById("peerReviewLink");
    const peerStyle = peer ? getComputedStyle(peer) : null;
    const overlay = document.getElementById("peerReviewOverlay");
    const peerFrame = document.getElementById("peerReviewFrame");
    const annotations = chart?.layout?.annotations || [];
    return {
      title: document.title,
      favicon: document.querySelector('link[rel="icon"]')?.href || "",
      peerLabel: peer?.textContent?.trim() || "",
      peerDisplay: peerStyle?.display || "",
      peerAlignItems: peerStyle?.alignItems || "",
      peerJustifyContent: peerStyle?.justifyContent || "",
      peerTextDecoration: peerStyle?.textDecorationLine || "",
      peerTarget: peerFrame?.getAttribute("src") || "",
      peerOverlayOpen: Boolean(overlay && !overlay.hidden),
      embeddedSelectionCardHidden:
        document.documentElement.dataset.embeddedReview === "true" &&
        getComputedStyle(document.querySelector(".selection-card")).display === "none",
      tradePickerHeaderExists: Boolean(document.querySelector(".trade-picker-head")),
      tradePickerMetaExists: Boolean(document.querySelector(".trade-picker-meta")),
      metrics: document.getElementById("tradeMetrics")?.innerText || "",
      legendY: chart?.layout?.legend?.y,
      chartAuditAnnotationCount: annotations.filter((item) =>
        /drop=|ratio=|基准=|阈值=|W基准=|^L=/.test(String(item.text || "")),
      ).length,
      url: location.href,
      errors: [],
    };
  });
}

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 2048, height: 1152 } });
const page = await context.newPage();
const errors = [];
page.on("console", (message) => {
  if (message.type() === "error") errors.push(message.text());
});
page.on("pageerror", (error) => errors.push(error.message));

await page.goto(entryUrl(testEntry, "v4_4_cross_instrument_comparison"), {
  waitUntil: "load",
});
await waitForReview(page);
await waitForPeerButton(page);
await page.click("#controlsCollapse");
await page.evaluate(() => {
  const reboundIndex = currentTrades.findIndex(
    (trade) => String(trade.exit_reason || "").toLowerCase() === "rebound_threshold",
  );
  if (reboundIndex < 0) return;
  const select = document.getElementById("tradeSelect");
  select.value = String(reboundIndex);
  select.dispatchEvent(new Event("change", { bubbles: true }));
});
await page.waitForTimeout(300);
const test = await state(page);
await page.hover("#peerReviewLink");
const peerHoverTextDecoration = await page.$eval(
  "#peerReviewLink",
  (node) => getComputedStyle(node).textDecorationLine,
);
await page.screenshot({ path: path.join(outputRoot, "test-dark.png"), fullPage: false });

await page.click("#lightThemeBtn");
await page.waitForTimeout(250);
await page.screenshot({ path: path.join(outputRoot, "test-light.png"), fullPage: false });

const testUrlBeforePeer = page.url();
await page.click("#peerReviewLink");
const trainingFrame = await waitForPeerFrame(page);
const testWithTrainingOpen = await state(page);
const training = await state(trainingFrame);
await page.screenshot({ path: path.join(outputRoot, "test-with-training-inline.png"), fullPage: false });
await page.click("#peerReviewClose");
await page.waitForFunction(() => document.getElementById("peerReviewOverlay")?.hidden === true);
const testAfterPeerClose = await state(page);

await page.goto(entryUrl(trainingEntry, "v4_4_all_completed_combined_union"), {
  waitUntil: "load",
});
await waitForReview(page);
await waitForPeerButton(page);
await page.click("#controlsCollapse");
await page.waitForTimeout(250);
const trainingUrlBeforePeer = page.url();
await page.click("#peerReviewLink");
const testFrame = await waitForPeerFrame(page);
const trainingWithTestOpen = await state(page);
const pairedTest = await state(testFrame);
await page.screenshot({ path: path.join(outputRoot, "training-with-test-inline.png"), fullPage: false });

const mobile = await browser.newPage({ viewport: { width: 390, height: 844 } });
const mobileErrors = [];
mobile.on("console", (message) => {
  if (message.type() === "error") mobileErrors.push(message.text());
});
mobile.on("pageerror", (error) => mobileErrors.push(error.message));
await mobile.goto(entryUrl(testEntry, "v4_4_cross_instrument_comparison"), {
  waitUntil: "load",
});
await waitForReview(mobile);
await waitForPeerButton(mobile);
await mobile.click("#controlsCollapse");
const mobileOverflow = await mobile.evaluate(
  () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
);
await mobile.screenshot({ path: path.join(outputRoot, "test-mobile.png"), fullPage: false });
await browser.close();

const assertions = {
  title_has_no_version: !/V\d/i.test(test.title) && test.title === "组合平仓逐笔查看",
  favicon_is_blue_z: test.favicon.includes("%231f77d0") && test.favicon.includes("%3EZ%3C"),
  peer_button_matches_toolbar:
    ["flex", "inline-flex"].includes(test.peerDisplay) &&
    test.peerAlignItems === "center" &&
    test.peerJustifyContent === "center" &&
    test.peerTextDecoration === "none" &&
    peerHoverTextDecoration === "none",
  test_opens_training_inline:
    test.peerLabel === "显示训练集" &&
    testWithTrainingOpen.peerOverlayOpen &&
    testWithTrainingOpen.url === testUrlBeforePeer &&
    new URL(testWithTrainingOpen.peerTarget, testUrlBeforePeer).searchParams.get("combo_id") === comboId &&
    new URL(testWithTrainingOpen.peerTarget, testUrlBeforePeer).searchParams.get("embedded") === "1" &&
    training.peerLabel === "显示测试集" &&
    training.embeddedSelectionCardHidden,
  inline_panel_can_hide:
    !testAfterPeerClose.peerOverlayOpen &&
    testAfterPeerClose.peerTarget === "",
  training_opens_test_inline:
    trainingWithTestOpen.peerOverlayOpen &&
    trainingWithTestOpen.url === trainingUrlBeforePeer &&
    new URL(trainingWithTestOpen.peerTarget, trainingUrlBeforePeer).searchParams.get("combo_id") === comboId &&
    new URL(trainingWithTestOpen.peerTarget, trainingUrlBeforePeer).searchParams.get("embedded") === "1" &&
    pairedTest.peerLabel === "显示训练集" &&
    pairedTest.embeddedSelectionCardHidden,
  picker_extras_removed:
    !test.tradePickerHeaderExists && !test.tradePickerMetaExists,
  metrics_moved_below_reason:
    /drop=/.test(test.metrics) &&
    /ratio=/.test(test.metrics) &&
    /基准=/.test(test.metrics) &&
    /阈值=/.test(test.metrics) &&
    test.chartAuditAnnotationCount === 0,
  legend_below_chart: Number(test.legendY) < 0,
  no_desktop_errors: errors.length === 0,
  no_mobile_errors: mobileErrors.length === 0,
  no_mobile_page_overflow: mobileOverflow === false,
};
const report = {
  schema_version: 1,
  status: Object.values(assertions).every(Boolean) ? "passed" : "failed",
  assertions,
  test,
  training,
  pairedTest,
  testWithTrainingOpen,
  testAfterPeerClose,
  trainingWithTestOpen,
  peerHoverTextDecoration,
  errors,
  mobileErrors,
  mobileOverflow,
};
fs.writeFileSync(
  path.join(outputRoot, "qa-report.json"),
  `${JSON.stringify(report, null, 2)}\n`,
  "utf8",
);
process.stdout.write(`${JSON.stringify(assertions, null, 2)}\n`);
if (report.status !== "passed") process.exitCode = 1;
