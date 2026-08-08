param(
    [string]$PackageName = 'Backtest_V4.41_source_release_20260809',
    [string]$OutputRoot = 'D:\Code\backtest-release'
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem

$projectRoot = Split-Path -Parent $PSScriptRoot
$packagePath = Join-Path $OutputRoot ($PackageName + '.zip')
$sidecarPath = $packagePath + '.sha256'
$auditPath = $packagePath + '.audit.json'
$stagingRoot = Join-Path (Join-Path $OutputRoot 'staging_recoverable') $PackageName

foreach ($target in @($packagePath, $sidecarPath, $auditPath, $stagingRoot)) {
    if (Test-Path -LiteralPath $target) {
        throw "Refusing to overwrite existing target: $target"
    }
}

function New-ParentDirectory([string]$Path) {
    $parent = Split-Path -Parent $Path
    if ($parent) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
}

function Copy-ProjectItem([string]$RelativePath, [string]$PackageRelativePath) {
    $source = Join-Path $projectRoot $RelativePath
    $target = Join-Path $stagingRoot $PackageRelativePath
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing package source: $source" }
    New-ParentDirectory $target
    Copy-Item -LiteralPath $source -Destination $target -Recurse
}

function Write-Utf8([string]$Path, [string]$Text) {
    New-ParentDirectory $Path
    [System.IO.File]::WriteAllText($Path, $Text, [System.Text.UTF8Encoding]::new($false))
}

function Get-RelativeSlashPath([string]$Root, [string]$Path) {
    return [System.IO.Path]::GetRelativePath($Root, $Path).Replace('\', '/')
}

New-Item -ItemType Directory -Path $stagingRoot -Force | Out-Null

$rootFiles = @(
    '.gitignore', '.python-version', 'AGENTS.md', 'AGENTS.zh.md', 'README.md',
    'GPTPRO_REVIEW_SCOPE.md', 'PRODUCT.md', 'RELEASE.json', 'RUNTIME.md', 'requirements-v4_4.txt',
    'package.json', 'package-lock.json'
)
foreach ($file in $rootFiles) { Copy-ProjectItem $file (Join-Path 'source\project' $file) }
Copy-ProjectItem 'GPTPRO_REVIEW_SCOPE.md' 'GPTPRO_REVIEW_SCOPE.md'

foreach ($directory in @('project_management', 'research_variants', 'runtime_inputs', 'tools')) {
    Copy-ProjectItem $directory (Join-Path 'source\project' $directory)
}
Copy-ProjectItem 'results\all_completed_union_analysis\current_snapshot.json' 'source\project\results\all_completed_union_analysis\current_snapshot.json'
Copy-ProjectItem 'results\all_completed_union_analysis\main\index.html' 'source\project\results\all_completed_union_analysis\main\index.html'
Copy-ProjectItem 'results\all_completed_union_analysis\main\analysis_data.js' 'source\project\results\all_completed_union_analysis\main\analysis_data.js'

$snapshotId = 'eb3398757b8ffe52332aec6ecdedc60df86b70afb4e1509c8fa3fcccd7b53dd5'
$snapshotRelative = Join-Path 'results\all_completed_union_analysis\snapshots' $snapshotId
$selectedComboId = 'v4_4_rolling_tr_sum_bpall_window_fillcalculated_threshold_execwait_next_real_trade_slip0_sx1_s340_rx1_e320_bh240_trw12_k1p25_w6_m4p5_ca63d64178'
$selectedChunk = 'c_f3252d3ccc494a46.js'

foreach ($mapping in @(
    @("$snapshotRelative\analysis_data.js", 'current_cumulative\analysis_data.js'),
    @("$snapshotRelative\index.html", 'current_cumulative\index.html'),
    @("$snapshotRelative\assets\plotly.min.js", 'current_cumulative\assets\plotly.min.js'),
    @("$snapshotRelative\trade_review\index.html", 'current_cumulative\trade_review\index.html'),
    @("$snapshotRelative\trade_review\all_results_catalog.js", 'current_cumulative\trade_review\all_results_catalog.js'),
    @("$snapshotRelative\trade_review\process_payload.js", 'current_cumulative\trade_review\process_payload.js'),
    @("$snapshotRelative\trade_review\trade_review_manifest.json", 'current_cumulative\trade_review\trade_review_manifest.json'),
    @("$snapshotRelative\trade_review\resource_audit.json", 'current_cumulative\trade_review\resource_audit.json'),
    @("$snapshotRelative\trade_review\v3_native_trades_js\$selectedChunk", "current_cumulative\trade_review\v3_native_trades_js\$selectedChunk"),
    @("$snapshotRelative\analysis_manifest.json", 'evidence\snapshot_manifests\analysis_manifest.json'),
    @("$snapshotRelative\completion_manifest.json", 'evidence\snapshot_manifests\completion_manifest.json'),
    @("$snapshotRelative\duplicate_coordinate_audit.json", 'evidence\snapshot_manifests\duplicate_coordinate_audit.json'),
    @("$snapshotRelative\source_stages.csv", 'evidence\snapshot_manifests\source_stages.csv')
)) {
    Copy-ProjectItem $mapping[0] $mapping[1]
}

$sourceManifestPath = Join-Path $projectRoot 'research_variants\short_momentum_net_drop_rebound_v4_4\SOURCE_MANIFEST.json'
$sourceManifestSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceManifestPath).Hash.ToLowerInvariant()

$releaseState = [ordered]@{
    schema_version = 3
    status = 'formal_source_release'
    package_date = '2026-08-09'
    release_identity = 'V4.41'
    strategy_ranking_major = 'V4.4'
    source_manifest_sha256 = $sourceManifestSha
    snapshot_id = $snapshotId
    coordinate_count = 37058
    trade_count = 11749606
    completed_stage_count = 109
    selected_research_contract_id = 'v4_4_all_completed_combined_union'
    selected_combo_id = $selectedComboId
    selected_trade_chunk = $selectedChunk
    full_trade_ledger_payload_included = $false
    existing_results_recomputed = $false
    observed_test_result = '0 failed, 112 passed, 2 skipped'
}
Write-Utf8 (Join-Path $stagingRoot 'CURRENT_RELEASE_STATE.json') ($releaseState | ConvertTo-Json -Depth 8)

$readme = @"
# Backtest V4.41 formal source release

This compact package is the formal V4.41 source release for Windows. It contains the released V4.41 source, the V4.4 strategy/ranking contract, project management, runtime inputs, the current cumulative browser payload, the stable main shell used by scenario tooling, and one representative per-trade chunk.

GPT Pro and other external reviewers must read `GPTPRO_REVIEW_SCOPE.md` before review. Findings that require omitted local data, local drives, raw ledgers, browser state, or machine-specific paths are outside this compact package's review scope.

The 3.5 GB full handoff and all raw per-stage trade ledgers are intentionally excluded. No raw trade, return, ranking, or retained result snapshot was recomputed or modified.

Current result authority: snapshot $snapshotId, 37,058 coordinates, 11,749,606 trades, and 109 completed stages.

HTML entry points:
- `current_cumulative/index.html`
- `current_cumulative/trade_review/index.html?research_contract_id=v4_4_all_completed_combined_union&combo_id=$selectedComboId`

Tests from the extracted package root:
```powershell
python -m pip install -r "source\project\requirements-v4_4.txt"
.\RUN_TESTS.ps1 -Python python
```

The extracted-package test gate contains the core suite and the focused market-scenario tool suite. The accepted result is 112 passed, 2 skipped, and 0 failed. Details are summarized in `TEST_STATUS.md`.
"@
Write-Utf8 (Join-Path $stagingRoot 'README_RELEASE.md') $readme

$testStatus = @"
# Current test status

Observed on 2026-08-09 with the extracted-package test selection:

- 112 passed
- 2 skipped
- 0 failed

The selection includes the core V4.4/V4.41 suite and `tools\test_v4_41_scenario_tools.py`. The two skips require closed local historical artifacts that are intentionally absent from the compact source release. The strategy engine, completed trades, returns, rankings, and retained result snapshots were not recomputed by the report, metadata, test, scenario-authoring, and packaging changes in this release.
"@
Write-Utf8 (Join-Path $stagingRoot 'TEST_STATUS.md') $testStatus

$testRunner = @'
param([string]$Python = "python")
$PackageRoot = $PSScriptRoot
$ProjectRoot = Join-Path $PackageRoot "source\project"
$CodeTests = "research_variants\short_momentum_net_drop_rebound_v4_4\code"
$DataPreparationTests = "research_variants\short_momentum_net_drop_rebound_v4_4\data_preparation\test_v4_4_data_preparation.py"
$ScenarioToolTests = "tools\test_v4_41_scenario_tools.py"
Push-Location -LiteralPath $ProjectRoot
try {
    & $Python -c "import numpy, pandas, pytest"
    if ($LASTEXITCODE -ne 0) { throw "Python dependencies are unavailable. Install source\project\requirements-v4_4.txt." }
    $env:PYTHONDONTWRITEBYTECODE = '1'
    & $Python -m pytest $CodeTests $DataPreparationTests $ScenarioToolTests -q -p no:cacheprovider
    $result = $LASTEXITCODE
}
finally { Pop-Location }
exit $result
'@
Write-Utf8 (Join-Path $stagingRoot 'RUN_TESTS.ps1') $testRunner

$manifestPath = Join-Path $stagingRoot 'PACKAGE_MANIFEST.json'
$records = @()
Get-ChildItem -LiteralPath $stagingRoot -Recurse -File | Where-Object { $_.FullName -ne $manifestPath } | Sort-Object FullName | ForEach-Object {
    $records += [ordered]@{
        entry_path = Get-RelativeSlashPath $stagingRoot $_.FullName
        size_bytes = $_.Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
    }
}
$recordLines = [string[]]($records | ForEach-Object { "$($_.entry_path)`t$($_.sha256)`t$($_.size_bytes)" })
[Array]::Sort($recordLines, [System.StringComparer]::Ordinal)
$recordSetHash = [System.BitConverter]::ToString(([System.Security.Cryptography.SHA256]::Create().ComputeHash([System.Text.Encoding]::UTF8.GetBytes(($recordLines -join "`n"))))).Replace('-', '').ToLowerInvariant()
$manifest = [ordered]@{
    schema_version = 2
    status = 'complete_formal_source_release'
    package_id = $PackageName
    release_identity = 'V4.41'
    strategy_ranking_major = 'V4.4'
    snapshot_id = $snapshotId
    coordinate_count = 37058
    trade_count = 11749606
    completed_stage_count = 109
    source_manifest_sha256 = $sourceManifestSha
    observed_test_result = '0 failed, 112 passed, 2 skipped'
    include_policy = @('package-level GPT Pro review-scope statement', 'released source and tests', 'project management', 'runtime inputs', 'market-scenario catalogs and focused tests', 'current cumulative browser payload and stable main shell', 'one representative per-trade chunk')
    exclude_policy = @('full raw/derived trade ledgers', 'all other per-coordinate trade chunks', 'bulk result CSVs', '.git, dependencies, caches, and local browser profiles')
    nonself_entry_count = $records.Count
    record_set_hash = $recordSetHash
    self_hash_policy = 'PACKAGE_MANIFEST.json is present in the ZIP but excluded from entries and record_set_hash.'
    entries = $records
}
Write-Utf8 $manifestPath ($manifest | ConvertTo-Json -Depth 8)

[System.IO.Compression.ZipFile]::CreateFromDirectory($stagingRoot, $packagePath, [System.IO.Compression.CompressionLevel]::Optimal, $false)
$packageSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $packagePath).Hash.ToLowerInvariant()
Write-Utf8 $sidecarPath "$packageSha *$([System.IO.Path]::GetFileName($packagePath))$([Environment]::NewLine)"

$archive = [System.IO.Compression.ZipFile]::OpenRead($packagePath)
$archiveEntryCount = $archive.Entries.Count
$archive.Dispose()

$audit = [ordered]@{
    status = 'formal_source_release_tests_passed'
    package_path = $packagePath
    package_size_bytes = (Get-Item -LiteralPath $packagePath).Length
    package_sha256 = $packageSha
    sidecar_path = $sidecarPath
    source_manifest_sha256 = $sourceManifestSha
    snapshot_id = $snapshotId
    archive_entry_count = $archiveEntryCount
    manifest_nonself_entry_count = $records.Count
    record_set_hash = $recordSetHash
    observed_test_result = '0 failed, 112 passed, 2 skipped'
    staging_root = $stagingRoot
    existing_results_recomputed = $false
}
Write-Utf8 $auditPath ($audit | ConvertTo-Json -Depth 8)
$audit | ConvertTo-Json -Depth 8
