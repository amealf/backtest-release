[CmdletBinding()]
param(
    [string]$ProjectRoot = 'D:\Code\backtest-release\Backtest V4.4',
    [string]$OutputDirectory = 'D:\Code\backtest-release',
    [string]$PackageName = 'Backtest_V4.4_with_trade_records_20260803'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Get-Sha256([string]$Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
        $stream.Dispose()
    }
}

function Get-RelativeSlashPath([string]$BasePath, [string]$Path) {
    $baseFull = [System.IO.Path]::GetFullPath($BasePath).TrimEnd('\') + '\'
    $pathFull = [System.IO.Path]::GetFullPath($Path)
    $baseUri = [System.Uri]::new($baseFull)
    $pathUri = [System.Uri]::new($pathFull)
    return [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($pathUri).ToString()).Replace('\', '/')
}

function Write-Utf8Json([string]$Path, $Value) {
    $directory = Split-Path -Parent $Path
    [System.IO.Directory]::CreateDirectory($directory) | Out-Null
    $json = $Value | ConvertTo-Json -Depth 16
    [System.IO.File]::WriteAllText($Path, $json + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
}

function Get-CsvRecordCount([string]$Path) {
    $reader = [System.IO.File]::OpenText($Path)
    try {
        $lines = 0
        while ($null -ne $reader.ReadLine()) { $lines++ }
        return [Math]::Max(0, $lines - 1)
    }
    finally {
        $reader.Dispose()
    }
}

function Copy-PackageFile([string]$SourcePath, [string]$TargetPath) {
    [System.IO.Directory]::CreateDirectory((Split-Path -Parent $TargetPath)) | Out-Null
    Copy-Item -LiteralPath $SourcePath -Destination $TargetPath -Force
    $sourceHash = Get-Sha256 $SourcePath
    $targetHash = Get-Sha256 $TargetPath
    if ($sourceHash -ne $targetHash) {
        throw "Copy hash mismatch: $SourcePath"
    }
}

function Copy-Tree([string]$SourceRoot, [string]$TargetRoot) {
    Get-ChildItem -LiteralPath $SourceRoot -Recurse -File | ForEach-Object {
        $segments = (Get-RelativeSlashPath $SourceRoot $_.FullName).Split('/')
        if ($segments | Where-Object { $_ -in @('.git', '.omo', 'node_modules', '.pytest_cache', '__pycache__', '.cache') }) {
            return
        }
        if ($_.Extension -in @('.pyc', '.pyo')) {
            return
        }
        Copy-PackageFile $_.FullName (Join-Path $TargetRoot (Get-RelativeSlashPath $SourceRoot $_.FullName))
    }
}

function Get-StreamSha256([System.IO.Stream]$Stream) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha.ComputeHash($Stream))).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

$projectRootResolved = (Resolve-Path -LiteralPath $ProjectRoot).Path
$outputRootResolved = (Resolve-Path -LiteralPath $OutputDirectory).Path
$zipPath = Join-Path $outputRootResolved ($PackageName + '.zip')
$sidecarPath = $zipPath + '.sha256'
$auditPath = Join-Path $outputRootResolved ($PackageName + '.zip.audit.json')
$stagingRoot = Join-Path (Join-Path $outputRootResolved 'staging_recoverable') $PackageName
$extractRoot = Join-Path (Join-Path $outputRootResolved 'staging_recoverable') ($PackageName + '_extracted')

foreach ($path in @($zipPath, $sidecarPath, $auditPath, $stagingRoot, $extractRoot)) {
    if (Test-Path -LiteralPath $path) {
        throw "Refusing to overwrite existing package target: $path"
    }
}

$runningStatuses = @(
    Get-ChildItem -LiteralPath (Join-Path $projectRootResolved 'results\campaigns') -Recurse -File -Filter 'progress.json' -ErrorAction SilentlyContinue
    Get-ChildItem -LiteralPath (Join-Path $projectRootResolved 'results\campaigns') -Recurse -File -Filter 'delivery_status.json' -ErrorAction SilentlyContinue
) | ForEach-Object {
    $status = (Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json).status
    if ($status -eq 'running') { $_.FullName }
}
if ($runningStatuses) {
    throw "Packaging is blocked by active compute or delivery status: $($runningStatuses -join '; ')"
}

[System.IO.Directory]::CreateDirectory($stagingRoot) | Out-Null

# Current source, runtime, and management material.
$rootFiles = @('.gitignore', '.python-version', 'AGENTS.md', 'AGENTS.zh.md', 'README.md', 'PRODUCT.md', 'RUNTIME.md', 'requirements-v4_4.txt', 'package.json', 'package-lock.json')
foreach ($name in $rootFiles) {
    $source = Join-Path $projectRootResolved $name
    if (Test-Path -LiteralPath $source) {
        Copy-PackageFile $source (Join-Path $stagingRoot $name)
    }
}
Copy-Tree (Join-Path $projectRootResolved 'research_variants\short_momentum_net_drop_rebound_v4_4') (Join-Path $stagingRoot 'research_variants\short_momentum_net_drop_rebound_v4_4')
Copy-Tree (Join-Path $projectRootResolved 'runtime_inputs') (Join-Path $stagingRoot 'runtime_inputs')
Copy-Tree (Join-Path $projectRootResolved 'project_management') (Join-Path $stagingRoot 'project_management')
Copy-Tree (Join-Path $projectRootResolved 'tools') (Join-Path $stagingRoot 'tools')

# Canonical reports are copied from team evidence into a clean package path.
$artifactRoot = Join-Path $projectRootResolved '.omo\teams\019fbd76-9439-7b10-a7e1-6b34003778c8\artifacts'
$reportNames = @(
    'v4_4_cost_adjusted_multiround_design_20260803.md',
    'round_01_interpretation_and_round_02_design_20260803.md',
    'round_01_delivery_final_C.md',
    'round_02_interpretation_and_round_03_terminal_design_20260803.md',
    'round_02_delivery_final_C.md',
    'round_03_terminal_interpretation_and_campaign_closure_20260803.md',
    'round_03_terminal_delivery_final_C.md',
    'v4_4_continuation_subseries_design_20260803.md',
    'continuation_round_01_delivery_final_C.md'
)
$reportRecords = @()
foreach ($name in $reportNames) {
    $source = Join-Path $artifactRoot $name
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Required canonical report is absent: $source"
    }
    $target = Join-Path $stagingRoot (Join-Path 'analysis_reports' $name)
    Copy-PackageFile $source $target
    $reportRecords += [ordered]@{
        report_name = $name
        package_entry = (Get-RelativeSlashPath $stagingRoot $target)
        source_artifact_name = $name
        size_bytes = (Get-Item -LiteralPath $source).Length
        sha256 = Get-Sha256 $source
        original_bytes_preserved = $true
    }
}
$reportsManifest = [ordered]@{
    schema_version = 1
    report_count = $reportRecords.Count
    source_policy = 'Canonical reports are copied byte-for-byte from retained team evidence without including .omo in the archive.'
    reports = $reportRecords
}
Write-Utf8Json (Join-Path $stagingRoot 'analysis_reports\REPORTS_MANIFEST.json') $reportsManifest

# The narrow results exception: immutable raw ledgers and derived per-stage ledgers.
$campaignsRoot = Join-Path $projectRootResolved 'results\campaigns'
$tradeRecords = @()
$completedStageCount = 0
Get-ChildItem -LiteralPath $campaignsRoot -Recurse -File -Filter 'completion_manifest.json' | Sort-Object FullName | ForEach-Object {
    $stageRoot = Split-Path -Parent $_.FullName
    $completion = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
    if ($completion.status -ne 'complete') { return }
    $stageManifestPath = Join-Path $stageRoot 'stage_manifest.json'
    if (-not (Test-Path -LiteralPath $stageManifestPath)) {
        throw "Completed stage lacks stage_manifest.json: $stageRoot"
    }
    $stageRelative = Get-RelativeSlashPath $campaignsRoot $stageRoot
    $completionHash = Get-Sha256 $_.FullName
    $stageManifestHash = Get-Sha256 $stageManifestPath
    $batchTrades = Get-ChildItem -LiteralPath (Join-Path $stageRoot 'batches') -Recurse -File -Filter 'trades.csv' | Sort-Object FullName
    if (-not $batchTrades) {
        throw "Completed stage lacks raw trade CSV files: $stageRoot"
    }
    foreach ($tradeFile in $batchTrades) {
        $batchName = Split-Path -Leaf (Split-Path -Parent $tradeFile.FullName)
        $target = Join-Path $stagingRoot (Join-Path 'trade_records\raw_batches' (Join-Path $stageRelative (Join-Path $batchName 'trades.csv')))
        Copy-PackageFile $tradeFile.FullName $target
        $tradeRecords += [ordered]@{
            record_role = 'immutable_raw_batch_trade_ledger'
            campaign_stage = $stageRelative
            source_relative_path = (Get-RelativeSlashPath $projectRootResolved $tradeFile.FullName)
            package_entry = (Get-RelativeSlashPath $stagingRoot $target)
            row_count = Get-CsvRecordCount $tradeFile.FullName
            size_bytes = (Get-Item -LiteralPath $tradeFile.FullName).Length
            sha256 = Get-Sha256 $tradeFile.FullName
            completion_manifest_sha256 = $completionHash
            stage_manifest_sha256 = $stageManifestHash
        }
    }
    $stageTradesPath = Join-Path $stageRoot 'analysis\stage_trades.csv'
    if (-not (Test-Path -LiteralPath $stageTradesPath)) {
        throw "Completed stage lacks derived stage_trades.csv: $stageRoot"
    }
    $derivedTarget = Join-Path $stagingRoot (Join-Path 'trade_records\stage_derived' (Join-Path $stageRelative 'stage_trades.csv'))
    Copy-PackageFile $stageTradesPath $derivedTarget
    $tradeRecords += [ordered]@{
        record_role = 'derived_stage_trade_ledger'
        campaign_stage = $stageRelative
        source_relative_path = (Get-RelativeSlashPath $projectRootResolved $stageTradesPath)
        package_entry = (Get-RelativeSlashPath $stagingRoot $derivedTarget)
        row_count = Get-CsvRecordCount $stageTradesPath
        size_bytes = (Get-Item -LiteralPath $stageTradesPath).Length
        sha256 = Get-Sha256 $stageTradesPath
        completion_manifest_sha256 = $completionHash
        stage_manifest_sha256 = $stageManifestHash
    }
    $completedStageCount++
}
if ($completedStageCount -eq 0) {
    throw 'No completed V4.4 stages were found for transaction-record packaging.'
}
$tradeManifest = [ordered]@{
    schema_version = 1
    completed_stage_count = $completedStageCount
    record_count = $tradeRecords.Count
    policy = 'Only immutable raw per-batch trades.csv and derived stage_trades.csv from completed stages are included. All other results payloads are excluded.'
    records = $tradeRecords
}
Write-Utf8Json (Join-Path $stagingRoot 'trade_records\TRADE_RECORDS_MANIFEST.json') $tradeManifest

$packageReadme = @"
# Backtest V4.4 handoff package

This archive contains the current V4.4 code, complete project-management tree, canonical analysis reports, repository-local runtime inputs including the hash-bound 15-second OHLC, and the narrow transaction-record exception under `trade_records/`.

`trade_records/raw_batches/` holds immutable per-batch trade ledgers. `trade_records/stage_derived/` holds the corresponding derived stage ledgers. `trade_records/TRADE_RECORDS_MANIFEST.json` binds every included ledger to its stage and source identities.

All other compute-result payloads are excluded. The archive also excludes `.omo`, `.git`, caches, dependencies, browser profiles, compiled bytecode, and sensitive material.
"@
[System.IO.File]::WriteAllText((Join-Path $stagingRoot 'PACKAGE_README.md'), $packageReadme, [System.Text.UTF8Encoding]::new($false))

$manifestPath = Join-Path $stagingRoot 'PACKAGE_MANIFEST.json'
$records = @()
Get-ChildItem -LiteralPath $stagingRoot -Recurse -File | Where-Object { $_.FullName -ne $manifestPath } | Sort-Object FullName | ForEach-Object {
    $records += [ordered]@{
        entry_path = Get-RelativeSlashPath $stagingRoot $_.FullName
        size_bytes = $_.Length
        sha256 = Get-Sha256 $_.FullName
    }
}
$recordLines = [string[]]($records | ForEach-Object { "$($_.entry_path)`t$($_.sha256)`t$($_.size_bytes)" })
[Array]::Sort($recordLines, [System.StringComparer]::Ordinal)
$recordSetHash = [System.BitConverter]::ToString(([System.Security.Cryptography.SHA256]::Create().ComputeHash([System.Text.Encoding]::UTF8.GetBytes(($recordLines -join "`n"))))).Replace('-', '').ToLowerInvariant()
$packageManifest = [ordered]@{
    schema_version = 2
    status = 'complete'
    package_id = $PackageName
    project_identity = 'Backtest V4.4'
    created_at_utc = [DateTime]::UtcNow.ToString('o')
    include_policy = @(
        'Current V4.4 root reproducibility/runtime/configuration documents',
        'Complete project_management tree',
        'Current research variant code, plans, tests, scripts, and data-preparation source',
        'Complete runtime_inputs tree including hash-bound 15-second OHLC',
        'Canonical analysis reports copied to analysis_reports',
        'Immutable raw per-batch and derived per-stage transaction CSVs copied to trade_records'
    )
    exclude_policy = @(
        'All result payloads except trade_records raw batches and derived stage ledgers',
        'Entire .omo team state',
        '.git, node_modules, .pytest_cache, __pycache__, .cache, browser profiles, caches, compiled bytecode',
        'Secrets and credential material'
    )
    hash_bound_dataset = [ordered]@{
        entry_path = 'runtime_inputs/market_data/k200_clean_15s_session_filled.csv'
        sha256 = Get-Sha256 (Join-Path $stagingRoot 'runtime_inputs\market_data\k200_clean_15s_session_filled.csv')
        size_bytes = (Get-Item -LiteralPath (Join-Path $stagingRoot 'runtime_inputs\market_data\k200_clean_15s_session_filled.csv')).Length
    }
    analysis_reports = [ordered]@{
        report_count = $reportRecords.Count
        reports_manifest_entry = 'analysis_reports/REPORTS_MANIFEST.json'
    }
    transaction_records = [ordered]@{
        completed_stage_count = $completedStageCount
        record_count = $tradeRecords.Count
        trade_records_manifest_entry = 'trade_records/TRADE_RECORDS_MANIFEST.json'
    }
    entry_record_policy = [ordered]@{
        self_hash_recursion = 'PACKAGE_MANIFEST.json is present in the ZIP but excluded from entries and record_set_hash.'
        record_set_hash_algorithm = 'SHA-256 of ordinal-sorted entry_path<TAB>sha256<TAB>size lines joined by LF with no terminal newline.'
        record_set_hash = $recordSetHash
    }
    entries = $records
}
Write-Utf8Json $manifestPath $packageManifest

[System.IO.Compression.ZipFile]::CreateFromDirectory($stagingRoot, $zipPath, [System.IO.Compression.CompressionLevel]::Optimal, $false)
$zipHash = Get-Sha256 $zipPath
[System.IO.File]::WriteAllText($sidecarPath, "$zipHash *$([System.IO.Path]::GetFileName($zipPath))$([Environment]::NewLine)", [System.Text.UTF8Encoding]::new($false))
if ((Get-Sha256 $zipPath) -ne $zipHash) {
    throw 'ZIP hash changed after sidecar creation.'
}

$archive = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    $entriesByNormalizedPath = @{}
    foreach ($entry in $archive.Entries) {
        $normalizedEntryPath = $entry.FullName.Replace([char]92, [char]47)
        if (-not $seen.Add($normalizedEntryPath)) { throw "Duplicate ZIP entry: $normalizedEntryPath" }
        $entriesByNormalizedPath[$normalizedEntryPath] = $entry
    }
    if ($archive.Entries.Count -ne ($records.Count + 1)) {
        throw "ZIP entry count mismatch: $($archive.Entries.Count) versus $($records.Count + 1)"
    }
    foreach ($record in $records) {
        $entry = $entriesByNormalizedPath[$record.entry_path]
        if ($null -eq $entry) { throw "Missing ZIP entry: $($record.entry_path)" }
        if ($entry.Length -ne [int64]$record.size_bytes) { throw "ZIP size mismatch: $($record.entry_path)" }
        $stream = $entry.Open()
        try { $streamHash = Get-StreamSha256 $stream } finally { $stream.Dispose() }
        if ($streamHash -ne $record.sha256) { throw "ZIP hash mismatch: $($record.entry_path)" }
    }
}
finally {
    $archive.Dispose()
}

[System.IO.Compression.ZipFile]::ExtractToDirectory($zipPath, $extractRoot)
foreach ($record in $records) {
    $extracted = Join-Path $extractRoot ($record.entry_path.Replace('/', '\'))
    if (-not (Test-Path -LiteralPath $extracted)) { throw "Extraction missing: $($record.entry_path)" }
    if ((Get-Item -LiteralPath $extracted).Length -ne [int64]$record.size_bytes) { throw "Extraction size mismatch: $($record.entry_path)" }
    if ((Get-Sha256 $extracted) -ne $record.sha256) { throw "Extraction hash mismatch: $($record.entry_path)" }
}

$forbidden = Get-ChildItem -LiteralPath $stagingRoot -Recurse -File | ForEach-Object { Get-RelativeSlashPath $stagingRoot $_.FullName } | Where-Object {
    $_ -match '(^|/)(\.omo|\.git|node_modules|\.pytest_cache|__pycache__|\.cache)(/|$)' -or $_ -match '\.(pyc|pyo)$'
}
if ($forbidden) { throw "Forbidden package paths: $($forbidden -join '; ')" }

$audit = [ordered]@{
    status = 'pass'
    package_path = $zipPath
    package_size_bytes = (Get-Item -LiteralPath $zipPath).Length
    package_sha256 = $zipHash
    sidecar_path = $sidecarPath
    sidecar_sha256 = Get-Sha256 $sidecarPath
    staging_root = $stagingRoot
    extraction_root = $extractRoot
    archive_entry_count = $records.Count + 1
    manifest_nonself_entry_count = $records.Count
    transaction_record_count = $tradeRecords.Count
    completed_stage_count = $completedStageCount
    record_set_hash = $recordSetHash
    zip_stream_checks = $records.Count
    extraction_checks = $records.Count
    forbidden_entry_count = 0
    duplicate_entry_count = 0
}
Write-Utf8Json $auditPath $audit
$audit | ConvertTo-Json -Depth 8
