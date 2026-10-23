param([switch]$ExecuteApproved)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
[Diagnostics.Process]::GetCurrentProcess().PriorityClass = 'BelowNormal'
$report = $PSScriptRoot
$rawRoot = 'C:\Users\hamediaa\RL_MPC_local_archive\storage_20261023'
$zipRoot = 'C:\Users\hamediaa\RL_MPC_local_archive\compressed_storage_20261023'
$manifestPath = Join-Path $zipRoot 'restore_manifest.json'
$manifestHash = '97af00c0dfbf371c8f2141d417d26b4ed09e6d272a0e609cb43be208745f96a9'
$codeRoot = [IO.Path]::GetFullPath((Join-Path $report '..\..'))

function Save-Json($Path, $Value) {
    $temporary = $Path + '.tmp'
    $Value | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Resolve-ArchiveChild([string]$Path, [string]$Root) {
    $full = [IO.Path]::GetFullPath($Path)
    $prefix = [IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes approved archive root: $full"
    }
    $cursor = $full
    while ($cursor.Length -ge $Root.Length) {
        $item = Get-Item -LiteralPath $cursor -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Refusing a reparse point: $cursor"
        }
        if ($cursor.Equals($Root, [StringComparison]::OrdinalIgnoreCase)) { break }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $full
}

if ((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifestHash) {
    throw 'The approved restoration manifest changed.'
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$records = @($manifest.archives)
if ($records.Count -ne 910) { throw 'Expected exactly 910 approved archive folders.' }
if (@($records.original_directory | Sort-Object -Unique).Count -ne 910) {
    throw 'Duplicate archive paths.'
}
Add-Type -AssemblyName System.IO.Compression

# Prove that current E1 recovery files are outside the approved historical root.
$assessment = Get-Content -LiteralPath (Join-Path $codeRoot 'change-reports\e1_recovery_20261023\recovery_assessment.json') -Raw | ConvertFrom-Json
$liveRecovery = foreach ($job in @($assessment.jobs | Where-Object study -eq 'E1_column')) {
    $indexPath = Join-Path $job.directory 'training_checkpoints\latest.json'
    $index = Get-Content -LiteralPath $indexPath -Raw | ConvertFrom-Json
    foreach ($file in @($index.native.file, $index.native.project_file, (Join-Path (Split-Path $indexPath) $index.file))) {
        if ([IO.Path]::GetFullPath($file).StartsWith($rawRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "A current recovery checkpoint depends on the historical archive: $file"
        }
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing current checkpoint: $file" }
    }
    [ordered]@{job=$job.job; completed_steps=$index.completed_steps; checkpoint=$indexPath}
}

$before = [IO.DriveInfo]::new('C:\').AvailableFreeSpace
$processes = @(Get-Process | Where-Object { $_.ProcessName -match 'python|aspen|adsim' } | ForEach-Object {
    $observedProcess = $_
    try { [ordered]@{pid=$observedProcess.Id; name=$observedProcess.ProcessName; created_utc=$observedProcess.StartTime.ToUniversalTime().ToString('o')} }
    catch { [ordered]@{pid=$observedProcess.Id; name=$observedProcess.ProcessName; created_utc=$null} }
})
Save-Json (Join-Path $report 'before.json') ([ordered]@{
    at_utc=[DateTime]::UtcNow.ToString('o'); free_bytes=$before; processes=$processes; current_e1_recovery=$liveRecovery
})

$verified = [Collections.Generic.List[object]]::new()
$originalBytes = [long]0
$originalFiles = 0
foreach ($record in $records) {
    $target = Resolve-ArchiveChild $record.original_directory $rawRoot
    if ([IO.Path]::GetFileName($target) -notmatch '^native_[01]_\d+$') {
        throw "Unexpected historical folder: $target"
    }
    $zip = Resolve-ArchiveChild $record.compressed $zipRoot
    if ((Get-Item -LiteralPath $zip).Length -ne $record.compressed_bytes -or
        (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $record.compressed_sha256) {
        throw "Compressed archive verification failed: $zip"
    }
    $all = @(Get-ChildItem -LiteralPath $target -Recurse -Force)
    if (@($all | Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 }).Count) {
        throw "Unexpected reparse point under $target"
    }
    $files = @($all | Where-Object { -not $_.PSIsContainer })
    if ($files.Count -ne @($record.files).Count) { throw "Unexpected file count under $target" }
    $stream = [IO.File]::OpenRead($zip)
    $archive = [IO.Compression.ZipArchive]::new($stream, [IO.Compression.ZipArchiveMode]::Read)
    $identities = @()
    try {
        if ($archive.Entries.Count -ne @($record.files).Count) { throw "Unexpected ZIP entry count: $zip" }
        foreach ($file in $record.files) {
            $path = Resolve-ArchiveChild (Join-Path $target $file.relative) $target
            $item = Get-Item -LiteralPath $path -Force
            if ($item.PSIsContainer -or $item.Length -ne $file.bytes -or
                (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) {
                throw "Original file verification failed: $path"
            }
            $entry = $archive.GetEntry($file.relative)
            if ($null -eq $entry -or $entry.Length -ne $file.bytes) { throw "ZIP mapping differs: $path" }
            $identities += [ordered]@{path=$path; bytes=$item.Length; ticks=$item.LastWriteTimeUtc.Ticks}
            $originalBytes += $item.Length
            $originalFiles++
        }
    }
    finally { $archive.Dispose(); $stream.Dispose() }
    $verified.Add([ordered]@{directory=$target; zip=$zip; zip_sha256=$record.compressed_sha256; files=$identities})
    if ($verified.Count % 25 -eq 0) {
        Write-Output ("Verified {0}/910 folders; {1:N2} GiB" -f $verified.Count, ($originalBytes / 1GB))
    }
}
if ($originalFiles -ne 3640 -or $originalBytes -ne 174583311764) { throw 'Approved totals differ.' }
Save-Json (Join-Path $report 'verified_deletion_plan.json') ([ordered]@{
    at_utc=[DateTime]::UtcNow.ToString('o'); manifest_sha256=$manifestHash; raw_root=$rawRoot;
    compressed_root=$zipRoot; files=$originalFiles; original_bytes=$originalBytes; folders=$verified
})
if (-not $ExecuteApproved) { Write-Output 'Verified plan only; no original deleted.'; exit 0 }

# The user explicitly approved removal of precisely these 910 duplicate folders.
# Every ZIP and original was hashed above before any deletion. Keep metadata and ZIPs.
$journal = Join-Path $report 'deletion_receipts.jsonl'
if (Test-Path -LiteralPath $journal) { throw 'A deletion journal already exists; review before resuming.' }
$removed = 0
foreach ($record in $verified) {
    $target = Resolve-ArchiveChild $record.directory $rawRoot
    $actualFiles = @(Get-ChildItem -LiteralPath $target -Recurse -Force -File)
    if ($actualFiles.Count -ne $record.files.Count) { throw "Directory changed after verification: $target" }
    foreach ($file in $record.files) {
        $item = Get-Item -LiteralPath $file.path -Force
        if ($item.Length -ne $file.bytes -or $item.LastWriteTimeUtc.Ticks -ne $file.ticks) {
            throw "Original changed after verification: $($file.path)"
        }
    }
    Remove-Item -LiteralPath $target -Recurse -Force
    if (Test-Path -LiteralPath $target) { throw "Removal incomplete: $target" }
    $removed++
    [ordered]@{at_utc=[DateTime]::UtcNow.ToString('o'); directory=$target; compressed=$record.zip;
        compressed_sha256=$record.zip_sha256; files=$record.files.Count} |
        ConvertTo-Json -Compress | Add-Content -LiteralPath $journal -Encoding utf8
    if ($removed % 50 -eq 0) { Write-Output "Removed $removed/910 verified duplicate folders" }
}
$result = [ordered]@{
    at_utc=[DateTime]::UtcNow.ToString('o'); removed_folders=$removed; removed_files=$originalFiles;
    removed_bytes=$originalBytes; kept_zip_count=$records.Count;
    kept_zip_bytes=($records | Measure-Object -Property compressed_bytes -Sum).Sum;
    free_bytes_before=$before; free_bytes_after=[IO.DriveInfo]::new('C:\').AvailableFreeSpace;
    live_simulator_calls=0; process_termination_calls=0; current_checkpoints_modified=0
}
Save-Json (Join-Path $report 'removal_result.json') $result
Save-Json (Join-Path $zipRoot 'raw_duplicate_removal_20261023.json') $result
$result | ConvertTo-Json
