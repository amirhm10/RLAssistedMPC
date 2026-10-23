[CmdletBinding()]
param([switch]$Apply)

# Author-approved, same-volume archival of the fixed candidate list. This script
# never deletes data, edits scientific sources, or signals a production process.
$ErrorActionPreference = 'Stop'
$taskHere = $PSScriptRoot
$taskRepo = [IO.Path]::GetFullPath((Join-Path $taskHere '..\..'))
$taskRepoPrefix = $taskRepo.TrimEnd('\') + '\'
$taskApprovedArchive = 'C:\Users\hamediaa\RL_MPC_local_archive\storage_20261023'
$taskArchive = [IO.Path]::GetFullPath($taskApprovedArchive)
$taskArchivePrefix = $taskArchive.TrimEnd('\') + '\'
$taskPlan = Get-Content -LiteralPath (Join-Path $taskHere 'archive_candidates.json') -Raw | ConvertFrom-Json -Depth 100
$taskAudit = Get-Content -LiteralPath (Join-Path $taskHere 'assessment.json') -Raw | ConvertFrom-Json -Depth 100
$taskReceipt = Join-Path $taskHere 'archive_receipts.jsonl'
$taskPreparedFile = Join-Path $taskHere 'archive_prepared.json'
$taskUtf8 = [Text.UTF8Encoding]::new($false)

function Assert-TaskPath([string]$Value, [string]$Prefix) {
    $taskFull = [IO.Path]::GetFullPath($Value)
    if (-not $taskFull.StartsWith($Prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path outside approved boundary: $taskFull"
    }
    return $taskFull
}

function Assert-TaskProcesses {
    foreach ($taskProperty in $taskAudit.protected_identities.PSObject.Properties) {
        $taskExpected = $taskProperty.Value
        if ($taskExpected.alive -ne $true -or $null -eq $taskExpected.created) {
            throw 'Protected production identity was not verified alive'
        }
        $taskCurrent = Get-Process -Id ([int]$taskExpected.pid) -ErrorAction Stop
        if ($taskCurrent.StartTime.ToFileTimeUtc() -ne [long]$taskExpected.created) {
            throw "Protected production identity changed: $($taskExpected.pid)"
        }
    }
}

function Assert-TaskUnreferenced([string]$Source) {
    $taskParent = [IO.Path]::GetDirectoryName($Source)
    $taskName = [IO.Path]::GetFileName($Source)
    if ($taskName -notmatch '^native_([01])_(\d+)$' -or [IO.Path]::GetFileName($taskParent) -ne 'training_checkpoints') {
        throw "Unexpected candidate directory: $Source"
    }
    $taskSlot = $taskName.Split('_')[1]
    $taskIndexPath = Join-Path $taskParent 'latest.json'
    if (-not (Test-Path -LiteralPath $taskIndexPath -PathType Leaf)) {
        throw "No current checkpoint index for $Source"
    }
    $taskIndex = Get-Content -LiteralPath $taskIndexPath -Raw
    if ($taskIndex.Contains($taskName)) { throw "Current checkpoint references $Source" }
    $taskPeers = @(Get-ChildItem -LiteralPath $taskParent -Directory | Where-Object {
        $_.Name -match ('^native_' + $taskSlot + '_\d+$')
    } | Sort-Object { [long]($_.Name.Split('_')[-1]) })
    $taskPreserve = @($taskPeers | Select-Object -First 3) + @($taskPeers | Select-Object -Last 4)
    if ($taskPreserve.Name -contains $taskName) { throw "Candidate is in the preserved generations: $Source" }
}

function Get-TaskExpectedFiles($Candidate) {
    $taskSource = Assert-TaskPath $Candidate.path $taskRepoPrefix
    $taskSource = (Resolve-Path -LiteralPath $taskSource).ProviderPath
    $taskSource = Assert-TaskPath $taskSource $taskRepoPrefix
    Assert-TaskUnreferenced $taskSource
    $taskFolder = Get-Item -LiteralPath $taskSource
    if ($taskFolder.LinkType) { throw "Link candidate rejected: $taskSource" }
    $taskRelative = $taskSource.Substring($taskRepoPrefix.Length)
    $taskDestination = Assert-TaskPath (Join-Path $taskArchive $taskRelative) $taskArchivePrefix
    if ([IO.Path]::GetPathRoot($taskSource) -ne [IO.Path]::GetPathRoot($taskDestination)) {
        throw 'The approved archival must stay on the same volume'
    }
    if (Test-Path -LiteralPath $taskDestination) { throw "Archive target already exists: $taskDestination" }
    $taskSourcePrefix = $taskSource.TrimEnd('\') + '\'
    $taskActual = @(Get-ChildItem -LiteralPath $taskSource -File -Recurse)
    if ($taskActual.Count -ne $Candidate.files.Count) { throw "Candidate file count changed: $taskSource" }
    $taskMetadataFiles = @($taskActual | Where-Object { $_.Name -eq 'metadata.json' })
    if ($taskMetadataFiles.Count -ne 1) { throw "Expected one export metadata file: $taskSource" }
    $taskMetadata = Get-Content -LiteralPath $taskMetadataFiles[0].FullName -Raw | ConvertFrom-Json
    $taskModel = Assert-TaskPath $taskMetadata.project_file $taskSourcePrefix
    $taskSnapshot = Assert-TaskPath $taskMetadata.file $taskSourcePrefix
    $taskFiles = @()
    foreach ($taskFile in $taskActual) {
        $taskFilePath = Assert-TaskPath $taskFile.FullName $taskSourcePrefix
        $taskInside = $taskFilePath.Substring($taskSourcePrefix.Length).Replace('\','/')
        $taskOriginal = @($Candidate.files | Where-Object { $_.relative -eq $taskInside })
        if ($taskOriginal.Count -ne 1 -or $taskFile.Length -ne [long]$taskOriginal[0].bytes) {
            throw "Candidate content inventory changed: $taskFilePath"
        }
        $taskWriteNs = ($taskFile.LastWriteTimeUtc.Ticks - [datetime]::UnixEpoch.Ticks) * 100
        if ([decimal]$taskWriteNs -ne [decimal]$taskOriginal[0].mtime_ns) { throw "Candidate write time changed: $taskFilePath" }
        if ($taskFile.LastWriteTimeUtc -gt [datetime]::UtcNow.AddHours(-1)) { throw "Candidate is too recent: $taskFilePath" }
        if (($taskFile.Attributes.value__ -band (0x1000 -bor 0x40000 -bor 0x400000)) -ne 0) {
            throw "Offline or recall file rejected: $taskFilePath"
        }
        if ($taskFilePath -eq $taskModel) {
            $taskExpectedHash = $taskMetadata.project_file_sha256
        } elseif ($taskFilePath -eq $taskSnapshot) {
            $taskExpectedHash = $taskMetadata.file_sha256
        } else {
            $taskExpectedHash = (Get-FileHash -LiteralPath $taskFilePath -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        if ($taskExpectedHash -notmatch '^[a-fA-F0-9]{64}$') { throw "Invalid expected file digest: $taskFilePath" }
        $taskFiles += [pscustomobject]@{relative=$taskInside; bytes=$taskFile.Length; write_ticks=$taskFile.LastWriteTimeUtc.Ticks; sha256=$taskExpectedHash.ToLowerInvariant()}
    }
    return [pscustomobject]@{source=$taskSource; destination=$taskDestination; bytes=[long]$Candidate.bytes; files=$taskFiles}
}

if ($taskPlan.proposed_destination -ne $taskApprovedArchive -or $taskPlan.applied -ne $false) {
    throw 'The source candidate plan differs from the author-approved proposal'
}
if ($taskArchive.StartsWith($taskRepoPrefix, [StringComparison]::OrdinalIgnoreCase) -or $taskArchive -match 'OneDrive') {
    throw 'Archive must be outside all OneDrive folders'
}
if (Test-Path -LiteralPath $taskReceipt) { throw 'An archive attempt is already recorded; inspect it before any retry' }
Assert-TaskProcesses
$taskPrepared = @()
foreach ($taskGroup in $taskPlan.groups) {
    foreach ($taskCandidate in $taskGroup.candidates) {
        $taskPrepared += Get-TaskExpectedFiles $taskCandidate
    }
    Assert-TaskProcesses
}
$taskPrepared | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $taskPreparedFile -Encoding utf8
$taskTotalBytes = [long](($taskPrepared | Measure-Object -Property bytes -Sum).Sum)
if ($taskPrepared.Count -ne $taskPlan.candidate_folders -or $taskTotalBytes -ne [long]$taskPlan.total_candidate_bytes) {
    throw 'Prepared archive differs from the approved candidate list'
}
Write-Output ([pscustomobject]@{mode=$(if ($Apply) {'apply'} else {'preflight_only'}); folders=$taskPrepared.Count; gib=$taskTotalBytes/1GB; protected_processes=@($taskAudit.protected_identities.PSObject.Properties).Count} | ConvertTo-Json -Compress)
if (-not $Apply) { return }

New-Item -ItemType Directory -Path $taskArchive -Force | Out-Null
if ((Get-Item -LiteralPath $taskArchive).LinkType) { throw 'Archive root must not be a link' }
Copy-Item -LiteralPath $taskPreparedFile -Destination (Join-Path $taskArchive 'restore_manifest.json') -ErrorAction Stop
[IO.File]::WriteAllText((Join-Path $taskArchive 'README.txt'), 'Historical native checkpoints archived with author approval. Relative paths are preserved. restore_manifest.json maps every original and archive path and records SHA-256 hashes. Do not move them back over an existing current checkpoint directory. All current Python checkpoints and recent native generations remain in the repository.', $taskUtf8)
$taskMoved = 0
$taskMovedBytes = [long]0
foreach ($taskEntry in $taskPrepared) {
    Assert-TaskProcesses
    $taskSource = Assert-TaskPath $taskEntry.source $taskRepoPrefix
    $taskDestination = Assert-TaskPath $taskEntry.destination $taskArchivePrefix
    Assert-TaskUnreferenced $taskSource
    if (Test-Path -LiteralPath $taskDestination) { throw "Archive destination appeared: $taskDestination" }
    foreach ($taskExpected in $taskEntry.files) {
        $taskFile = Get-Item -LiteralPath (Join-Path $taskSource $taskExpected.relative)
        if ($taskFile.Length -ne $taskExpected.bytes -or $taskFile.LastWriteTimeUtc.Ticks -ne $taskExpected.write_ticks) {
            throw "Source changed since preparation: $($taskFile.FullName)"
        }
    }
    $taskDestinationParent = [IO.Path]::GetDirectoryName($taskDestination)
    New-Item -ItemType Directory -Path $taskDestinationParent -Force | Out-Null
    $taskAttempt = [pscustomobject]@{state='prepared'; source=$taskSource; destination=$taskDestination; at_utc=[datetime]::UtcNow.ToString('o')}
    [IO.File]::AppendAllText($taskReceipt, (($taskAttempt | ConvertTo-Json -Compress) + "`n"), $taskUtf8)
    Move-Item -LiteralPath $taskSource -Destination $taskDestination -ErrorAction Stop
    try {
        if (Test-Path -LiteralPath $taskSource) { throw 'Source remains after the same-volume move' }
        $taskArchivedFiles = @(Get-ChildItem -LiteralPath $taskDestination -File -Recurse)
        if ($taskArchivedFiles.Count -ne $taskEntry.files.Count) { throw 'Archive file count differs' }
        foreach ($taskExpected in $taskEntry.files) {
            $taskArchivedPath = Assert-TaskPath (Join-Path $taskDestination $taskExpected.relative) ($taskDestination.TrimEnd('\')+'\')
            $taskArchivedFile = Get-Item -LiteralPath $taskArchivedPath
            $taskHash = (Get-FileHash -LiteralPath $taskArchivedPath -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($taskArchivedFile.Length -ne $taskExpected.bytes -or $taskHash -ne $taskExpected.sha256) {
                throw "Archived file failed verification: $taskArchivedPath"
            }
        }
    } catch {
        if ((Test-Path -LiteralPath $taskDestination) -and -not (Test-Path -LiteralPath $taskSource)) {
            Move-Item -LiteralPath $taskDestination -Destination $taskSource -ErrorAction Stop
        }
        [IO.File]::AppendAllText($taskReceipt, (([pscustomobject]@{state='verification_failed_current_folder_restored'; source=$taskSource; error=$_.Exception.Message} | ConvertTo-Json -Compress)+"`n"), $taskUtf8)
        throw
    }
    $taskMoved++
    $taskMovedBytes += $taskEntry.bytes
    [IO.File]::AppendAllText($taskReceipt, (([pscustomobject]@{state='verified'; source=$taskSource; destination=$taskDestination; bytes=$taskEntry.bytes; files=$taskEntry.files.Count; at_utc=[datetime]::UtcNow.ToString('o')} | ConvertTo-Json -Compress)+"`n"), $taskUtf8)
    if ($taskMoved % 25 -eq 0) { Write-Output ([pscustomobject]@{verified_folders=$taskMoved; total=$taskPrepared.Count; archived_gib=$taskMovedBytes/1GB} | ConvertTo-Json -Compress) }
}
Assert-TaskProcesses
$taskResult = [pscustomobject]@{state='completed'; verified_folders=$taskMoved; verified_bytes=$taskMovedBytes; destination=$taskArchive; protected_identities_unchanged=$true; at_utc=[datetime]::UtcNow.ToString('o')}
$taskResult | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskHere 'archive_result.json') -Encoding utf8
Copy-Item -LiteralPath $taskReceipt -Destination (Join-Path $taskArchive 'archive_receipts.jsonl')
$taskResult | ConvertTo-Json
