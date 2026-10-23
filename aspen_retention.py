"""Independent, bounded-memory retention for sealed Aspen experiment workers.

Only obsolete native checkpoint directories and closed private load copies are
eligible. Scientific result directories, both Python slots, source models and
the newest two complete native checkpoints are never removal targets.
"""
from collections import OrderedDict
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import uuid
import zipfile

from aspen_retention_pickle import metadata, InspectionError

NATIVE = re.compile(r'^native_([01])_(\d+)$')
LOAD = re.compile(r'^load_[0-9a-f]{8}$')
SESSION = re.compile(r'^native_[0-9a-f]{8}$')
SHA = re.compile(r'^[0-9a-f]{64}$')
METADATA_KEYS = ('file', 'file_sha256', 'project_file', 'project_file_sha256', 'name')
RESULT_EXTENSIONS = frozenset(('.npz', '.npy', '.pkl', '.png', '.pdf', '.csv', '.tsv', '.pt', '.pth'))
ACTIVE_NAMES = frozenset(('aspenamlock.aml', 'am_sai.appdf', 'plotdata.cpd'))
_ATTRIBUTES_API = None


class Deferred(RuntimeError):
    """A live or ambiguous generation must be left intact and retried later."""


def now():
    return datetime.now(timezone.utc).isoformat()


def attributes(path):
    """Query Win32 attributes without opening a replaceable file handle."""
    global _ATTRIBUTES_API
    path = Path(path)
    if os.name != 'nt':
        value = path.lstat()
        return dict(flags=0x400 if path.is_symlink() else 0, creation=value.st_ctime_ns,
                    size=value.st_size, mtime=value.st_mtime_ns, volume=value.st_dev)
    import ctypes
    from ctypes import wintypes
    if _ATTRIBUTES_API is None:
        class Data(ctypes.Structure):
            _fields_ = [('flags', wintypes.DWORD), ('created', wintypes.FILETIME),
                        ('accessed', wintypes.FILETIME), ('written', wintypes.FILETIME),
                        ('size_high', wintypes.DWORD), ('size_low', wintypes.DWORD)]
        function = ctypes.WinDLL('kernel32', use_last_error=True).GetFileAttributesExW
        function.argtypes = [wintypes.LPCWSTR, ctypes.c_int, ctypes.POINTER(Data)]
        function.restype = wintypes.BOOL
        _ATTRIBUTES_API = function, Data
    function, Data = _ATTRIBUTES_API
    value = Data()
    absolute = str(path.absolute())
    extended = ('\\\\?\\UNC\\' + absolute[2:]) if absolute.startswith('\\\\') else ('\\\\?\\' + absolute)
    if not function(extended, 0, ctypes.byref(value)):
        raise ctypes.WinError(ctypes.get_last_error())
    creation = (value.created.dwHighDateTime << 32) | value.created.dwLowDateTime
    written = (value.written.dwHighDateTime << 32) | value.written.dwLowDateTime
    return dict(flags=value.flags, creation=creation,
                size=(value.size_high << 32) | value.size_low,
                mtime=(written - 116444736000000000) * 100, volume=path.anchor.upper())


def version(path):
    value = attributes(path)
    return (value['volume'], value['creation'], value['size'], value['mtime'])


@contextmanager
def open_read(path, *, exclusive=False):
    """Open immutable/private files; live replaceable paths use a snapshot alias."""
    if os.name != 'nt':
        with Path(path).open('rb') as stream:
            yield stream
        return
    import ctypes
    from ctypes import wintypes
    import msvcrt
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    absolute = str(Path(path).absolute())
    extended = ('\\\\?\\UNC\\' + absolute[2:]) if absolute.startswith('\\\\') else ('\\\\?\\' + absolute)
    handle = kernel.CreateFileW(extended, 0x80000000,
                                0 if exclusive else 7, None, 3, 0x08000000, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle(handle)
        raise
    with os.fdopen(descriptor, 'rb') as stream:
        yield stream


@contextmanager
def private_snapshot(path, snapshots=None):
    """Read a hard-link alias so a writer can atomically replace the original.

    Windows/OneDrive replacement can reject even a shared-delete handle opened
    by the original filename. A second directory entry preserves the old file
    while allowing the original entry to be replaced. Never fall back to
    opening an active slot if a safe same-volume snapshot cannot be created.
    """
    directory = Path(snapshots) if snapshots is not None else Path(__file__).parent / '.retention_read_snapshots'
    directory.mkdir(parents=True, exist_ok=True)
    alias = directory / ('.read_' + uuid.uuid4().hex + '.link')
    os.link(path, alias)
    try:
        with open_read(alias) as stream:
            yield stream
    finally:
        safe_child(alias, directory)
        alias.unlink()


def read_json(path, *, snapshots=None):
    with private_snapshot(path, snapshots) as stream:
        return json.loads(stream.read().decode('utf-8-sig'))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def digest_stream(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(block)
    return result.hexdigest()


def digest(path):
    with open_read(path) as stream:
        return digest_stream(stream)


def safe_child(path, root):
    """Require a real descendant, with no reparse points at any path component."""
    path, root = Path(os.path.abspath(path)), Path(os.path.abspath(root))
    if path == root or not path.is_relative_to(root):
        raise Deferred('Path is outside its approved root: ' + str(path))
    for item in (path, *path.parents):
        if attributes(item)['flags'] & 0x400:
            raise Deferred('Reparse point is excluded: ' + str(item))
        if item == root:
            break
    return path


def walk_error(error):
    raise error


def inventory(folder, root, *, allow_empty=False):
    folder = safe_child(folder, root)
    files = {}
    for parent, dirs, names in os.walk(folder, followlinks=False, onerror=walk_error):
        for name in [*dirs, *names]:
            path = safe_child(Path(parent) / name, root)
            if path.is_file():
                if path.suffix.lower() in RESULT_EXTENSIONS:
                    raise Deferred('Scientific result file excludes a retention target: ' + str(path))
                if path.name.lower() in ACTIVE_NAMES or path.suffix.lower() in ('.tmp', '.aml', '.appdf'):
                    raise Deferred('An Aspen working-document marker remains: ' + str(path))
                files[path.relative_to(folder).as_posix()] = version(path)
    if not files and not allow_empty:
        raise Deferred('Empty or unfinished native directory')
    return files


class Verifier:
    def __init__(self, state, *, quiet_seconds=0):
        self.state = state
        self.snapshots = state / 'read_snapshots'
        self.cache = OrderedDict()
        self.pairs = {}
        self.quiet_seconds = quiet_seconds
        self.observations = {}
        self.anchors = read_json(state / 'checkpoint_hashes.json') if (state / 'checkpoint_hashes.json').exists() else {}

    def file(self, path, *, inspect_pickle=False, alias=None):
        stamp = version(path)
        if alias is not None and version(alias) != stamp:
            raise Deferred('The snapshotted slot was replaced before its verification')
        key = (str(path), stamp, inspect_pickle)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        context = open_read(alias) if alias is not None else private_snapshot(path, self.snapshots)
        with context as stream:
            before = os.fstat(stream.fileno())
            checksum = digest_stream(stream)
            parsed = None
            if inspect_pickle:
                stream.seek(0)
                parsed = metadata(stream)
            after = os.fstat(stream.fileno())
        if version(path) != stamp or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise Deferred('Checkpoint changed during its read')
        result = dict(sha256=checksum, metadata=parsed, version=stamp)
        self.cache[key] = result
        while len(self.cache) > 128:
            self.cache.popitem(last=False)
        return result

    def pair(self, directory, results_root):
        directory = safe_child(directory, results_root)
        layout = quiet_checkpoint_layout(directory)
        cached = self.pairs.get(str(directory))
        if cached and cached['layout']['native_entries'] == layout['native_entries'] and same_generation(cached['layout'], layout) and still_current(cached):
            return cached
        key = str(directory)
        prior = self.observations.get(key)
        if prior is None or prior[0] != layout:
            self.observations[key] = (layout, time.monotonic())
            if self.quiet_seconds:
                raise Deferred('Waiting for a stable committed checkpoint before snapshot acquisition')
        elif time.monotonic() - prior[1] < self.quiet_seconds:
            raise Deferred('Checkpoint observation interval is incomplete')
        # Snapshot all three entries quickly, before inspecting either large
        # Python state. A new native directory/tmp file signals a writer well
        # before its atomic slot/index replacement; defer while that is active.
        with ExitStack() as stack:
            aliases = {}
            self.snapshots.mkdir(parents=True, exist_ok=True)
            for name in ('latest.json', 'training_state_0.pkl', 'training_state_1.pkl'):
                if quiet_checkpoint_layout(directory) != layout:
                    raise Deferred('A checkpoint writer started during snapshot acquisition')
                if attributes(directory)['mtime'] != layout['directory_mtime']:
                    raise Deferred('A checkpoint directory entry changed before snapshot acquisition')
                alias = self.snapshots / ('.read_' + uuid.uuid4().hex + '.link')
                os.link(directory / name, alias)
                aliases[name] = alias
                stack.callback(alias.unlink)
            if quiet_checkpoint_layout(directory) != layout:
                raise Deferred('Checkpoint generation changed during snapshot acquisition')
            with open_read(aliases['latest.json']) as stream:
                index = json.loads(stream.read().decode('utf-8-sig'))
            pair = self._pair(directory, results_root, index, aliases, layout)
        self.pairs[str(directory)] = pair
        return pair

    def _pair(self, directory, results_root, index, aliases, layout):
        if index.get('schema') != 2 or index.get('slot') not in (0, 1):
            raise Deferred('Unsupported native checkpoint index')
        expected_file = f'training_state_{index["slot"]}.pkl'
        if index.get('file') != expected_file:
            raise Deferred('Unexpected checkpoint slot filename')
        records, components = [], []
        for slot in (0, 1):
            path = safe_child(directory / f'training_state_{slot}.pkl', directory)
            checked = self.file(path, inspect_pickle=True, alias=aliases[path.name])
            payload = checked['metadata']
            step = payload['completed_steps']
            native = payload['native']
            if payload.get('source_hashes') != index['source_hashes']:
                raise Deferred('Checkpoint slots have different source provenance')
            if not isinstance(payload.get('runner_sha256'), str) or not SHA.fullmatch(payload['runner_sha256']):
                raise Deferred('Checkpoint runner digest is absent')
            native_dir = safe_child(directory / f'native_{slot}_{step}', directory)
            for key in ('file', 'project_file'):
                component = safe_child(native[key], native_dir)
                verified = self.file(component)
                if not SHA.fullmatch(native[key + '_sha256']) or verified['sha256'] != native[key + '_sha256']:
                    raise Deferred('Native checkpoint component checksum differs')
                components.append((str(component), verified['version']))
            native_meta = Path(native['file']).parent / 'metadata.json'
            meta_version = version(native_meta)
            stored = read_json(native_meta)
            if version(native_meta) != meta_version:
                raise Deferred('Native metadata changed during verification')
            components.append((str(native_meta), meta_version))
            if any(stored.get(key) != native.get(key) for key in METADATA_KEYS):
                raise Deferred('Python/native metadata linkage differs')
            anchor_key = str(directory) + f'::{slot}::{step}'
            old_hash = self.anchors.get(anchor_key)
            if old_hash and old_hash != checked['sha256']:
                raise Deferred('Previously verified checkpoint contents changed')
            if slot == index['slot']:
                if step != index['completed_steps'] or checked['sha256'] != index['sha256']:
                    raise Deferred('Latest Python checkpoint checksum differs')
                if any(native.get(key) != index['native'].get(key) for key in METADATA_KEYS):
                    raise Deferred('Latest index/native linkage differs')
            records.append(dict(slot=slot, completed_steps=step, python_file=str(path),
                                sha256=checked['sha256'], native_directory=str(native_dir), native=native,
                                runner_sha256=payload['runner_sha256'], version=checked['version'],
                                private_roots=payload.get('retention_private_roots', []),
                                verification='writer_checksum' if slot == index['slot'] else
                                'prior_verified_checksum' if old_hash else 'legacy_structure_native_hashes_and_new_checksum'))
        ordered = sorted(records, key=lambda r: r['completed_steps'], reverse=True)
        if ordered[0]['slot'] != index['slot'] or ordered[0]['completed_steps'] == ordered[1]['completed_steps']:
            raise Deferred('An uncommitted checkpoint generation is present')
        if ordered[0]['runner_sha256'] != ordered[1]['runner_sha256']:
            raise Deferred('Checkpoint runner generations differ')
        if (quiet_checkpoint_layout(directory) != layout or
                any(version(Path(r['python_file'])) != r['version'] for r in records) or
                any(version(Path(name)) != stamp for name, stamp in components)):
            raise Deferred('Checkpoint generation changed during validation')
        for record in records:
            self.anchors[str(directory) + f'::{record["slot"]}::{record["completed_steps"]}'] = record['sha256']
        atomic_json(self.state / 'checkpoint_hashes.json', self.anchors)
        return dict(directory=str(directory), index=index, checkpoints=ordered, layout=layout,
                    token=[(r['python_file'], r['version']) for r in ordered], components=components)


def quiet_checkpoint_layout(directory):
    """Use attributes only to refuse an in-flight native/Python/index commit.

    The sealed writer creates its native directory and copies/flushed models
    before writing either slot or index. Existing native directories also gain
    new child entries before a repeated step is saved. No live file is opened.
    """
    directory_mtime = attributes(directory)['mtime']
    latest = version(directory / 'latest.json')
    slots = [version(directory / f'training_state_{slot}.pkl') for slot in (0, 1)]
    if any(stamp[3] > latest[3] for stamp in slots):
        raise Deferred('Python slot committed before its latest index; wait for the writer')
    native_entries = []
    for path in directory.iterdir():
        if path.name.startswith('training_state_') and path.name.endswith('.tmp'):
            raise Deferred('A Python checkpoint temporary write is active')
        if NATIVE.fullmatch(path.name) and path.is_dir():
            safe_child(path, directory)
            changed = attributes(path)['mtime']
            child_names = []
            for child in path.iterdir():
                if child.is_dir():
                    safe_child(child, directory)
                    changed = max(changed, attributes(child)['mtime'])
                    child_names.append(child.name)
            native_entries.append((path.name, tuple(sorted(child_names))))
            if changed > latest[3]:
                raise Deferred('A native checkpoint is being saved/restored; wait for the writer')
    if attributes(directory)['mtime'] != directory_mtime or version(directory / 'latest.json') != latest:
        raise Deferred('Checkpoint directory/index changed during its attribute scan')
    return dict(index=latest, slots=slots, directory_mtime=directory_mtime,
                native_entries=tuple(sorted(native_entries)))


def same_generation(first, second):
    return first['index'] == second['index'] and first['slots'] == second['slots']


def still_current(pair):
    try:
        directory = Path(pair['directory'])
        return (version(directory / 'latest.json') == pair['layout']['index'] and
                attributes(directory)['mtime'] == pair['layout']['directory_mtime'] and
                all(version(Path(name)) == tuple(stamp) for name, stamp in pair['token']) and
                all(version(Path(name)) == tuple(stamp) for name, stamp in pair['components']))
    except (OSError, Deferred):
        return False


def native_candidates(pair):
    minimum = min(r['completed_steps'] for r in pair['checkpoints'])
    protected = {Path(r['native_directory']) for r in pair['checkpoints']}
    return sorted([p for p in Path(pair['directory']).iterdir()
                   if NATIVE.fullmatch(p.name) and p.is_dir() and p not in protected
                   and int(NATIVE.fullmatch(p.name)[2]) < minimum],
                  key=lambda p: int(NATIVE.fullmatch(p.name)[2]))


def load_candidates(job, *, sessions=None):
    """Each loader always creates a fresh directory; retain the newest two per role.

    Active markers, locks, recent writes and exact file identity are checked
    separately before any candidate can be archived or retired.
    """
    candidates = []
    # Runtime callers supply known sessions from saved metadata. Do not
    # enumerate a job root that also contains the worker's status outputs.
    for session in (job.iterdir() if sessions is None else sessions):
        if not SESSION.fullmatch(session.name) or not session.is_dir():
            continue
        for role in ('main', 'branch'):
            parent = session / role
            if not parent.is_dir():
                continue
            folders = [p for p in parent.iterdir() if LOAD.fullmatch(p.name) and p.is_dir()]
            folders.sort(key=lambda p: (p.stat().st_ctime_ns, p.name), reverse=True)
            candidates.extend(folders[2:])
    return candidates


class Limiter:
    def __init__(self, bytes_per_second):
        self.rate = bytes_per_second
        self.credit = 2 * 1024 * 1024
        self.last = time.monotonic()

    def consume(self, size):
        current = time.monotonic()
        self.credit = min(2 * 1024 * 1024, self.credit + (current - self.last) * self.rate) - size
        if self.credit < 0:
            time.sleep(-self.credit / self.rate)
            self.credit = 0
        self.last = time.monotonic()


def native_paths_in(value, results_root):
    """Extract recovery dependencies from an already-frozen metadata object.

    The maintenance loop deliberately never opens live queue manifests.
    """
    found = set()

    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            path = Path(value)
            if path.is_absolute() and path.is_relative_to(results_root):
                for parent in (path, *path.parents):
                    if NATIVE.fullmatch(parent.name) and parent.parent.name == 'training_checkpoints':
                        found.add(parent)
                        break
    visit(value)
    return found


class Retainer:
    def __init__(self, config, *, clock=time.time):
        self.config = config
        self.results = Path(config['results_root']).resolve()
        self.state = Path(config['state_root']).resolve()
        if self.state == self.results or self.state.is_relative_to(self.results):
            raise ValueError('Retention archives must be outside experiment results')
        self.jobs = [safe_child(Path(row['directory']), self.results) for row in config['jobs']]
        self.labels = {str(Path(row['directory'])): row['label'] for row in config['jobs']}
        self.sessions = {str(Path(row['directory'])): {Path(path) for path in row.get('private_sessions', [])}
                         for row in config['jobs']}
        self.state.mkdir(parents=True, exist_ok=True)
        for name in ('archives', 'receipts'):
            (self.state / name).mkdir(exist_ok=True)
        self.verifier = Verifier(self.state, quiet_seconds=config.get('checkpoint_quiet_seconds', 2))
        self.clock = clock
        self.seen = {}
        self.limiter = Limiter(config.get('io_mib_per_second', 128) * 1024 * 1024)
        self.pins = {Path(path) for path in config.get('pinned_native_directories', [])}
        self.totals = dict(removed_folders=0, original_bytes=0, compressed_bytes=0)
        self.job_status = []
        self.active = None
        self.errors = []
        self.job_rotation = 0
        self.load_guards = {}
        for path in (self.state / 'receipts').glob('*.json'):
            value = read_json(path)
            if value.get('state') == 'raw_removed':
                self._count(value)

    def _count(self, record):
        self.totals['removed_folders'] += 1
        self.totals['original_bytes'] += record['original_bytes']
        self.totals['compressed_bytes'] += record['compressed_bytes']

    def status(self, state='running'):
        import shutil
        result = dict(state=state, updated_at=now(), pid=os.getpid(),
                      protected_complete_checkpoints_per_job=2, all_scientific_results_preserved=True,
                      scientific_sources_modified=0, live_simulator_calls=0, process_termination_calls=0,
                      totals=self.totals, jobs=self.job_status, active=self.active, issues=self.errors[-20:],
                      free_disk_bytes=shutil.disk_usage(self.state).free,
                      stop_file=str(self.state / 'stop.request'))
        atomic_json(self.state / 'status.json', result)
        lines = ['# Automatic Aspen retention', '', f'Updated: {result["updated_at"]}', '',
                 f'State: {state}. Worker PID: {os.getpid()}.', '',
                 f'Archived and reclaimed: {self.totals["removed_folders"]} folders; '
                 f'{self.totals["original_bytes"] / 1024**3:.2f} GiB raw -> '
                 f'{self.totals["compressed_bytes"] / 1024**3:.2f} GiB verified ZIPs.', '',
                 f'C: free space: {result["free_disk_bytes"] / 1024**3:.1f} GiB.', '',
                 'The newest two complete Python/native checkpoints, all result folders and live Aspen documents are protected.', '']
        for row in self.job_status:
            steps = ', '.join(str(x['completed_steps']) for x in row.get('checkpoints', []))
            lines.append(f'- {row["label"]}: {row["state"]}; verified steps {steps or "pending"}; '
                         f'obsolete native folders {row.get("obsolete_native", "unknown")}; '
                         f'old private-copy candidates {row.get("old_private", "unknown")}.')
        if self.active:
            lines.extend(['', 'Current operation: ' + self.active['phase'] + ' — ' + self.active['directory']])
        if self.errors:
            lines.extend(['', 'Recent deferred checks:', *['- ' + e['message'] for e in self.errors[-5:]]])
        temporary = self.state / 'STATUS.md.tmp'
        temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        os.replace(temporary, self.state / 'STATUS.md')
        return result

    def _eligible(self, folder, kind, job, pair):
        safe_child(folder, self.results)
        if not still_current(pair):
            raise Deferred('A new checkpoint committed; revalidate before cleanup')
        if kind == 'native':
            match = NATIVE.fullmatch(folder.name)
            minimum = min(row['completed_steps'] for row in pair['checkpoints'])
            if (not match or folder.parent != Path(pair['directory']) or int(match[2]) >= minimum or
                    folder in {Path(row['native_directory']) for row in pair['checkpoints']}):
                raise Deferred('Checkpoint is one of the protected/current generations')
            if folder in self.pins:
                raise Deferred('Checkpoint is referenced by a queue/recovery manifest')
        elif kind == 'private':
            parent = folder.parent
            if (not LOAD.fullmatch(folder.name) or parent.name not in ('main', 'branch') or
                    not SESSION.fullmatch(parent.parent.name) or parent.parent.parent != job):
                raise Deferred('Working directory is outside a known private session role')
            stamp = parent.stat().st_mtime_ns
            cached = self.load_guards.get(str(parent))
            if cached is None or cached[0] != stamp:
                copies = [p for p in parent.iterdir() if LOAD.fullmatch(p.name) and p.is_dir()]
                copies.sort(key=lambda p: (p.stat().st_ctime_ns, p.name), reverse=True)
                if parent.stat().st_mtime_ns != stamp:
                    raise Deferred('A new private document appeared during the role scan')
                cached = (stamp, {p.name for p in copies[:2]})
                self.load_guards[str(parent)] = cached
            if folder.name in cached[1]:
                raise Deferred('Working directory is one of the newest two copies')
        else:
            raise Deferred('Unrecognized retention target kind')
        files = inventory(folder, self.results)
        if not any(Path(name).suffix.lower() == '.dynf' for name in files):
            raise Deferred('A private Aspen project is absent')
        age = self.clock() - max(folder.stat().st_ctime, *(stamp[3] / 1e9 for stamp in files.values()))
        if age < self.config.get('minimum_age_seconds', 120):
            raise Deferred('Directory is too recent')
        return files

    def _archive(self, folder, files, kind, pair):
        fingerprint = hashlib.sha256((str(folder) + json.dumps(files, sort_keys=True)).encode()).hexdigest()[:24]
        receipt_path = self.state / 'receipts' / (fingerprint + '.json')
        if receipt_path.exists():
            record = read_json(receipt_path)
            if record['state'] == 'raw_removed':
                raise Deferred('A previously retired source path was recreated; inspect it manually')
            if digest(Path(record['compressed'])) != record['compressed_sha256']:
                raise Deferred('An existing retained ZIP checksum differs')
            for file in record['files']:
                if digest(folder / file['relative']) != file['sha256']:
                    raise Deferred('Source differs from its completed ZIP receipt')
            return receipt_path, record

        target = self.state / 'archives' / (fingerprint + '_' + folder.name + '.zip')
        temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.part')
        if target.exists():
            raise Deferred('An orphan archive needs inspection; no source removed')
        manifest = []
        try:
            # Exclusive reads are acquired only for already-obsolete candidates.
            # An existing Aspen/other handle fails this probe without changing it.
            with ExitStack() as stack:
                streams = {name: stack.enter_context(open_read(folder / name, exclusive=True)) for name in files}
                if inventory(folder, self.results) != files:
                    raise Deferred('Source changed before archival')
                with temporary.open('xb') as output:
                    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED,
                                         compresslevel=6, allowZip64=True) as archive:
                        for name, source in streams.items():
                            check, size = hashlib.sha256(), 0
                            with archive.open(name, 'w', force_zip64=True) as saved:
                                for block in iter(lambda: source.read(1024 * 1024), b''):
                                    self.limiter.consume(len(block))
                                    check.update(block)
                                    size += len(block)
                                    saved.write(block)
                            if size != files[name][2]:
                                raise Deferred('Source length changed during archival')
                            manifest.append(dict(relative=name, bytes=size, sha256=check.hexdigest(), version=files[name]))
                    output.flush()
                    os.fsync(output.fileno())
                self.active['phase'] = 'verifying every decompressed file'
                self.status()
                with zipfile.ZipFile(temporary) as archive:
                    if len(archive.infolist()) != len(files) or set(archive.namelist()) != set(files):
                        raise Deferred('Compressed file mapping differs')
                    for file in manifest:
                        check = hashlib.sha256()
                        with archive.open(file['relative']) as source:
                            for block in iter(lambda: source.read(1024 * 1024), b''):
                                self.limiter.consume(len(block))
                                check.update(block)
                        if check.hexdigest() != file['sha256']:
                            raise Deferred('Decompressed file checksum differs')
                if inventory(folder, self.results) != files:
                    raise Deferred('Source changed while its ZIP was verified')
            os.replace(temporary, target)
            record = dict(state='archive_verified', at_utc=now(), original_directory=str(folder),
                          kind=kind, compressed=str(target), compressed_sha256=digest(target),
                          compressed_bytes=target.stat().st_size, files=manifest,
                          original_bytes=sum(f['bytes'] for f in manifest),
                          protected_checkpoints=pair['checkpoints'], restored_hashes_verified=True)
            atomic_json(receipt_path, record)
            return receipt_path, record
        finally:
            if temporary.exists():
                safe_child(temporary, self.state)
                temporary.unlink()

    def _remove_retired(self, receipt_path, record):
        retired = safe_child(Path(record['retired_directory']), self.results)
        if not re.fullmatch(r'^\.aspen_retired_[0-9a-f]{32}$', retired.name):
            raise Deferred('Unexpected retirement staging directory')
        if digest(Path(record['compressed'])) != record['compressed_sha256']:
            raise Deferred('Retained archive checksum differs before source removal')
        expected = {f['relative']: f for f in record['files']}
        remaining = inventory(retired, self.results, allow_empty=True)
        if not set(remaining).issubset(expected):
            raise Deferred('Unexpected new file in retired directory')
        for name in remaining:
            if digest(retired / name) != expected[name]['sha256']:
                raise Deferred('Retired source checksum differs')
        # Remove only enumerated, hash-verified files inside the resolved owned
        # staging directory. No recursive shell command or results-root removal.
        for name in remaining:
            path = safe_child(retired / name, retired)
            if os.name == 'nt' and getattr(path.stat(), 'st_file_attributes', 0) & 1:
                path.chmod(stat.S_IWRITE)
            path.unlink()
        for parent, dirs, names in os.walk(retired, topdown=False, onerror=walk_error):
            if names:
                # os.walk reflects the post-unlink tree; any new file blocks it.
                raise Deferred('New file appeared during retired-directory removal')
            path = safe_child(Path(parent), self.results)
            # OneDrive native-copy directories carry READONLY (attributes 49).
            # Clear it only on this owned, archived and verified retirement
            # tree; current checkpoints/working documents are never changed.
            if os.name == 'nt' and getattr(path.stat(), 'st_file_attributes', 0) & 1:
                path.chmod(stat.S_IWRITE)
            path.rmdir()
        if retired.exists():
            raise Deferred('Retirement directory was not completely removed')
        record.update(state='raw_removed', removed_at_utc=now())
        atomic_json(receipt_path, record)
        self._count(record)

    def retire(self, folder, kind, job, pair):
        files = self._eligible(folder, kind, job, pair)
        key = str(folder)
        stamp = json.dumps(files, sort_keys=True)
        observed = self.seen.get(key)
        if not observed or observed[0] != stamp:
            self.seen[key] = (stamp, self.clock())
            raise Deferred('Waiting for a second unchanged directory observation')
        if self.clock() - observed[1] < self.config.get('stable_seconds', 20):
            raise Deferred('Directory observation interval is incomplete')
        self.active = dict(directory=str(folder), phase='lossless compression')
        self.status()
        receipt_path, record = self._archive(folder, files, kind, pair)
        if self._eligible(folder, kind, job, pair) != files:
            raise Deferred('Directory changed after archive verification')
        # Reprobe each obsolete file; a late reader/writer blocks retirement.
        with ExitStack() as stack:
            for name in files:
                stack.enter_context(open_read(folder / name, exclusive=True))
        if not still_current(pair) or inventory(folder, self.results) != files:
            raise Deferred('Generation changed before retirement')
        if digest(Path(record['compressed'])) != record['compressed_sha256']:
            raise Deferred('Retained ZIP changed before retirement')
        retired = folder.with_name('.aspen_retired_' + uuid.uuid4().hex)
        safe_child(folder, self.results)
        if retired.parent.resolve() != folder.parent.resolve() or not retired.is_relative_to(self.results):
            raise Deferred('Retirement destination escapes its approved parent')
        record['retired_directory'] = str(retired)
        record['state'] = 'rename_planned'
        atomic_json(receipt_path, record)
        os.rename(folder, retired)
        record['state'] = 'raw_renamed'
        atomic_json(receipt_path, record)
        self._remove_retired(receipt_path, record)
        # This worker's own rename changes the checkpoint-directory epoch.
        # Refresh that epoch only after another attributes-only writer check.
        if kind == 'native':
            try:
                current_layout = quiet_checkpoint_layout(Path(pair['directory']))
                if same_generation(current_layout, pair['layout']):
                    pair['layout'] = current_layout
            except (OSError, Deferred):
                pass
        self.seen.pop(key, None)
        self.active = None
        self.status()
        return record

    def recover_pending(self):
        for path in (self.state / 'receipts').glob('*.json'):
            record = read_json(path)
            if record.get('state') in ('raw_renamed', 'rename_planned'):
                try:
                    retired = Path(record['retired_directory'])
                    if retired.exists():
                        self._remove_retired(path, record)
                    elif not Path(record['original_directory']).exists():
                        if digest(Path(record['compressed'])) != record['compressed_sha256']:
                            raise Deferred('Missing retired source has an invalid archive')
                        record.update(state='raw_removed', removed_at_utc=now(), recovered_after_restart=True)
                        atomic_json(path, record)
                        self._count(record)
                except (OSError, ValueError, KeyError, Deferred) as error:
                    self.errors.append(dict(at_utc=now(), directory=record['retired_directory'], message=str(error)))

    def scan(self, *, execute=False):
        if execute:
            self.recover_pending()
        self.job_status = []
        choices = []
        for job in self.jobs:
            row = dict(label=self.labels[str(job)], directory=str(job), state='pending verification')
            try:
                pair = self.verifier.pair(job / 'training_checkpoints', self.results)
                native = native_candidates(pair)
                sessions = self.sessions[str(job)]
                for checkpoint in pair['checkpoints']:
                    for value in checkpoint.get('private_roots', []):
                        parent = Path(value)
                        session = parent.parent
                        if parent.name in ('main', 'branch') and SESSION.fullmatch(session.name) and session.parent == job:
                            sessions.add(session)
                private = load_candidates(job, sessions=sessions)
                row.update(state='two complete checkpoints verified', checkpoints=pair['checkpoints'],
                           obsolete_native=len(native), old_private=len(private))
                # Alternate kinds within each job, then jobs, so one backlog
                # cannot starve private-copy cleanup or another experiment.
                mixed = []
                for i in range(max(len(native), len(private))):
                    if i < len(native): mixed.append((native[i], 'native', job, pair))
                    if i < len(private): mixed.append((private[i], 'private', job, pair))
                choices.append(mixed)
            except (OSError, ValueError, KeyError, IndexError, Deferred) as error:
                row.update(state='deferred; all files retained', reason=str(error))
            self.job_status.append(row)
        candidates = []
        depth = max((len(items) for items in choices), default=0)
        if choices:
            offset = self.job_rotation % len(choices)
            choices = choices[offset:] + choices[:offset]
            self.job_rotation += 1
        for i in range(depth):
            candidates.extend(items[i] for items in choices if i < len(items))
        count = 0
        for folder, kind, job, pair in candidates:
            if execute and (self.state / 'stop.request').exists():
                break
            try:
                if not execute:
                    self._eligible(folder, kind, job, pair)
                    continue
                self.retire(folder, kind, job, pair)
                count += 1
                if count >= self.config.get('maximum_folders_per_pass', 12):
                    break
            except (OSError, ValueError, KeyError, IndexError, Deferred) as error:
                if 'second unchanged' not in str(error) and 'observation interval' not in str(error):
                    self.errors.append(dict(at_utc=now(), directory=str(folder), message=str(error)))
            finally:
                self.active = None
        self.errors = self.errors[-100:]
        return self.status('running' if execute else 'verified_dry_run')


def restore(receipt_path, destination):
    """Restore one retained directory into a new stage, never over live data."""
    record = read_json(receipt_path)
    source = Path(record['compressed'])
    if digest(source) != record['compressed_sha256']:
        raise ValueError('Retained ZIP hash differs')
    destination = Path(destination).absolute()
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(source) as archive:
        expected = {f['relative']: f for f in record['files']}
        if len(archive.infolist()) != len(expected) or set(archive.namelist()) != set(expected):
            raise ValueError('Archive file map differs')
        for name, info in expected.items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe archive member')
            target = destination / relative
            if not target.absolute().is_relative_to(destination):
                raise ValueError('Archive member escapes the staging directory')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as saved, target.open('xb') as output:
                for block in iter(lambda: saved.read(1024 * 1024), b''):
                    output.write(block)
            if target.stat().st_size != info['bytes'] or digest(target) != info['sha256']:
                raise ValueError('Extracted file hash differs')
    result = dict(verified=True, at_utc=now(), source_receipt=str(receipt_path),
                  original_directory=record['original_directory'], staged_directory=str(destination),
                  files=len(expected), simulator_calls=0, existing_files_replaced=0)
    atomic_json(destination / 'restoration_receipt.json', result)
    return result
