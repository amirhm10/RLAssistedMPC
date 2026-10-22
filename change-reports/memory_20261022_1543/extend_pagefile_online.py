"""One-shot, runtime-only paging capacity increase; never signals another process.

Run with the base interpreter's -S -B options. The default is a read-only plan.
An administrator must approve Windows UAC before --apply can enable the specific
SeCreatePagefilePrivilege. No registry setting, service or application is changed.
"""
import argparse
import ctypes as c
from ctypes import wintypes as w
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import winreg

HERE = Path(__file__).resolve().parent
GIB = 1024 ** 3
MINIMUM = 96 * GIB
MAXIMUM = 256 * GIB
DISK_RESERVE = 64 * GIB
KERNEL = c.WinDLL("kernel32", use_last_error=True)
PSAPI = c.WinDLL("psapi", use_last_error=True)


class Performance(c.Structure):
    _fields_ = [("cb", w.DWORD)] + [(name, c.c_size_t) for name in (
        "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal",
        "PhysicalAvailable", "SystemCache", "KernelTotal", "KernelPaged",
        "KernelNonpaged", "PageSize")] + [(name, w.DWORD) for name in (
        "HandleCount", "ProcessCount", "ThreadCount")]


class PageFile(c.Structure):
    _fields_ = [("cb", w.DWORD), ("Reserved", w.DWORD),
                ("TotalSize", c.c_size_t), ("TotalInUse", c.c_size_t),
                ("PeakUsage", c.c_size_t)]


class UnicodeString(c.Structure):
    _fields_ = [("Length", w.USHORT), ("MaximumLength", w.USHORT),
                ("Buffer", w.LPWSTR)]


class Luid(c.Structure):
    _fields_ = [("LowPart", w.DWORD), ("HighPart", w.LONG)]


class TokenPrivileges(c.Structure):
    _fields_ = [("PrivilegeCount", w.DWORD), ("Luid", Luid),
                ("Attributes", w.DWORD)]


def save(name, value):
    with (HERE / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def process_identity(pid):
    KERNEL.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    KERNEL.OpenProcess.restype = w.HANDLE
    KERNEL.GetProcessTimes.argtypes = [w.HANDLE] + [c.POINTER(w.FILETIME)] * 4
    KERNEL.GetExitCodeProcess.argtypes = [w.HANDLE, c.POINTER(w.DWORD)]
    KERNEL.CloseHandle.argtypes = [w.HANDLE]
    handle = KERNEL.OpenProcess(0x1000, False, pid)
    if not handle:
        raise c.WinError(c.get_last_error())
    try:
        code = w.DWORD()
        stamps = [w.FILETIME() for _ in range(4)]
        if not KERNEL.GetProcessTimes(handle, *map(c.byref, stamps)):
            raise c.WinError(c.get_last_error())
        if not KERNEL.GetExitCodeProcess(handle, c.byref(code)):
            raise c.WinError(c.get_last_error())
        return dict(pid=pid, alive=code.value == 259,
                    created=(stamps[0].dwHighDateTime << 32) + stamps[0].dwLowDateTime)
    finally:
        KERNEL.CloseHandle(handle)


def protected_unchanged():
    plan = json.loads((HERE / "console_cleanup_plan.json").read_text(encoding="utf-8"))
    checked = 0
    for pid, expected in plan["protected"].items():
        # System-owned ancestors are recorded as inaccessible, rather than as
        # worker identities. This helper never opens a handle for mutation.
        if expected["alive"] is None and expected["created"] is None:
            continue
        actual = process_identity(int(pid))
        if actual != expected:
            raise RuntimeError(f"Protected process identity changed: {pid}")
        checked += 1
    return checked


def paging_configuration():
    path = r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
        return dict(PagingFiles=winreg.QueryValueEx(key, "PagingFiles")[0])


def measure():
    value = Performance()
    value.cb = c.sizeof(value)
    PSAPI.GetPerformanceInfo.argtypes = [c.POINTER(Performance), w.DWORD]
    PSAPI.GetPerformanceInfo.restype = w.BOOL
    if not PSAPI.GetPerformanceInfo(c.byref(value), value.cb):
        raise c.WinError(c.get_last_error())
    pages = []
    callback_type = c.WINFUNCTYPE(w.BOOL, w.LPVOID, c.POINTER(PageFile), w.LPCWSTR)

    @callback_type
    def collect(context, info, name):
        pages.append(dict(path=name, allocated_bytes=info.contents.TotalSize * value.PageSize,
                          used_bytes=info.contents.TotalInUse * value.PageSize))
        return True

    PSAPI.EnumPageFilesW.argtypes = [callback_type, w.LPVOID]
    PSAPI.EnumPageFilesW.restype = w.BOOL
    if not PSAPI.EnumPageFilesW(collect, None):
        raise c.WinError(c.get_last_error())
    return dict(at_utc=datetime.now(timezone.utc).isoformat(),
                committed_bytes=value.CommitTotal * value.PageSize,
                commit_limit_bytes=value.CommitLimit * value.PageSize,
                headroom_bytes=(value.CommitLimit - value.CommitTotal) * value.PageSize,
                physical_bytes=value.PhysicalTotal * value.PageSize,
                available_physical_bytes=value.PhysicalAvailable * value.PageSize,
                pagefiles=pages, disk_free_bytes=shutil.disk_usage("C:\\").free,
                configuration=paging_configuration())


def validate(before):
    matches = [p for p in before["pagefiles"]
               if p["path"].lower().removeprefix("\\??\\") == r"c:\pagefile.sys"]
    if len(matches) != 1:
        raise RuntimeError("Expected one existing C:\\pagefile.sys")
    allocated = matches[0]["allocated_bytes"]
    if allocated > MINIMUM:
        raise RuntimeError("Current paging file already exceeds the proposed minimum; no change")
    if not 50 * GIB < before["physical_bytes"] < 80 * GIB:
        raise RuntimeError("This plan is restricted to the inspected 64 GiB workstation")
    if MAXIMUM < before["physical_bytes"] * 3:
        raise RuntimeError("Do not reduce the normal system-managed maximum")
    if before["disk_free_bytes"] - (MAXIMUM - allocated) < DISK_RESERVE:
        raise RuntimeError("Insufficient disk reserve even at the proposed maximum")
    return allocated


def enable_privilege():
    advapi = c.WinDLL("advapi32", use_last_error=True)
    KERNEL.GetCurrentProcess.restype = w.HANDLE
    advapi.OpenProcessToken.argtypes = [w.HANDLE, w.DWORD, c.POINTER(w.HANDLE)]
    advapi.LookupPrivilegeValueW.argtypes = [w.LPCWSTR, w.LPCWSTR, c.POINTER(Luid)]
    advapi.AdjustTokenPrivileges.argtypes = [w.HANDLE, w.BOOL, c.POINTER(TokenPrivileges),
                                           w.DWORD, w.LPVOID, w.LPVOID]
    token = w.HANDLE()
    if not advapi.OpenProcessToken(KERNEL.GetCurrentProcess(), 0x20 | 0x8, c.byref(token)):
        raise c.WinError(c.get_last_error())
    try:
        privileges = TokenPrivileges()
        privileges.PrivilegeCount = 1
        privileges.Attributes = 2
        if not advapi.LookupPrivilegeValueW(None, "SeCreatePagefilePrivilege", c.byref(privileges.Luid)):
            raise c.WinError(c.get_last_error())
        c.set_last_error(0)
        ok = advapi.AdjustTokenPrivileges(token, False, c.byref(privileges), 0, None, None)
        error = c.get_last_error()
        if not ok or error:
            raise c.WinError(error)
    finally:
        KERNEL.CloseHandle(token)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.apply and digest != args.expected_sha256:
        raise RuntimeError("Reviewed script hash required before elevation")
    before = measure()
    allocated = validate(before)
    protected_count = protected_unchanged()
    plan = dict(before=before, minimum_bytes=MINIMUM, maximum_bytes=MAXIMUM,
                immediate_disk_growth_bytes=MINIMUM - allocated,
                disk_reserve_at_maximum_bytes=before["disk_free_bytes"] - (MAXIMUM - allocated),
                protected_processes=protected_count, script_sha256=digest,
                scope="Runtime-only existing paging file extension; no persistent setting, reboot or process signal")
    if not args.apply:
        save("pagefile_plan.json", plan)
        print(json.dumps(plan, indent=2))
        return
    result = dict(plan=plan, status="starting")
    try:
        enable_privilege()
        native = c.WinDLL("ntdll")
        native.RtlInitUnicodeString.argtypes = [c.POINTER(UnicodeString), w.LPCWSTR]
        native.RtlInitUnicodeString.restype = None
        native.NtCreatePagingFile.argtypes = [c.POINTER(UnicodeString), c.POINTER(c.c_int64),
                                            c.POINTER(c.c_int64), w.ULONG]
        native.NtCreatePagingFile.restype = w.LONG
        name = UnicodeString()
        native_path = c.create_unicode_buffer(r"\??\C:\pagefile.sys")
        native.RtlInitUnicodeString(c.byref(name), c.cast(native_path, w.LPCWSTR))
        minimum, maximum = c.c_int64(MINIMUM), c.c_int64(MAXIMUM)
        protected_unchanged()
        status = native.NtCreatePagingFile(c.byref(name), c.byref(minimum), c.byref(maximum), 0)
        result["ntstatus"] = f"0x{status & 0xffffffff:08x}"
        if status < 0:
            raise RuntimeError(f"Native extension failed: {result['ntstatus']}")
        time.sleep(2)
        result["after"] = measure()
        result["protected_processes_unchanged"] = protected_unchanged()
        if result["after"]["configuration"] != before["configuration"]:
            raise RuntimeError("Persistent paging configuration unexpectedly changed")
        required_increase = max(0, MINIMUM - allocated - 128 * 1024 ** 2)
        if result["after"]["commit_limit_bytes"] - before["commit_limit_bytes"] < required_increase:
            raise RuntimeError("No verified online commit-limit increase")
        result["status"] = "verified_online_extension"
    except Exception as error:
        result["status"] = "not_verified"
        result["error"] = repr(error)
        result["after_error"] = measure()
    save("pagefile_apply_result.json", result)
    print(json.dumps(dict(status=result["status"], ntstatus=result.get("ntstatus"),
                         error=result.get("error")), indent=2))


if __name__ == "__main__":
    main()
