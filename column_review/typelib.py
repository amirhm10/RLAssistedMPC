"""Inspect registered Aspen API metadata without creating/attaching a COM app."""
from __future__ import annotations

TYPELIB = "{E85043BB-0E9D-11D3-82C4-00104B468DE2}"
REQUIRED = {
    "IAspenModelerDocument": {"Results", "Time"},
    "IAspenModelerResults": {"TakeSnapshot", "FindSnapshot", "LoadSnapshot"},
    "IAspenModelerSnapshot": {"ExportasBinary", "Converged"},
}


def inspect_api():
    import pythoncom
    import pywintypes
    library = pythoncom.LoadRegTypeLib(pywintypes.IID(TYPELIB), 1, 0, 0)
    interfaces = {}
    for index in range(library.GetTypeInfoCount()):
        info = library.GetTypeInfo(index)
        name = library.GetDocumentation(index)[0]
        if name not in REQUIRED:
            continue
        methods, properties = {}, []
        for position in range(info.GetTypeAttr().cFuncs):
            description = info.GetFuncDesc(position)
            names = info.GetNames(description.memid)
            methods[names[0]] = {"arguments": list(names[1:]), "signature": str(description)}
        for position in range(info.GetTypeAttr().cVars):
            properties.append(info.GetNames(info.GetVarDesc(position).memid)[0])
        available = set(methods) | set(properties)
        missing = REQUIRED[name] - available
        if missing:
            raise RuntimeError(f"Installed Aspen interface {name} lacks {sorted(missing)}.")
        interfaces[name] = {"methods": {key: value for key, value in methods.items() if key in REQUIRED[name]},
                            "properties": sorted(set(properties) & REQUIRED[name])}
    if set(interfaces) != set(REQUIRED):
        raise RuntimeError("The installed Aspen type library lacks a required snapshot interface.")
    return {"type_library": TYPELIB, "version": [1, 0], "name": library.GetDocumentation(-1)[0],
            "interfaces": interfaces, "live_aspen_opened": False,
            "scope": "Registered API signatures only; live snapshot restoration still needs its pilot."}
