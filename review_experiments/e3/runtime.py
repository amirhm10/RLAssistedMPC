"""Process-local entrypoint configuration and passive plant-step instrumentation."""
from __future__ import annotations

import ast
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import functools
import importlib
import inspect
from pathlib import Path
import textwrap

from .common import ENTRIES, PROFILES, ROOT, SEED_KEYS, sha256

TARGETS = {"markov": ("utils.markov_runner", "run_single_closed_loop"),
           "combined": ("utils.combined_runner", "run_combined_supervisor")}


def configure_defaults(defaults, plant, category, seed, output, aspen=None):
    configured = deepcopy(defaults)
    for key in SEED_KEYS:
        if isinstance(configured.get(key), dict):
            configured[key]["seed"] = seed
    configured["results_dir_override"] = str(output)
    configured["result_prefix_override"] = "e3"
    configured["compare_prefix_override"] = "e3_comparison"
    if category == "nominal":
        configured.update(run_mode="nominal", training_profile_override=("legacy_gradual_200" if plant == "polymer" else "legacy_200"),
                          n_tests_override=2, warm_start_override=1, test_cycle_override=[True],
                          baseline_save_path_override=str(Path(output) / "unused_canonical_baseline.pkl"))
        if plant == "distillation":
            configured["disturbance_profile"] = "none"
    else:
        if configured.get("n_tests_override") not in (None, 300):
            raise ValueError("E3 preserves the full 300-episode study; n_tests_override conflicts.")
        configured.update(run_mode="disturb", training_profile_override=PROFILES[plant])
        if plant == "distillation":
            configured["disturbance_profile"] = "fluctuation"
            configured["agent_mode"] = "sg"
        else:
            configured["agent_kind"] = "sg_td3"
        if category == "combined":
            configured["combined_agent_mode"] = "sg"
    if aspen is not None:
        configured["aspen_path_override"] = aspen["aspen_path"]
        configured["snaps_path_override"] = aspen["snaps_path"]
    return configured


def _prefix(statements):
    """Trim after the training assignment, retaining enclosing try/finally cleanup."""
    prefix = []
    for original in statements:
        node = deepcopy(original)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "result_bundle" for t in node.targets):
            return prefix + [node], True
        if isinstance(node, ast.Try):
            body, found = _prefix(node.body)
            if found:
                node.body, node.orelse = body, []
                # Exception handlers can perform unrelated postprocessing; the
                # inspected active entrypoints have cleanup-only try/finally.
                if node.handlers:
                    raise ValueError("Unsupported entrypoint handlers around result_bundle.")
                return prefix + [node], True
        prefix.append(node)
    return prefix, False


def execute_entrypoint(plant, category, configure):
    path = ROOT / ENTRIES[plant][category]
    body, found = _prefix(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body)
    if not found:
        raise ValueError(f"No recognized training assignment in {path.name}.")
    tree = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    module = importlib.import_module(f"systems.{plant}")
    name = f"get_{plant}_notebook_defaults"
    original = getattr(module, name)
    family = "baseline" if category == "nominal" else category

    def provider(requested):
        defaults = original(requested)
        if requested == family:
            return configure(defaults)
        # Combined's parity checks must see the same seeds, and nothing else.
        seeded = deepcopy(defaults)
        selected = configure(deepcopy(defaults))
        for key in SEED_KEYS:
            if isinstance(seeded.get(key), dict):
                seeded[key]["seed"] = selected[key]["seed"]
        return seeded

    namespace = {"__name__": "__e3_entrypoint__", "__file__": str(path)}
    setattr(module, name, provider)
    try:
        exec(compile(tree, str(path), "exec"), namespace, namespace)
    finally:
        setattr(module, name, original)
    return namespace


class _PlantHooks(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def visit_Expr(self, node):
        call = node.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "step_system_with_disturbance":
            self.count += 1
            return [ast.copy_location(ast.parse("__e3_before(locals())").body[0], node), node,
                    ast.copy_location(ast.parse("__e3_after(locals())").body[0], node)]
        return self.generic_visit(node)


def instrument_function(function, before, after):
    lines, start_line = inspect.getsourcelines(function)
    source = textwrap.dedent("".join(lines))
    tree = ast.parse(source, filename=inspect.getsourcefile(function))
    root = tree.body[0]
    if not isinstance(root, ast.FunctionDef) or function.__closure__:
        raise ValueError("E3 supports only the inspected module-level runner functions.")
    root.decorator_list = []
    hooks = _PlantHooks()
    tree = hooks.visit(tree)
    if hooks.count != 1:
        raise ValueError(f"Expected one plant step in {function.__name__}; found {hooks.count}.")
    namespace = dict(function.__globals__)
    namespace.update(__e3_before=before, __e3_after=after)
    tree = ast.increment_lineno(ast.fix_missing_locations(tree), start_line - 1)
    exec(compile(tree, inspect.getsourcefile(function), "exec"), namespace)
    return functools.wraps(function)(namespace[function.__name__])


@contextmanager
def passive_capture(category, session):
    module_name, function_name = TARGETS[category]
    module = importlib.import_module(module_name)
    original = getattr(module, function_name)
    instrumented = instrument_function(original, session.before, session.after)

    @functools.wraps(original)
    def run(*args, **kwargs):
        if category == "markov" and not kwargs.get("use_markov", True):
            return original(*args, **kwargs)
        result = instrumented(*args, **kwargs)
        session.finish()
        return result

    setattr(module, function_name, run)
    try:
        yield
    finally:
        setattr(module, function_name, original)


def resolve_aspen(category, root_override=None):
    from systems.distillation import get_distillation_notebook_defaults
    from systems.distillation.config import resolve_aspen_paths
    family = "baseline" if category == "nominal" else category
    defaults = get_distillation_notebook_defaults(family)
    path, snaps, source = resolve_aspen_paths(
        family=family, disturbance_profile="none" if category == "nominal" else "fluctuation",
        aspen_preset=defaults["aspen_preset"], dyn_path_override=defaults["aspen_path_override"],
        snaps_path_override=defaults["snaps_path_override"], aspen_root=defaults["aspen_root_override"])
    # The existing resolver's family-default branch does not use aspen_root.
    # Preserve the resolved basename and matching snapshot directory convention.
    if root_override:
        from systems.distillation.config import resolve_snapshot_dir
        path = Path(root_override).expanduser().resolve() / Path(path).name
        snaps = resolve_snapshot_dir(path)
        source = "e3-root-override"
    path, snaps = Path(path).resolve(), Path(snaps).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing Aspen file: {path}")
    return {"aspen_path": str(path), "snaps_path": str(snaps), "path_source": source,
            "aspen_sha256": sha256(path)}


@contextmanager
def aspen_resources(paths):
    """Reuse verified COM dispatch and locks, with idempotent owned-session cleanup."""
    from run_distillation_phase2_repeats import aspen_com_compatibility, aspen_file_lock
    import systems.distillation as distillation
    import systems.distillation.plant as plant
    original, exported, columns = plant.DistillationColumnAspen, distillation.DistillationColumnAspen, []

    class Column(original):
        def __init__(self, *args, **kwargs):
            self._e3_closed = False
            columns.append(self)
            super().__init__(*args, **kwargs)

        def close(self, snaps_path=None, prefix="snp"):
            if not self._e3_closed:
                super().close(snaps_path or paths["snaps_path"], prefix=prefix)
                self._e3_closed = True

    with ExitStack() as stack:
        lock_dir = ROOT / "Distillation" / "Results" / "_aspen_file_locks"
        stack.enter_context(aspen_file_lock(Path(paths["aspen_path"]), lock_dir))
        if paths.get("aspen_sha256") and sha256(paths["aspen_path"]) != paths["aspen_sha256"]:
            raise RuntimeError("Aspen model changed after E3 preflight; start a new study.")
        stack.enter_context(aspen_com_compatibility())
        plant.DistillationColumnAspen = distillation.DistillationColumnAspen = Column
        try:
            yield
        finally:
            errors = []
            try:
                for column in columns:
                    if getattr(column, "ad", None) is not None:
                        try:
                            column.close(paths["snaps_path"])
                        except Exception as error:
                            errors.append(error)
                if errors:
                    raise RuntimeError("E3 Aspen cleanup failed; inspect the worker's owned session before resuming.") from errors[0]
            finally:
                plant.DistillationColumnAspen, distillation.DistillationColumnAspen = original, exported
