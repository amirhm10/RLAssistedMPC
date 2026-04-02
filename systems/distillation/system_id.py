import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.helpers import apply_min_max, reverse_min_max


def generate_step_test_data(u_start, step_value, initial_duration=40, step_duration=200, step_index=0):
    initial_array = np.full((int(initial_duration), len(u_start)), np.asarray(u_start, float))
    stepped_input = np.asarray(u_start, float).copy()
    stepped_input[int(step_index)] += float(step_value)
    step_array = np.full((int(step_duration), len(u_start)), stepped_input)
    return np.concatenate((initial_array, step_array), axis=0)


def simulate_distillation_system(system, input_sequence):
    outputs = [np.asarray(system.current_output, float)]
    for inp in np.asarray(input_sequence, float):
        system.current_input = np.asarray(inp, float)
        system.step()
        outputs.append(np.asarray(system.current_output, float))
    return {"inputs": np.asarray(input_sequence, float), "outputs": np.asarray(outputs, float)}


def run_distillation_experiment(
    system_cls,
    path,
    ss_inputs,
    nominal_conditions,
    delta_t,
    step_value,
    step_channel,
    save_filename,
    data_dir,
):
    system = system_cls(path=path, ss_inputs=ss_inputs, initialization_point=nominal_conditions, delta_t=delta_t)
    try:
        step_data = generate_step_test_data(system.current_input, step_value, step_index=step_channel)
        results = simulate_distillation_system(system, step_data)
    finally:
        try:
            system.close()
        except Exception:
            pass

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    save_path = data_dir / save_filename
    data_to_save = np.concatenate((results["inputs"], results["outputs"][1:]), axis=1)
    if int(step_channel) == 0:
        column_names = ["Reflux", "Reboiler", "Tray24_C2H6", "Tray85_T"]
    else:
        column_names = ["Reflux", "Reboiler", "Tray24_C2H6", "Tray85_T"]
    pd.DataFrame(data_to_save, columns=column_names).to_csv(save_path, index=False)
    return results


def scaling_min_max_factors(file_paths):
    data_min = []
    data_max = []
    for path in file_paths.values():
        df = pd.read_csv(path)
        data_min.append(df.min())
        data_max.append(df.max())
    return np.min(data_min, axis=0), np.max(data_max, axis=0)


def apply_deviation_form_scaled(steady_states, file_paths, data_min, data_max):
    u_ss = np.asarray(steady_states["ss_inputs"], float)
    y_ss = np.asarray(steady_states["y_ss"], float)
    ss = np.concatenate((u_ss, y_ss), axis=0)
    ss_scaled = apply_min_max(ss, data_min, data_max)
    deviations = {}
    for key, path in file_paths.items():
        df = pd.read_csv(path)
        deviations[key] = apply_min_max(df, data_min, data_max) - ss_scaled
    return deviations


def save_canonical_system_identification(data_dir, system_dict, scaling_factor, min_max_states=None, carry_forward_min_max_path=None):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "system_dict.pickle", "wb") as handle:
        pickle.dump(system_dict, handle)
    with open(data_dir / "scaling_factor.pickle", "wb") as handle:
        pickle.dump(scaling_factor, handle)

    if min_max_states is None and carry_forward_min_max_path is not None:
        with open(carry_forward_min_max_path, "rb") as handle:
            min_max_states = pickle.load(handle)
    if min_max_states is not None:
        with open(data_dir / "min_max_states.pickle", "wb") as handle:
            pickle.dump(min_max_states, handle)


def extract_fopdt_2863(df, input_idx=0, Ts=1 / 6, limit=25, pre_win=30, post_win=60, plot=True):
    """
    Estimate FOPDT-style channel parameters from a single step test using the 28/63% method.

    The function supports mild inverse-response behavior by identifying an early extremum and
    shifting the 28/63% crossing search to start after that extremum.
    """

    cols = list(df.columns)
    if len(cols) < 4:
        raise ValueError("Expected step-test data with at least 4 columns: 2 inputs and 2 outputs.")

    in_col = cols[int(input_idx)]
    out_cols = cols[-2:]

    u = df[in_col].to_numpy(dtype=float)
    base = float(np.median(u[: max(pre_win, 5)]))
    du = np.abs(u - base)
    noise = float(np.median(np.abs(du[: max(pre_win, 20)] - np.median(du[: max(pre_win, 20)]))))
    threshold = max(1e-9, 5.0 * noise)
    k0 = int(np.argmax(du > threshold))
    k0 = max(k0 - 1, 0)
    data = df.iloc[k0:].reset_index(drop=True)

    t = np.arange(len(data), dtype=float) * float(Ts)
    results = {}

    denom = np.log(0.72 / 0.37)
    log072 = np.log(0.72)

    def crossing_time_after(y, target, start_idx):
        if start_idx >= len(y) - 1:
            return np.nan
        sign = np.sign(y - target)
        sign[:start_idx] = sign[start_idx]
        crossings = np.where(np.diff(sign) != 0)[0]
        if crossings.size == 0:
            return np.nan
        k = int(crossings[0])
        y0_local = float(y[k])
        y1_local = float(y[k + 1])
        if y1_local == y0_local:
            return float(t[k])
        frac = (float(target) - y0_local) / (y1_local - y0_local)
        return float(t[k] + float(Ts) * frac)

    for yname in out_cols:
        y = data[yname].to_numpy(dtype=float)
        u_seg = data[in_col].to_numpy(dtype=float)

        k_pre = min(pre_win, max(1, len(y) // 10))
        k_post = min(post_win, max(1, len(y) // 10))
        y0 = float(np.mean(y[:k_pre]))
        yF = float(np.mean(y[-k_post:]))
        u0 = float(np.mean(u_seg[:k_pre]))
        uF = float(np.mean(u_seg[-k_post:]))

        dY = yF - y0
        dU = uF - u0 if abs(uF - u0) > 0 else np.nan
        kp = dY / dU if np.isfinite(dU) else np.nan

        final_sign = np.sign(dY) if dY != 0 else 0.0
        search_end = max(int(0.4 * len(y)), k_pre + 5)
        if final_sign >= 0:
            k_ext = int(np.argmin(y[:search_end]))
        else:
            k_ext = int(np.argmax(y[:search_end]))

        inverse = False
        if final_sign != 0:
            early_move = np.sign(float(y[min(k_ext, len(y) - 1)]) - y0)
            inverse = (early_move != 0) and (early_move != final_sign)

        if inverse:
            y_ext = float(y[k_ext])
            M = abs((y_ext - y0) / dY) if dY != 0 else 0.0
            start_idx = k_ext
        else:
            M = 0.0
            start_idx = 0

        y28 = y0 + 0.28 * dY
        y63 = y0 + 0.63 * dY

        t28 = crossing_time_after(y, y28, start_idx)
        t63 = crossing_time_after(y, y63, start_idx)

        if (not np.isfinite(t28)) or (not np.isfinite(t63)) or (t63 <= t28):
            tau = np.nan
            theta = np.nan
            tz_est = np.nan
        else:
            tau = (t63 - t28) / denom
            theta = t28 + tau * log072
            tz_est = M * tau if inverse else 0.0

        results[yname] = {
            "kp": float(kp) if np.isfinite(kp) else np.nan,
            "t28": float(t28) if np.isfinite(t28) else np.nan,
            "t63": float(t63) if np.isfinite(t63) else np.nan,
            "taup": float(tau) if np.isfinite(tau) else np.nan,
            "theta": float(max(0.0, theta)) if np.isfinite(theta) else np.nan,
            "inverse": bool(inverse),
            "t_ext": float(t[k_ext]) if inverse else None,
            "M": float(M),
            "Tz_est": float(tz_est) if np.isfinite(tz_est) else np.nan,
            "y0": y0,
            "yF": yF,
        }

        if plot:
            plt.figure(figsize=(7.5, 5.5))
            plt.plot(t, y, label=f"{yname} response")
            plt.axhline(y28, linestyle="--", label="28% level")
            plt.axhline(y63, linestyle="--", label="63% level")
            if inverse:
                plt.axvline(t[k_ext], linestyle=":", label="inverse extremum")
            if np.isfinite(t28):
                plt.axvline(t28, linestyle="--", label="t28")
                plt.scatter([t28], [y28], zorder=5)
            if np.isfinite(t63):
                plt.axvline(t63, linestyle="--", label="t63")
                plt.scatter([t63], [y63], zorder=5)
            plt.xlim([0.0, t[min(limit, len(t) - 1)]])
            plt.xlabel("Time (h)")
            plt.ylabel(yname)
            plt.title(f"28/63 identification for {yname}")
            plt.legend()
            plt.tight_layout()
            plt.show()

    return results, data


def build_delay_list_hours(u1_dict, u2_dict, y1_name, y2_name, Ts_hours=1 / 6, quantization="nearest"):
    """
    Build the 2x2 delay matrix [d11 d12; d21 d22] and return a flattened delay list in hours.
    """

    D = np.array(
        [
            [u1_dict[y1_name].get("theta", 0.0), u2_dict[y1_name].get("theta", 0.0)],
            [u1_dict[y2_name].get("theta", 0.0), u2_dict[y2_name].get("theta", 0.0)],
        ],
        dtype=float,
    )
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
    D = np.clip(D, 0.0, None)

    if quantization == "ceil":
        k = np.ceil(D / Ts_hours)
    elif quantization == "floor":
        k = np.floor(D / Ts_hours)
    else:
        k = np.rint(D / Ts_hours)

    k = k.astype(int)
    Dq = k * float(Ts_hours)
    return Dq.flatten(order="C").tolist(), k


def state_space_form_using_matlab(
    u1_dict,
    u2_dict,
    delay_list_hours,
    data_u1,
    data_u2,
    sampling_time=1 / 6,
    use_rhp_zero=True,
    pre_win=30,
    post_win=60,
    return_step=True,
):
    """
    Build a discrete 2x2 state-space model in MATLAB with absorbed delays.
    """

    try:
        import matlab
        import matlab.engine
    except ImportError as exc:
        raise ImportError("matlab.engine is required to build the distillation system model.") from exc

    def _step_amp(df, col, pre=30, post=60):
        x = df[col].to_numpy(dtype=float)
        pre = min(pre, max(1, len(x) // 10))
        post = min(post, max(1, len(x) // 10))
        return float(np.mean(x[-post:]) - np.mean(x[:pre]))

    def _safe(v, floor=None):
        if v is None or not np.isfinite(v):
            return None
        v = float(v)
        if floor is not None and v < floor:
            return floor
        return v

    def _sanitize_channel(ch):
        kp = _safe(ch.get("kp"))
        tau = _safe(ch.get("taup"), floor=1e-6)
        theta = _safe(ch.get("theta"), floor=0.0)
        tz = _safe(ch.get("Tz_est", 0.0), floor=0.0)

        if kp is None or tau is None:
            return 0.0, 1.0, 0.0, 0.0
        if theta is None:
            theta = 0.0
        for value in (kp, tau, theta, tz):
            if not np.isfinite(value):
                return 0.0, 1.0, 0.0, 0.0
        return float(kp), float(tau), float(theta), float(tz)

    def _fmt(value):
        if not np.isfinite(value):
            return "0"
        value = float(value)
        if abs(value) < 1e-15:
            return "0"
        return f"{value:.12g}"

    u1_name, u2_name, y1_name, y2_name = data_u1.columns[:4]
    delta_u = [
        _step_amp(data_u1, u1_name, pre=pre_win, post=post_win),
        _step_amp(data_u2, u2_name, pre=pre_win, post=post_win),
    ]

    kp11, tau11, th11, tz11 = _sanitize_channel(u1_dict[y1_name])
    kp21, tau21, th21, tz21 = _sanitize_channel(u1_dict[y2_name])
    kp12, tau12, th12, tz12 = _sanitize_channel(u2_dict[y1_name])
    kp22, tau22, th22, tz22 = _sanitize_channel(u2_dict[y2_name])

    if delay_list_hours is None or len(delay_list_hours) != 4:
        delay_list_hours = [th11, th12, th21, th22]

    d11, d12, d21, d22 = [
        max(0.0, float(d)) if np.isfinite(d) else 0.0
        for d in delay_list_hours
    ]

    if use_rhp_zero:
        num = (
            f"num = {{[{_fmt(-kp11 * tz11)}, {_fmt(kp11)}], [{_fmt(-kp12 * tz12)}, {_fmt(kp12)}]; "
            f"[{_fmt(-kp21 * tz21)}, {_fmt(kp21)}], [{_fmt(-kp22 * tz22)}, {_fmt(kp22)}]}};"
        )
    else:
        num = (
            f"num = {{{_fmt(kp11)}, {_fmt(kp12)}; "
            f"{_fmt(kp21)}, {_fmt(kp22)}}};"
        )

    den = (
        f"den = {{[{_fmt(tau11)}, 1], [{_fmt(tau12)}, 1]; "
        f"[{_fmt(tau21)}, 1], [{_fmt(tau22)}, 1]}};"
    )
    delay = f"delay = [{_fmt(d11)}, {_fmt(d12)}; {_fmt(d21)}, {_fmt(d22)}];"

    eng = matlab.engine.start_matlab()
    Ts = float(sampling_time)
    end_time = (data_u1.shape[0] - 1) * Ts

    eng.eval(num, nargout=0)
    eng.eval(den, nargout=0)
    eng.eval(delay, nargout=0)
    eng.eval("tf_system = tf(num, den, 'IODelay', delay, 'TimeUnit','hours');", nargout=0)
    eng.workspace["Ts"] = Ts
    eng.eval("ss_system = ss(tf_system);", nargout=0)
    eng.eval("mimo_ss_dis = c2d(ss_system, Ts, 'zoh');", nargout=0)
    eng.eval("mimo_ss_dis_ab_delay = absorbDelay(mimo_ss_dis);", nargout=0)

    A = np.array(eng.eval("mimo_ss_dis_ab_delay.A", nargout=1))
    B = np.array(eng.eval("mimo_ss_dis_ab_delay.B", nargout=1))
    C = np.array(eng.eval("mimo_ss_dis_ab_delay.C", nargout=1))
    D = np.array(eng.eval("mimo_ss_dis_ab_delay.D", nargout=1))

    out = [A, B, C, D]
    if return_step:
        eng.workspace["end_time"] = float(end_time)
        eng.eval("t = 0:Ts:end_time;", nargout=0)
        eng.eval(
            f"opt = stepDataOptions('InputOffset',0,'StepAmplitude',[{_fmt(delta_u[0])} {_fmt(delta_u[1])}]);",
            nargout=0,
        )
        eng.eval("[y_step, tOut] = step(mimo_ss_dis_ab_delay, t, opt);", nargout=0)
        y_step = np.array(eng.workspace["y_step"])
        tOut = np.array(eng.workspace["tOut"]).ravel()
        out.extend([y_step.sum(axis=2), y_step, tOut])

    return tuple(out)


def plot_state_space_validation(tOut, y_step, data_u1, data_u2, Ts=1 / 6, pre_win=30):
    """
    Plot measured vs simulated response for each input-output pair.
    """

    if y_step.ndim != 3:
        raise ValueError("Expected y_step with shape (T, Ny, Nu).")

    input_name1, input_name2 = data_u1.columns[:2]
    output_name1, output_name2 = data_u1.columns[2:4]

    def detect_step_idx(u, pre_window=30, min_run=3):
        u = np.asarray(u, float)
        base = np.median(u[: max(pre_window, 5)])
        atol = 10 * np.finfo(float).eps * max(1.0, abs(base))
        mask = ~np.isclose(u, base, rtol=1e-6, atol=atol)
        run = np.convolve(mask.astype(int), np.ones(min_run, dtype=int), "same") >= min_run
        idxs = np.flatnonzero(run)
        k = int(idxs[0]) if idxs.size else 0
        return max(k - 1, 0)

    def fit_percent(y_sim, y_meas):
        y_sim = np.asarray(y_sim).ravel()
        y_meas = np.asarray(y_meas).ravel()
        n = min(len(y_sim), len(y_meas))
        y_sim = y_sim[:n]
        y_meas = y_meas[:n]
        denom = np.linalg.norm(y_meas - np.mean(y_meas))
        if denom < 1e-12:
            return 0.0
        return 100.0 * (1.0 - np.linalg.norm(y_sim - y_meas) / denom)

    def prep_actual(df, in_col, out_col):
        u = df[in_col].to_numpy()
        y = df[out_col].to_numpy()
        k0 = detect_step_idx(u, pre_window=pre_win)
        y0 = float(np.mean(y[max(0, k0 - pre_win) : k0 + 1])) if k0 > 0 else float(np.mean(y[:pre_win]))
        y_dev = y[k0:] - y0
        n = min(len(tOut), len(y_dev))
        return tOut[:n], y_dev[:n]

    t_ref_y1, y1_actual_ref = prep_actual(data_u1, input_name1, output_name1)
    t_ref_y2, y2_actual_ref = prep_actual(data_u1, input_name1, output_name2)
    t_reb_y1, y1_actual_reb = prep_actual(data_u2, input_name2, output_name1)
    t_reb_y2, y2_actual_reb = prep_actual(data_u2, input_name2, output_name2)

    y1_sim_ref = y_step[: len(t_ref_y1), 0, 0]
    y2_sim_ref = y_step[: len(t_ref_y2), 1, 0]
    y1_sim_reb = y_step[: len(t_reb_y1), 0, 1]
    y2_sim_reb = y_step[: len(t_reb_y2), 1, 1]

    fit_ref_y1 = fit_percent(y1_sim_ref, y1_actual_ref)
    fit_ref_y2 = fit_percent(y2_sim_ref, y2_actual_ref)
    fit_reb_y1 = fit_percent(y1_sim_reb, y1_actual_reb)
    fit_reb_y2 = fit_percent(y2_sim_reb, y2_actual_reb)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axs[0, 0].plot(t_ref_y1, y1_actual_ref, "r-", linewidth=2, label="Measured")
    axs[0, 0].plot(t_ref_y1, y1_sim_ref, "b--", linewidth=2, label="Simulated")
    axs[0, 0].set(xlabel="Time (h)", ylabel=output_name1, title=f"Step in {input_name1} | FIT={fit_ref_y1:.1f}%")

    axs[1, 0].plot(t_ref_y2, y2_actual_ref, "r-", linewidth=2, label="Measured")
    axs[1, 0].plot(t_ref_y2, y2_sim_ref, "b--", linewidth=2, label="Simulated")
    axs[1, 0].set(xlabel="Time (h)", ylabel=output_name2, title=f"Step in {input_name1} | FIT={fit_ref_y2:.1f}%")

    axs[0, 1].plot(t_reb_y1, y1_actual_reb, "r-", linewidth=2, label="Measured")
    axs[0, 1].plot(t_reb_y1, y1_sim_reb, "b--", linewidth=2, label="Simulated")
    axs[0, 1].set(xlabel="Time (h)", ylabel=output_name1, title=f"Step in {input_name2} | FIT={fit_reb_y1:.1f}%")

    axs[1, 1].plot(t_reb_y2, y2_actual_reb, "r-", linewidth=2, label="Measured")
    axs[1, 1].plot(t_reb_y2, y2_sim_reb, "b--", linewidth=2, label="Simulated")
    axs[1, 1].set(xlabel="Time (h)", ylabel=output_name2, title=f"Step in {input_name2} | FIT={fit_reb_y2:.1f}%")

    for ax in axs.flat:
        ax.grid(True)
        ax.legend()

    plt.show()
