from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from .config import (
    VANDEVUSSE_BENCHMARK_STATE_SEED,
    VANDEVUSSE_DELTA_T_HOURS,
    VANDEVUSSE_DESIGN_PARAMS,
    VANDEVUSSE_SS_INPUTS,
    VANDEVUSSE_SYSTEM_PARAMS,
)


class VanDeVusseCSTR:
    """
    Nonlinear Van de Vusse jacketed CSTR plant.

    Reaction network:
    A -> B -> C and 2A -> D

    States: [c_A, c_B, T, T_K]
    Inputs: [F, Q_K]
    Outputs: [c_B, T]

    This class only provides the nonlinear plant layer for the Van de Vusse
    case study. Later system-identification, MPC, and RL layers are added
    separately. Temperatures are handled in Kelvin internally.
    """

    def __init__(self, params, design_params, ss_inputs, delta_t, deviation_form=False):
        self.params = np.asarray(params, dtype=float)
        self.design_params = np.asarray(design_params, dtype=float)
        self.ss_inputs = np.asarray(ss_inputs, dtype=float)
        self.delta_t = float(delta_t)
        self.deviation_form = bool(deviation_form)

        self.k_10, self.k_20, self.k_30, self.E_a1, self.E_a2, self.E_a3, self.dH_AB, self.dH_BC, self.dH_AD, self.rho, self.C_p, self.C_pK, self.k_w, self.A_R, self.V_R, self.m_K = self.params
        self.c_A0, self.T_in = self.design_params
        # The literature parameters mix concentrations in mol/L with reactor
        # volume reported in m^3, so use liters in the heat-transfer term.
        self.V_R_liters = 1000.0 * self.V_R

        self.steady_trajectory = self.ss_params()
        self.y_ss = np.array([self.steady_trajectory[1], self.steady_trajectory[2]], dtype=float)

        if self.deviation_form:
            self.current_state = np.zeros(len(self.steady_trajectory), dtype=float)
            self.current_input = np.zeros(len(self.ss_inputs), dtype=float)
            self.current_output = np.zeros(len(self.y_ss), dtype=float)
        else:
            self.current_state = self.steady_trajectory.copy()
            self.current_input = self.ss_inputs.copy()
            self.current_output = self.y_ss.copy()

    def odes_deviation(self, t, x, u):
        x_abs = np.asarray(x, dtype=float) + self.steady_trajectory
        u_abs = np.asarray(u, dtype=float) + self.ss_inputs
        return self.odes(t, x_abs, u_abs)

    def odes(self, t, x, u):
        c_A, c_B, T, T_K = np.asarray(x, dtype=float)
        F, Q_K = np.asarray(u, dtype=float)

        k_1 = self.k_10 * np.exp(self.E_a1 / T)
        k_2 = self.k_20 * np.exp(self.E_a2 / T)
        k_3 = self.k_30 * np.exp(self.E_a3 / T)

        dc_A_dt = F * (self.c_A0 - c_A) - k_1 * c_A - k_3 * c_A ** 2
        dc_B_dt = -F * c_B + k_1 * c_A - k_2 * c_B
        dT_dt = (
            F * (self.T_in - T)
            - (k_1 * c_A * self.dH_AB + k_2 * c_B * self.dH_BC + k_3 * c_A ** 2 * self.dH_AD) / (self.rho * self.C_p)
            + (self.k_w * self.A_R / (self.rho * self.C_p * self.V_R_liters)) * (T_K - T)
        )
        dT_K_dt = (Q_K + self.k_w * self.A_R * (T - T_K)) / (self.m_K * self.C_pK)

        return dc_A_dt, dc_B_dt, dT_dt, dT_K_dt

    def ss_params(self):
        # The benchmark operating point is used as a physically meaningful seed
        # for the steady-state solve, but the final steady state still comes
        # from the nonlinear PINN-parameterized model and chosen ss_inputs.
        x_0 = np.asarray(VANDEVUSSE_BENCHMARK_STATE_SEED, dtype=float).copy()
        x_ss = fsolve(lambda x: self.odes(0.0, x, self.ss_inputs), x_0)
        return np.asarray(x_ss, dtype=float)

    def step(self):
        if self.deviation_form:
            sol = solve_ivp(self.odes_deviation, [0.0, self.delta_t], self.current_state, args=(self.current_input,))
            self.current_state = sol.y[:, -1]
            current_abs_state = self.current_state + self.steady_trajectory
            self.current_output = np.array([current_abs_state[1] - self.y_ss[0], current_abs_state[2] - self.y_ss[1]], dtype=float)
        else:
            sol = solve_ivp(self.odes, [0.0, self.delta_t], self.current_state, args=(self.current_input,))
            self.current_state = sol.y[:, -1]
            self.current_output = np.array([self.current_state[1], self.current_state[2]], dtype=float)


def build_vandevusse_system(params=None, design_params=None, ss_inputs=None, delta_t=None, deviation_form=False):
    return VanDeVusseCSTR(
        params=VANDEVUSSE_SYSTEM_PARAMS if params is None else params,
        design_params=VANDEVUSSE_DESIGN_PARAMS if design_params is None else design_params,
        ss_inputs=VANDEVUSSE_SS_INPUTS if ss_inputs is None else ss_inputs,
        delta_t=VANDEVUSSE_DELTA_T_HOURS if delta_t is None else delta_t,
        deviation_form=deviation_form,
    )


def vandevusse_system_stepper(system, disturbance_step=None):
    if disturbance_step is None:
        system.step()
        return

    if not isinstance(disturbance_step, dict):
        raise ValueError("Van de Vusse disturbance_step must be None or a dict with 'c_A0' and/or 'T_in'.")

    if "c_A0" in disturbance_step:
        system.c_A0 = float(disturbance_step["c_A0"])
    if "T_in" in disturbance_step:
        system.T_in = float(disturbance_step["T_in"])
    system.step()
