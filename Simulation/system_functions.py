import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp


class PolymerCSTR:
    def __init__(self, params, design_params, ss_inputs, delta_t, deviation_form=False):
        self.params = params
        self.ss_inputs = ss_inputs

        # Parameter Design of the reactor
        self.CIf, self.CMf, self.Qi, self.Qs, self.Tf, self.Tcf, self.V, self.Vc = design_params
        self.Ad, self.Ed, self.Ap, self.Ep, self.At, self.Et, self.fi, self.m_delta_H_r, self.hA, self.rhocp, self.rhoccpc, self.Mm = params
        self.delta_t = delta_t
        self.steady_trajectory = self.ss_params()
        self.viscosity_ss = 0.0012 * (self.params[-1] *
                                      self.steady_trajectory[-1] / self.steady_trajectory[-2]) ** 0.71
        self.y_ss = np.array([self.viscosity_ss, self.steady_trajectory[2]])

        self.deviation_form = deviation_form

        if deviation_form:
            # Since we are in deviation form
            self.current_state = np.zeros(len(self.steady_trajectory))
            self.current_input = np.zeros(len(self.ss_inputs))
            self.current_viscosity = 0.0
        else:
            # out of deviation form
            self.current_state = self.steady_trajectory
            self.current_input = self.ss_inputs
            self.current_viscosity = self.viscosity_ss
        self.current_output = np.array([self.current_viscosity, self.current_state[2]])

    def odes_deviation(self, t, x, u):
        """
        dCI/dt = (Qi*CIf - Qt*CI)/V - k*dCI
        dCM/dt = (Qm*CMf - Qt*CM)/V - kp*CM*CP
        dT/dt  = Qt*(Tf-T)/V - (Delta_H/(rhocp))*kp*CM*CP - hA/(rhocp*V)*(T - Tc)
        dTc/dt = Qc*(Tcf-Tc)/Vc + hA/(rhoccpc*Vc)*(T-Tc)
        dD0/dt = 0.5*kt*CP**2 - Qt*D0/V
        dD1/dt = Mm*kp*CM*CP - Qt*D1/V
        dD2/dt = 5*Mm*kp*CM*CP + (3*Mm*kp**2)/kt*CM**2 - Qt*D2/V
        CP = sqrt(2*fi*kd*CI/kt)
        Qt = Qi + Qs + Qm
        """
        # Instead of modifying in place, create new arrays for the absolute state and input:
        x_abs = x + self.steady_trajectory
        u_abs = u + self.ss_inputs

        return self.odes(t, x_abs, u_abs)

    def odes(self, t, x, u):
        # current inputs and previous states
        Qc, Qm = u
        CI, CM, T, Tc, D0, D1, D2 = x

        # Calculating dependent parameters in the odes
        kd = self.Ad * np.exp(-self.Ed / T)
        kp = self.Ap * np.exp(-self.Ep / T)
        kt = self.At * np.exp(-self.Et / T)
        CP = (2 * self.fi * kd * CI / kt) ** 0.5
        Qt = self.Qi + self.Qs + Qm

        # System of ODEs
        dCIdt = (self.Qi * self.CIf - Qt * CI) / self.V - kd * CI
        dCMdt = (Qm * self.CMf - Qt * CM) / self.V - kp * CM * CP
        dTdt = Qt * (self.Tf - T) / self.V - (self.m_delta_H_r / self.rhocp) * kp * CM * CP - self.hA / (self.rhocp * self.V) * (T - Tc)
        # In the paper itself for the dTcdt first term was Qt, but instead it should be Qc
        dTcdt = Qc * (self.Tcf - Tc) / self.Vc + self.hA / (self.rhoccpc * self.Vc) * (T - Tc)
        dD0dt = 0.5 * kt * CP ** 2 - Qt * D0 / self.V
        dD1dt = self.Mm * kp * CM * CP - Qt * D1 / self.V
        dD2dt = 5 * self.Mm * kp * CM * CP + (3 * self.Mm * kp ** 2) / kt * CM ** 2 - Qt * D2 / self.V
        return dCIdt, dCMdt, dTdt, dTcdt, dD0dt, dD1dt, dD2dt

    def ss_params(self):
        x_0 = np.array([6e-2, 3.3, 320, 310, 0, 0, 1])

        x_ss = fsolve(lambda x: self.odes(0.0, x, self.ss_inputs), x_0)

        return x_ss

    def step(self):
        if self.deviation_form:
            sol = solve_ivp(self.odes_deviation, [0, self.delta_t], self.current_state, args=(self.current_input,))

            self.current_state = sol.y[:, -1]

            viscosity = (0.0012 * (self.params[-1] *
                                   (self.current_state[-1] + self.steady_trajectory[-1]) / (self.current_state[-2]
                                                                                            + self.steady_trajectory[
                                                                                                -2]))
                         ** 0.71)

            self.current_viscosity = viscosity - self.viscosity_ss
        else:
            sol = solve_ivp(self.odes, [0, self.delta_t], self.current_state, args=(self.current_input,))

            self.current_state = sol.y[:, -1]

            viscosity = (0.0012 * (self.params[-1] *
                                   (self.current_state[-1]) / (self.current_state[-2])) ** 0.71)

            self.current_viscosity = viscosity

        self.current_output = np.array([self.current_viscosity, self.current_state[2]])
