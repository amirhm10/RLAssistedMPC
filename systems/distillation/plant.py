from pathlib import Path

import numpy as np

try:
    import win32com.client  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    win32com = None


class DistillationColumnAspen:
    def __init__(self, path, ss_inputs, initialization_point, delta_t=1.0 / 6.0, visible=True):
        if win32com is None:  # pragma: no cover - depends on local Aspen install
            raise ImportError("win32com.client is required to use DistillationColumnAspen.")

        self.path = str(Path(path).expanduser())
        self.delta_t = float(delta_t)
        self.initialization_point = np.asarray(initialization_point, dtype=float)
        self.ss_inputs = np.asarray(ss_inputs, dtype=float)

        self.ad = win32com.client.DispatchEx("AD application")
        self.ad.NewDocument()
        self.ad.Visible = bool(visible)
        self.ad.activate()
        self.ad.Maximize()
        self.ad.openDocument(self.path)

        self.sim = self.ad.Simulation
        self.fsheet = self.sim.Flowsheet
        self.streams = self.fsheet.Streams
        self.block = self.fsheet.Blocks
        self.col = self.block("C2S")
        self.feed = self.streams("Feed")

        self.y_ss = self.ss_outputs()
        self.current_input = self.ss_inputs.copy()
        self.current_output = self.y_ss.copy()

    def enable_records(self):
        self.col.Reflux.FmR.Record = True
        self.col.QRebR.Record = True
        self.col.Stage(24).y("C2H6").Record = True
        self.col.Stage(85).T.Record = True

    def initialize_system(self):
        feed = self.streams("Feed")
        feed.FmR.Value = self.initialization_point[0]
        feed.T.Value = self.initialization_point[1]
        feed.P.Value = self.initialization_point[2]
        feed.ZR("C2H4").Value = self.initialization_point[3]
        feed.ZR("C2H6").Value = self.initialization_point[4]

        hx = self.block("HX")
        hx.T.Spec = "Fixed"
        hx.T.Value = self.initialization_point[5]
        hx.QR.Spec = "Free"

        self.fsheet.TC.Cascade.Value = 0
        self.fsheet.TC.AutoMan.Value = 0
        self.fsheet.EAC.Cascade.Value = 0
        self.fsheet.EAC.AutoMan.Value = 0

        self.sim.RunMode = "Initialization"
        self.sim.Run(True)

        self.fsheet.TC.Cascade.Value = 1
        self.fsheet.TC.AutoMan.Value = 0
        self.fsheet.EAC.Cascade.Value = 1
        self.fsheet.EAC.AutoMan.Value = 0

        self.sim.RunMode = "Steady State"
        self.sim.Run(True)

        for block_name in ["TC", "EAC"]:
            self.fsheet.RemoveBlock(block_name)
        for stream_name in ["S1", "S2", "S4", "S9"]:
            self.fsheet.RemoveStream(stream_name)

        self.col.Reflux.FmR.Value = self.ss_inputs[0]
        self.col.QRebR.Value = self.ss_inputs[1]

    def ss_outputs(self):
        self.initialize_system()
        self.sim.RunMode = "Dynamic"
        self.sim.options.TimeSettings.RecordHistory = False
        self.sim.endtime = 40
        self.sim.run(1)
        return np.array([self.col.Stage(24).y("C2H6").Value, self.col.Stage(85).T.Value], dtype=float)

    def step(self, disturbances=None):
        self.col.Reflux.FmR.Value = float(self.current_input[0])
        self.col.QRebR.Value = float(self.current_input[1])
        self.sim.Step(True)
        self.current_output = np.array([self.col.Stage(24).y("C2H6").Value, self.col.Stage(85).T.Value], dtype=float)
        if disturbances is not None:
            disturbances = np.atleast_1d(np.asarray(disturbances, dtype=float))
            self.feed.FmR.Value = float(disturbances[0])

    def close(self, snaps_path=None, prefix="snp"):
        if snaps_path:
            snaps_path = Path(snaps_path)
            if snaps_path.exists():
                files = sorted(
                    [path for path in snaps_path.iterdir() if path.is_file() and path.name.startswith(prefix)],
                    key=lambda item: item.stat().st_ctime,
                )
                for path in files:
                    path.unlink(missing_ok=True)
        self.ad.CloseDocument(False)
        self.ad.Quit()


def distillation_system_stepper(system, disturbance_step):
    if disturbance_step is None:
        system.step()
        return
    system.step(disturbances=np.atleast_1d(np.asarray(disturbance_step, dtype=float)))


def build_distillation_system(path, ss_inputs, initialization_point, delta_t=1.0 / 6.0, visible=True):
    return DistillationColumnAspen(
        path=path,
        ss_inputs=ss_inputs,
        initialization_point=initialization_point,
        delta_t=delta_t,
        visible=visible,
    )

