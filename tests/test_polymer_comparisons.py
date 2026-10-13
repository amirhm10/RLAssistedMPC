"""Scientific contracts for isolated Scenario1 comparators (no production runs)."""
from __future__ import annotations
from copy import deepcopy
import pickle
from pathlib import Path
import unittest
import numpy as np

from comparison_studies.arx import IncrementalARX,nominal_coefficients
from comparison_studies.bayes import tune
from comparison_studies.common import ROOT,PlantTiming,compress_bundle,window_bundle
from comparison_studies.threshold import calibrate,normalized_error,acceptance_override
from comparison_studies.analysis import trajectories
from unittest.mock import patch


class ComparatorMathTests(unittest.TestCase):
    def test_identification_callback_is_in_controller_time_not_plant_time(self):
        class Plant:
            def step(self):
                pass
        with patch("comparison_studies.common.time.perf_counter",side_effect=[0.,1.,2.,4.,5.]):
            timing = PlantTiming()
            plant = timing.attach(Plant(),lambda p:None)
            plant.step();plant.step()
        self.assertEqual(timing.control_seconds,[1.,2.])
        self.assertEqual(timing.plant_seconds,[1.,1.])

    def test_canonical_baseline_inputs_are_physical(self):
        with (ROOT/"Polymer/Data/mpc_results_dist.pickle").open("rb") as stream:
            data = pickle.load(stream)
        _,u,_,_ = trajectories(data)
        previous = np.vstack([data["steady_states"]["ss_inputs"],u[:-1]])
        np.testing.assert_allclose((u-previous)/(data["data_max"][:2]-data["data_min"][:2]),
                                   data["delta_u_storage"],rtol=1e-7,atol=1e-8)
    def test_nominal_arx_reproduces_state_space_with_delayed_channel(self):
        with (ROOT/"Polymer/Data/system_dict").open("rb") as stream:
            data = pickle.load(stream)
        A,B,C = [np.asarray(data[k]) for k in ("A","B","C")]
        predictor = IncrementalARX(nominal_coefficients(A,B,C),np.zeros(2),np.zeros(2),adaptive=False)
        x = np.zeros(A.shape[0])
        rng = np.random.default_rng(7)
        inputs = rng.uniform(-.5,.5,(160,2))
        for k,u in enumerate(inputs[:-20]):
            forecast = predictor.predict(inputs[k:k+20],20)
            xx = x.copy();expected=[]
            for proposed in inputs[k:k+20]:
                xx = A@xx+B@proposed
                expected.append(C@xx)
            np.testing.assert_allclose(forecast,expected,rtol=2e-7,atol=2e-10)
            x = A@x+B@u
            predictor.observe(C@x,u)

    def test_rls_recovers_known_mimo_model(self):
        theta = [np.array([.4,.1,.2,.1,-.08,.3,-.1]),np.array([.3,.15,-.2,.04,.25,.09])]
        true = IncrementalARX(theta,np.zeros(2),np.zeros(2),adaptive=False)
        initial = [np.r_[t[:2]*.9,t[2:]*.5] for t in theta]
        fitted = IncrementalARX(initial,np.zeros(2),np.zeros(2))
        rng = np.random.default_rng(12)
        for _ in range(1600):
            u = rng.uniform(-.5,.5,2)
            y = true.predict(u.reshape(1,2),1)[0]
            true.observe(y,u);fitted.observe(y,u)
        for expected,actual in zip(theta,fitted.theta):
            np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-4)
        for P in fitted.covariance:
            self.assertTrue(np.min(np.linalg.eigvalsh(P)) > 0)

    def test_rejects_unstable_and_nonfinite_updates(self):
        self.assertFalse(IncrementalARX.valid(np.array([2.,0.,.1])))
        self.assertFalse(IncrementalARX.valid(np.array([np.nan,0.,.1])))
        self.assertTrue(IncrementalARX.valid(np.array([.4,.1,.1])))

    def test_candidate_prediction_does_not_leak_into_rls_history(self):
        theta = [np.array([.4,.1,.2,.1,-.08,.3,-.1]),np.array([.3,.15,-.2,.04,.25,.09])]
        predictor = IncrementalARX(theta,np.zeros(2),np.zeros(2))
        before = deepcopy(predictor.__dict__)
        predictor.predict(np.ones((3,2)),9)
        np.testing.assert_array_equal(predictor.current_y,before["current_y"])
        np.testing.assert_array_equal(predictor.du,before["du"])
        self.assertEqual(predictor.samples,0)

    def test_threshold_units_calibration_equality_and_nonfinite(self):
        params = {"k_rel":np.array([.003,.0003]),"band_floor_phys":np.array([.006,.07])}
        reference = np.array([4.5,324.])
        bands = params["k_rel"]*reference
        self.assertAlmostEqual(normalized_error(reference+bands,reference,params),1.)
        self.assertEqual(normalized_error([np.nan,324.],reference,params),float("inf"))
        ys = np.stack([reference,reference+bands])
        self.assertAlmostEqual(calibrate(ys,np.stack([reference,reference]),params),1.1875)
        ctx = {"error":1.,"threshold":1.,"calls":0}
        import TD3Agent.supervisor_gated_agent as td3
        original = td3.choose_policy_or_supervisor
        with acceptance_override(ctx):
            self.assertEqual(td3.choose_policy_or_supervisor(-1e10,1e10),"policy")
            ctx["error"] = 1.01
            self.assertEqual(td3.choose_policy_or_supervisor(1e10,-1e10),"supervisor")
            ctx.update(eligible=False,error=0.)
            self.assertEqual(td3.choose_policy_or_supervisor(-1e10,1e10),"supervisor")
        self.assertIs(td3.choose_policy_or_supervisor,original)

    def test_bo_budget_bounds_failure_exclusion_and_reproducibility(self):
        def evaluate(alpha,beta):
            return (alpha-1.3)**2+(beta-.9)**2,{"valid":not(alpha==.75 and beta==.75)}
        best,a = tune(evaluate,seed=7,trials=7)
        same,b = tune(evaluate,seed=7,trials=7)
        self.assertEqual(a,b);self.assertEqual(best,same)
        self.assertEqual(len(a),7)
        self.assertFalse(a[1]["valid"])
        self.assertTrue(best["valid"])
        self.assertTrue(all(.75<=r["alpha"]<=2 and .75<=r["beta"]<=2 for r in a))

    def test_calibration_uses_original_ramp_indices(self):
        full = {"nFE":160000,"time_in_sub_episodes":800,"warm_start_step":8000,
                "y_sp":np.zeros((160000,2)),"qi":np.linspace(108.,91.8,160000),
                "qs":np.linspace(459.,596.7,160000),"ha":np.arange(160000.),
                "sub_episodes_changes_dict":{k:k//800+1 for k in range(799,160000,800)},
                "sub_episode_changes_dict":{k:k//800+1 for k in range(799,160000,800)},
                "test_train_dict":{k:k==159200 for k in range(0,160000,800)}}
        trial = window_bundle(full,8000,9600)
        np.testing.assert_array_equal(trial["qi"],full["qi"][8000:9600])
        self.assertNotAlmostEqual(trial["qi"][-1],91.8)
        smoke = compress_bundle(full)
        self.assertEqual(smoke["nFE"],800)
        self.assertEqual(smoke["warm_start_step"],40)
        self.assertTrue(smoke["test_train_dict"][796])
        self.assertEqual(full["nFE"],160000)
        full["phase_switch_step"] = None
        self.assertIsNone(compress_bundle(full)["phase_switch_step"])


if __name__ == "__main__":
    unittest.main()
