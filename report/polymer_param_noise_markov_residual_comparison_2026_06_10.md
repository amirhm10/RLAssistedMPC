# Polymer Markov/Residual Parameter-Noise Check

Date: 2026-06-10

## Scope

This note compares the latest polymer standalone Markov and residual SG-TD3 parameter-noise runs against the previous Gaussian-action-noise references. It reads saved `input_data.pkl` bundles only and does not launch polymer simulations.

## Main Conclusion

The polymer parameter-noise switch looks acceptable. Markov improved clearly relative to the same `z_bound = 0.7` Gaussian reference, and residual improved mildly relative to the June 8 Gaussian reference. The remaining caution is that all polymer reward values are still negative under the saved reward convention, so the decision should be based on relative reward, handoff behavior, tracking, and source diagnostics rather than the sign of reward.

## Summary Table

| Family | Variant | Tail reward | Delta vs OF-MPC | Worst post-warm | First-live reward | Tail policy frac | Tail authority | Tail gate adv | Tail y1 MAE | Tail y2 MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OF-MPC | disturbed baseline | -3.808 | NA | -4.384 | -4.338 | NA | NA | NA | 3.679 | 323.913 |
| Markov | Gaussian reference | -2.904 | 0.904 | -7.712 | -3.815 | 0.134 | 0.156 | 1.491 | 3.685 | 323.901 |
| Markov | Parameter-noise latest | -2.857 | 0.951 | -4.714 | -4.023 | 0.395 | 0.304 | 0.871 | 3.684 | 323.903 |
| Residual | Gaussian reference | -2.985 | 0.823 | -11.111 | -4.904 | 0.058 | 0.019 | 0.603 | 3.692 | 323.865 |
| Residual | Parameter-noise latest | -2.935 | 0.874 | -7.106 | -4.212 | 0.068 | 0.023 | 0.602 | 3.680 | 323.952 |

## Interpretation

- Markov param-noise tail reward is `-2.857`, compared with `-2.904` for the Gaussian `z_bound = 0.7` reference and `-3.808` for OF-MPC.
- Markov worst post-warm reward improves from `-7.712` to `-4.714`. This is the strongest safety signal in favor of the parameter-noise rerun.
- Markov tail policy execution increases from `0.134` to `0.395`, while tail reward and worst-postwarm reward both improve. That suggests the parameter-noise actor was more usable for the SG gate, not merely more aggressive.
- Residual param-noise tail reward is `-2.935`, compared with `-2.985` for the Gaussian reference. The improvement is modest but positive.
- Residual worst post-warm reward improves from `-11.111` to `-7.106`, but the latest residual run is still not as clean as the best older non-mismatch residual family. Treat it as acceptable, not a dramatic win.

## Recommendation

- Keep polymer Markov and residual parameter noise `0.10 -> 0.02` as the default for the next combined/standalone checks.
- For Markov, the result supports using parameter noise in the final standalone table because it improves tail reward and reduces the worst handoff/post-warm downside versus the previous Gaussian `z_bound = 0.7` run.
- For residual, parameter noise is acceptable and slightly better than the immediate Gaussian reference, but the claim should be weaker: it is a polish improvement, not a new best mechanism by itself.
- In the paper, report both the standalone Gaussian reference and parameter-noise final rows, because this is a useful example where temporally coherent exploration improves the high-authority Markov block without removing the SG envelope.

## Artifacts

- `report/figures/polymer_param_noise_markov_residual_20260610/polymer_param_noise_markov_residual_summary.csv`
- `report/figures/polymer_param_noise_markov_residual_20260610/fig_polymer_param_noise_reward_histories.png`
- `report/figures/polymer_param_noise_markov_residual_20260610/fig_polymer_param_noise_metric_summary.png`

## Files Inspected

- `Polymer/Results/mpc_offsetfree_disturb_unified/20260608_143324/input_data.pkl`
- `Polymer/Results/sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_mismatch/20260609_220217/input_data.pkl`
- `Polymer/Results/sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_mismatch/20260610_170822/input_data.pkl`
- `Polymer/Results/sg_td3_residual_critic_warm3_conservative_disturb_mismatch/20260608_152126/input_data.pkl`
- `Polymer/Results/sg_td3_residual_critic_warm3_conservative_disturb_mismatch/20260610_131706/input_data.pkl`
