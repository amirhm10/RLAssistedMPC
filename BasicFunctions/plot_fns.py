from datetime import datetime


def save_and_plot_rl_results(y_sp, steady_states, nFE, delta_t, time_in_sub_episodes,
                             y_mpc, u_mpc, avg_rewards, data_min, data_max, directory, prefix_name="agent_result",
                             yhat=None):
    # Create a timestamped directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    directory = os.path.join(directory, prefix_name, timestamp)
    os.makedirs(directory, exist_ok=True)

    # Saving inputs in a pickle file
    input_data = {
        "y_sp": y_sp,
        "steady_states": steady_states,
        "nFE": nFE,
        "delta_t": delta_t,
        "time_in_sub_episodes": time_in_sub_episodes,
        "y_mpc": y_mpc,
        "u_mpc": u_mpc,
        "avg_rewards": avg_rewards,
        "data_min": data_min,
        "data_max": data_max,
        "yhat": yhat
    }

    with open(os.path.join(directory, 'input_data.pkl'), 'wb') as f:
        pickle.dump(input_data, f)

    # Canceling the deviation form
    y_ss = apply_min_max(steady_states["y_ss"], data_min[2:], data_max[2:])
    y_sp = (y_sp + y_ss)
    y_sp = (reverse_min_max(y_sp, data_min[2:], data_max[2:])).T

    time_plot = np.linspace(0, nFE * delta_t, nFE + 1)
    time_plot_hour = np.linspace(0, time_in_sub_episodes * delta_t, time_in_sub_episodes + 1)

    # Plotting functions
    def save_plot(name):
        plt.tight_layout()
        plt.savefig(os.path.join(directory, name), dpi=300)
        plt.close()

    # Plot 1
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(time_plot, y_mpc[:, 0], 'b-', lw=2, label='RL')
    plt.step(time_plot[:-1], y_sp[0, :], 'r--', lw=2, label='Setpoint')
    plt.ylabel('η (L/g)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    plt.subplot(2, 1, 2)
    plt.plot(time_plot, y_mpc[:, 1], 'b-', lw=2, label='RL')
    plt.step(time_plot[:-1], y_sp[1, :], 'r--', lw=2, label='Setpoint')
    plt.ylabel('T (K)', fontsize=18)
    plt.xlabel('Time (hour)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    save_plot('plot_1.png')

    # Last 400 plot
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(time_plot_hour, y_mpc[nFE - time_in_sub_episodes:, 0], 'b-', lw=2, label='RL')
    plt.step(time_plot_hour[:-1], y_sp[0, nFE - time_in_sub_episodes:], 'r--', lw=2, label='Setpoint')
    plt.ylabel('η (L/g)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    plt.subplot(2, 1, 2)
    plt.plot(time_plot_hour, y_mpc[nFE - time_in_sub_episodes:, 1], 'b-', lw=2, label='RL')
    plt.step(time_plot_hour[:-1], y_sp[1, nFE - time_in_sub_episodes:], 'r--', lw=2, label='Setpoint')
    plt.ylabel('T (K)', fontsize=18)
    plt.xlabel('Time (hr)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    save_plot('plot_last_400.png')

    # Plot 2
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.step(time_plot[:-1], u_mpc[:, 0], 'k-', lw=2, label='Qc')
    plt.ylabel('Qc (L/h)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    plt.subplot(2, 1, 2)
    plt.step(time_plot[:-1], u_mpc[:, 1], 'k-', lw=2, label='Qm')
    plt.ylabel('Qm (L/h)', fontsize=18)
    plt.xlabel('Time (hour)', fontsize=18)
    plt.grid(True)
    plt.legend(loc='best', fontsize=16)

    save_plot('plot_2.png')

    # Plot 3 (Reward)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(avg_rewards) + 1), avg_rewards, 'ko-', lw=2, label='Reward per Episode')
    plt.ylabel('Avg. Reward', fontsize=16, fontweight='bold')
    plt.xlabel('Episode #', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='best', fontsize=16)

    save_plot('plot_reward.png')

    # Optional yhat plot
    if yhat is not None:
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))

        y_mpc_scaled_1 = ((y_mpc[:, 0] - steady_states["y_ss"][0])) / (data_max[2] - data_min[2])
        axs[0].plot(yhat[0, :], 'b-', linewidth=2, label='T (Observer)')
        axs[0].plot(y_mpc_scaled_1, 'r--', linewidth=2, label='T (Measurement)')
        axs[0].set_ylabel('Scaled Deviation')
        axs[0].legend()
        axs[0].grid(True)

        y_mpc_scaled_2 = ((y_mpc[:, 1] - steady_states["y_ss"][1])) / (data_max[3] - data_min[3])
        axs[1].plot(yhat[1, :], 'b-', linewidth=2, label='η (Observer)')
        axs[1].plot(y_mpc_scaled_2, 'r--', linewidth=2, label='η (Measurement)')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Scaled Deviation')
        axs[1].legend()
        axs[1].grid(True)

        fig.tight_layout()
        fig.savefig(os.path.join(directory, 'plot_observer.png'), dpi=300)
        plt.close(fig)