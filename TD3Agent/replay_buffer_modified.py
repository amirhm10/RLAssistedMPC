import numpy as np
import torch


# ----------------------
# PER Recent Replay Buffer
# ----------------------
class PERRecentReplayBuffer:
    def __init__(
            self,
            capacity: int,
            state_dim: int,
            action_dim: int,
            eps: float = 1e-6,
            alpha: float = 0.4,  # PER exponent (0 = uniform, 1 = full PER)
            beta_start: float = 0.4,  # IS correction starts small, anneals to 1.0
            beta_end: float = 1.0,
            beta_steps: int = 50_000,  # steps over which beta anneals
            per_lam_age=0.0,
            per_tau_use=0.0,
            rec_lam_age=0.0,
            rec_tau_use=0.0,
            u_lam_age=0.0,
            u_tau_use=0.0,
    ):

        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.ptr = 0
        self.size = 0
        self.step_counter = 0

        self.states = np.zeros((self.capacity, self.state_dim), np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), np.float32)
        self.rewards = np.zeros((self.capacity,), np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), np.float32)
        self.dones = np.zeros((self.capacity,), np.float32)  # 0/1 floats

        # Birth and Use count
        self.birth_step = np.zeros(capacity, np.int64)
        self.use_count = np.zeros(capacity, np.int64)

        # PER
        self.priorities = np.zeros(self.capacity, np.float32)
        self.eps = eps
        self.alpha = alpha

        # Importance Weight Sampling params
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.beta_t = 0

        # max priority
        self._max_priority = 1.0

        # probability values
        self.per_lam_age = float(per_lam_age)
        self.per_tau_use = float(per_tau_use)
        self.rec_lam_age = float(rec_lam_age)
        self.rec_tau_use = float(rec_tau_use)
        self.u_lam_age = float(u_lam_age)
        self.u_tau_use = float(u_tau_use)

    # ----- helpers ------
    def _beta(self):
        frac = min(1.0, self.beta_t / max(1, self.beta_steps))
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def push(self, s, a, r, ns, done, p0=None):
        i = self.ptr
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.next_states[i] = ns
        self.dones[i] = float(done)

        pri = float(p0) if (p0 is not None and p0 > 0) else self._max_priority
        self.priorities[i] = pri
        self._max_priority = max(self._max_priority, pri)

        self.birth_step[i] = self.step_counter
        self.use_count[i] = 0

        self.step_counter += 1
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
            self,
            batch_size: int,
            device="cpu",
            frac_per: float = 0.2,
            frac_recent: float = 0.6,
            recent_window=5000,
            w_clip_min=1e-3,
            w_clip_max=1e3
    ):

        assert self.size > 0

        # fraction of each method
        k_per = int(batch_size * frac_per)
        k_recent = int(batch_size * frac_recent)
        k_uniform = batch_size - k_per - k_recent

        B = batch_size
        f_per = k_per / max(1, B)
        f_rec = k_recent / max(1, B)
        f_uni = k_uniform / max(1, B)

        N = self.size
        idx_all = np.arange(N)

        # recent pool R
        recent_window = min(N, int(recent_window))
        if recent_window <= 0:
            recent_window = N
        cutoff = np.partition(self.birth_step[:N], -recent_window)[-recent_window]
        pool_recent = idx_all[self.birth_step[:N] >= cutoff]
        if pool_recent.size == 0:
            pool_recent = idx_all

        # compute age/use arrays (vectorize)
        birth = self.birth_step[:N]
        use = self.use_count[:N]
        age = (self.step_counter - birth).astype(np.float32)

        # ---- PER probabilities (non-stationary aware) ----
        if k_per > 0:
            pr = np.maximum(self.priorities[:N], self.eps).astype(np.float32)
            per_pen = np.exp(-self.per_lam_age * age - self.per_tau_use * use.astype(np.float32))
            score = (pr ** self.alpha) * per_pen
            ssum = float(score.sum())
            if not np.isfinite(ssum) or ssum <= 0.0:
                probs = np.full(N, 1.0/N, dtype=np.float32)
            else:
                probs = (score / ssum).astype(np.float32)

            idx_per = np.random.choice(N, size=k_per, replace=True, p=probs)
        else:
            probs=None
            idx_per = np.array([], dtype=np.int64)

        # ---- Recent probabilities (non-uniform) ----
        if k_recent > 0:
            pr_idx = pool_recent
            age_r = age[pr_idx]
            use_r = use[pr_idx].astype(np.float32)
            w_r = np.exp(-self.rec_lam_age * age_r - self.rec_tau_use * use_r)
            wsum = float(w_r.sum())
            if not np.isfinite(wsum) or wsum <= 0.0:
                p_r = np.full(pr_idx.size, 1.0 / pr_idx.size, dtype=np.float32)
            else:
                p_r = (w_r / wsum).astype(np.float32)

            idx_recent = np.random.choice(pr_idx, size=k_recent, replace=(pr_idx.size < k_recent), p=p_r)

            # to get P_rec(idx) fast for sampled idx, sort pool_recent once
            order = np.argsort(pr_idx)
            pr_sorted = pr_idx[order]
            p_r_sorted = p_r[order]

        else:
            idx_recent = np.array([], dtype=np.int64)
            pr_sorted = None
            p_r_sorted = None

        # ---- Uniform -----
        if k_uniform > 0:
            idx_uni = np.random.choice(N, size=k_uniform, replace=True)
        else:
            idx_uni = np.array([], dtype=np.int64)

        # merge
        idx = np.concatenate([idx_per, idx_recent, idx_uni]).astype(np.int64)

        # ---- compute q(idx) for importance weight ----
        q = np.full(idx.shape[0], f_uni * (1.0 / N), dtype=np.float32)

        if k_per > 0 and probs is not None:
            q += f_per * probs[idx]

        if k_recent > 0 and pr_sorted is not None:
            pos = np.searchsorted(pr_sorted, idx)
            p_rec_idx = np.zeros(idx.shape[0], dtype=np.float32)
            valid = pos < pr_sorted.size
            if np.any(valid):
                posv = pos[valid]
                idxv = idx[valid]
                ok = pr_sorted[posv] == idxv
                if np.any(ok):
                    vv = np.where(valid)[0][ok]
                    p_rec_idx[vv] = p_r_sorted[pos[vv]]
            q += f_rec * p_rec_idx

        q = np.maximum(q, 1e-12)

        # ---- target tilde_u(idx) ----
        age_idx = age[idx]
        use_idx = use[idx].astype(np.float32)
        u_tilde = np.exp(-self.u_lam_age * age_idx - self.u_tau_use * use_idx).astype(np.float32)
        u_tilde = np.maximum(u_tilde, 1e-12)

        beta = self._beta()
        self.beta_t += 1

        w = (u_tilde / q) ** beta
        w = np.clip(w, w_clip_min, w_clip_max)
        w = w / max(1e-12, float(w.max()))
        is_w = w.astype(np.float32)

        # update counts training usage
        np.add.at(self.use_count, idx, 1)

        # tensors
        s = torch.from_numpy(self.states[idx]).to(device)
        a = torch.from_numpy(self.actions[idx]).to(device)
        r = torch.from_numpy(self.rewards[idx]).to(device)
        ns = torch.from_numpy(self.next_states[idx]).to(device)
        d = torch.from_numpy(self.dones[idx]).to(device)
        is_w = torch.from_numpy(is_w).to(device)

        return s, a, r, ns, d, idx, is_w

    def update_priorities(self, idx, td_errors):
        if isinstance(td_errors, torch.Tensor):
            td = td_errors.detach().abs().view(-1).cpu().numpy()
        else:
            td = np.abs(td_errors).reshape(-1)

        p = td + self.eps
        p = np.clip(p, 1e-4, 1e4).astype(np.float32)

        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        if idx.size != p.size:
            p = p.reshape(-1)[:idx.size]

        np.maximum.at(self.priorities, idx, p)
        self._max_priority = max(self._max_priority, float(p.max()))

    def __len__(self):
        return self.size
