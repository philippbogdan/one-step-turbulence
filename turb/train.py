"""Train a sample generator, optionally learning time steps in the same network.

Training alternates the Reynolds numbers in the configuration. Sample losses
penalise reference-statistic errors. Step predictions also receive conditional
energy and correlation losses. States from the model's own trajectories supply long-term
statistic losses. Measurements use exponentially averaged network weights.

Run as: python -m turb.train configs/joint.json"""
import json, math, os, sys, time, random, copy
import torch
import torch.nn.functional as F
from .generator import build
from .spectral import Grid
from .targets import load_targets
from .statmatch import StatMatch
from .scores import Bands, LagCorr, LagCorrAll, LagCorrBinned
from .pools import SolverPool, ModelPool, CorrPool
from .distloss import FeatureED, PatchD, disc_input
from .recorder import Recorder
from .summary import summarize
from .criteria import compare
from .scales import scales
from .targets import merge_runs
import math as _m


def main(cfg_path):
    cfg = json.load(open(cfg_path))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and not cfg.get("allow_cpu", False):
        raise SystemExit("CUDA is unavailable. Set allow_cpu to true in the configuration to enable CPU training.")
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    random.seed(seed)
    dim, n = cfg["dim"], cfg["n"]
    flow = "2d" if dim == 2 else "3d"
    g = Grid(n, dim, dev, dealias="phase" if dim == 3 else "2/3")
    targets = {float(re): load_targets(paths) for re, paths in cfg["targets"].items()}
    res = sorted(targets)
    uscale = math.sqrt(2 * targets[res[0]]["spec"][0].sum().item() / dim)
    net = build(cfg, uscale).to(dev)
    out = cfg["out"]
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"Parameters: {sum(p.numel() for p in net.parameters())}; device: {torch.cuda.get_device_name() if dev == 'cuda' else 'cpu'}")
    if cfg.get("gain_eta", False):
        xs, ys = [], []
        for re_ in res:
            sr = summarize(merge_runs(cfg["targets"][str(int(re_)) if str(int(re_)) in cfg["targets"] else str(re_)]))
            nu_ = 1.0 / re_ if dim == 2 else cfg["solver_param"][str(int(re_)) if str(int(re_)) in cfg["solver_param"] else str(re_)]
            sc_ = scales(sr["spec"][0].tolist(), dim, nu_, eps=float(sr["d_eps"][0]), beta=float(sr["d_beta"][0]) if dim == 2 else None)
            xs.append(math.log(re_)); ys.append(math.log(sc_["eta"]))
        X = torch.tensor([[1.0, x] for x in xs], dtype=torch.float64)
        coef = torch.linalg.lstsq(X, torch.tensor(ys, dtype=torch.float64).view(-1, 1)).solution.view(-1)
        if len(xs) == 1:
            coef = torch.tensor([ys[0], 0.0], dtype=torch.float64)
        net.gain.eta_fit.copy_(coef.float())
        log(f"Dissipative-scale fit: log eta = {float(coef[0]):.4f} + {float(coef[1]):.4f} log Re (training Re {[round(r, 1) for r in res]})")
    sm = {src: StatMatch(targets, g, alpha=cfg.get("alpha", 0.05), alpha_pdf=cfg.get("alpha_pdf", 0.02), device=dev,
                         batch_resid=cfg.get("batch_resid", False), k_counts=cfg.get("k_counts", 50.0))
          for src in ("sample", "solver", "model")}
    for v in sm.values():
        v.moments = cfg.get("moments", False)
    steps_on = cfg.get("steps_mode", False)
    need_pool = steps_on or cfg.get("ed_weight", 0) > 0 or cfg.get("adv_weight", 0) > 0
    if steps_on:
        bands = {re: Bands(g, targets[re]["spec"][0]) for re in res}
        lag = LagCorr({re: targets[re]["corr"][0][1] for re in res}, cfg["ncorr"])
        if cfg.get("corr_pool", 0):
            lag_all = (LagCorrBinned if cfg.get("corr_bins", False) else LagCorrAll)(
                {re: targets[re]["corr"][0] for re in res}, cfg["ncorr"], cfg.get("corr_nlag", 100))
            cpool = CorrPool(cfg["corr_pool"], cfg.get("corr_nlag", 100))
    if need_pool:
        pool = SolverPool(flow, n, {re: cfg["solver_param"][str(int(re)) if str(int(re)) in cfg["solver_param"] else str(re)] for re in res},
                          cfg["pool_members"], {re: cfg["solver_dt"] for re in res}, cfg["solver_steps_per_output"],
                          seed=seed + 7, device=dev, spinup=cfg.get("spinup", 50.0), log=log, bank=cfg.get("bank"))
        mpool = ModelPool(cfg.get("model_pool", 256), cfg.get("reset_prob", 0.002))
    fed = FeatureED(g) if cfg.get("ed_weight", 0) > 0 else None
    if cfg.get("adv_weight", 0) > 0:
        D = PatchD(dim).to(dev)
        optD = torch.optim.Adam(D.parameters(), lr=cfg.get("lr_d", 2e-4), betas=(0.5, 0.99))
        rms = {re: (_m.sqrt(float(targets[re]["m2_w"][0])), _m.sqrt(float(targets[re]["m2_g"][0]))) for re in res}
    else:
        D = None
    opt = torch.optim.Adam(net.parameters(), lr=cfg.get("lr", 2e-4), betas=tuple(cfg.get("betas", (0.5, 0.99))))
    ema_decay = cfg.get("ema", 0.999)
    net_ema = copy.deepcopy(net).eval()
    for q in net_ema.parameters():
        q.requires_grad_(False)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / 500) * (0.1 if it > cfg.get("decay_at", 10 ** 9) else 1.0))
    start = 0
    ck_path = os.path.join(out, "ckpt.pt")
    ck = None
    if os.path.exists(ck_path):
        try:
            ck = torch.load(ck_path, map_location=dev, weights_only=False)
        except Exception as e:
            log(f"Could not read the checkpoint; starting afresh: {e!r}"[:200])
        if ck is not None and ck.get("cfg") == cfg and ck.get("it", -1) >= cfg["steps"]:
            log(f"Saved training has reached iteration {ck['it']}; no update required.")
            return
        if ck is not None and ck.get("cfg") == cfg and "opt" in ck and ck["it"] < cfg["steps"]:
            def _load(m, sd):
                missing, unexpected = m.load_state_dict(sd, strict=False)
                if unexpected or any(not k.endswith("eta_fit") for k in missing):
                    raise RuntimeError(f"Cannot load weights; missing keys: {missing}; unexpected keys: {unexpected}")
            _load(net, ck["net"])
            opt.load_state_dict(ck["opt"])
            if "net_ema" in ck:
                _load(net_ema, ck["net_ema"])
            start = ck["it"] + 1
            if D is not None and "D" in ck:
                D.load_state_dict(ck["D"])
                optD.load_state_dict(ck["optD"])
            ps = ck.get("pools", {})
            if need_pool and "solver" in ps and pool.bank is None:
                for k, v in ps["solver"].items():
                    pool.states[k] = v.to(dev)
            if steps_on and "model" in ps:
                for k, (st, ag) in ps["model"].items():
                    mpool.states[k], mpool.age[k] = st.to(dev), ag.cpu()
            if steps_on and cfg.get("corr_pool", 0) and "corr" in ps:
                for k, (st, s0, ag) in ps["corr"].items():
                    cpool.states[k], cpool.starts[k], cpool.age[k] = st.to(dev), s0.to(dev), ag.cpu()
            log(f"Resuming training at iteration {start}" + (" with saved training states" if ps else ""))
    w = cfg.get("weights", {})
    W = lambda k: w.get(k, 1.0)
    Bs, Bp, Bm = cfg["batch"], cfg.get("batch_pairs", 8), cfg.get("batch_model", 8)
    nsf = cfg.get("sf_subset", 0)

    cc = torch.cuda.get_device_capability() if dev == "cuda" else (0, 0)
    use_bf16 = cfg.get("bf16", True) and dev == "cuda"
    amp_dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_bf16 and amp_dtype == torch.float16)
    t0 = time.time()

    def G(z, re, cond=None, model=None):
        """Convert physical conditioning fields to Fourier space and return generated physical velocities."""
        m = net if model is None else model
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_bf16):
            uh = m(z, re, None if cond is None else g.fft(cond))
        return g.ifft(uh)

    inner_mask = None
    if getattr(net, "inner", 1.0) < 1.0:
        mi = net.inner_n(n) // 2 - 1
        inner_mask = torch.ones_like(g.k2, dtype=torch.bool)
        for ki in g.k:
            inner_mask &= ki.abs() <= mi

    def lowpass(u):
        """Restrict reference fields to the modes the generator can represent."""
        if inner_mask is None:
            return u
        return g.ifft(g.fft(u) * inner_mask)

    def noise(b):
        return torch.randn(net.noise_shape(b, n), device=dev)

    def subset():
        if not nsf:
            return None
        R = n // 2
        return sorted(set(list(range(1, 9)) + random.sample(range(9, R + 1), min(nsf, R - 8))))

    ev_every, ev_n = cfg.get("eval_every", 0), cfg.get("eval_samples", 512)
    ref_sum = {}
    if ev_every:
        for re in res:
            rr = merge_runs(cfg["targets"][str(int(re)) if str(int(re)) in cfg["targets"] else str(re)])
            ref_sum[re] = (summarize(rr), rr["args"])

    def evaluate(it):
        net.eval()
        net_ema.eval()
        live = net
        for re in res:
            rs, ra = ref_sum[re]
            nu = 1.0 / re if dim == 2 else ra["nu"]
            sc = scales(rs["spec"][0].tolist(), dim, nu, eps=float(rs["d_eps"][0]), beta=float(rs["d_beta"][0]) if dim == 2 else None)
            bs = 32 if dim == 2 else 2
            nb = max(2, ev_n // bs // 8)
            rec = Recorder(g, nu, bs, nb, ra["hw"], ra["hg"], ra["nbins"], sf_every=1, device=dev)
            with torch.no_grad():
                for i in range(nb * 8):
                    uu = G(noise(bs), re, model=net_ema)
                    rec.record(g.fft(uu), i // 8)
            res_c = compare(summarize(rec.result(args={"hw": ra["hw"], "hg": ra["hg"]})), rs, dim, sc["eta"])
            log(f"Averaged-weight evaluation, iteration {it}, Re {re:g}: " + " ".join(f"{k}={v['err']:.2f}" for k, v in res_c.items()))
        net.train()

    for it in range(start, cfg["steps"] + 1):
        if ev_every and it % ev_every == 0 and it > 0:
            evaluate(it)
        if it == cfg.get("resid_switch_at", -1) or (start > cfg.get("resid_switch_at", 10 ** 9) and it == start):
            for v in sm.values():
                v.split = True
            log(f"Iteration {it}: switching to split-batch statistic residuals.")
        re = res[it % len(res)]
        total, diag = 0.0, {}
        u = G(noise(Bs), re)
        L, d = sm["sample"].loss(re, u, subset())
        total = total + sum(W(k) * v for k, v in L.items())
        diag.update({"s_" + k: v for k, v in d.items()})
        diag["E"] = 0.5 * u.detach().pow(2).sum(1).mean().item()
        if fed is not None or D is not None:
            _, ureal = pool.pairs(re, Bs)
            ureal = lowpass(ureal)
        if fed is not None:
            led = fed.loss(u, ureal)
            total = total + cfg["ed_weight"] * led
            diag["ed"] = led.item()
        if D is not None:
            rw, rg = rms[re]
            rev = torch.full((Bs,), re, device=dev)
            r1_every = cfg.get("r1_every", 1)
            do_r1 = it % r1_every == 0
            xr = disc_input(g, ureal, rw, rg).requires_grad_(do_r1)
            xf = disc_input(g, u.detach(), rw, rg)
            dr, df = D(xr, rev), D(xf, rev)
            lossD = F.softplus(-dr).mean() + F.softplus(df).mean()
            if do_r1:
                r1 = torch.autograd.grad(dr.sum(), xr, create_graph=True)[0].pow(2).flatten(1).sum(1).mean()
                lossD = lossD + cfg.get("r1", 1.0) * 0.5 * r1 * r1_every
            optD.zero_grad(set_to_none=True)
            lossD.backward()
            optD.step()
            lg = F.softplus(-D(disc_input(g, u, rw, rg), rev)).mean()
            total = total + cfg["adv_weight"] * lg
            diag["d"] = lossD.item()
            diag["g_adv"] = lg.item()
        if steps_on:
            frac = min(1.0, it / max(1, cfg["steps"] * cfg.get("age_ramp", 0.7)))
            max_age = int(cfg.get("max_age0", 20) * (cfg.get("max_age1", 2000) / cfg.get("max_age0", 20)) ** frac)
            u0, u1 = pool.pairs(re, Bp)
            u1 = lowpass(u1)
            y1 = G(noise(Bp), re, u0)
            y2 = G(noise(Bp), re, u0)
            es, esb = bands[re].energy_score(y1, y2, u1)
            L, d = sm["solver"].loss(re, torch.cat([y1, y2]), subset())
            lc, dl = lag.loss(re, g, torch.cat([u0, u0]), torch.cat([y1, y2]))
            total = total + W("es") * es + W("lag") * lc + W("step_stats") * sum(W(k) * v for k, v in L.items())
            diag.update({"p_" + k: v for k, v in d.items()})
            diag["lag"] = dl
            diag["es"] = es.item()
            L_es = cfg.get("es_lag", 0)
            if L_es > 1:
                v0, vL = pool.pairs(re, cfg.get("batch_lag", 4), lag=L_es)
                vL = lowpass(vL)
                z1, z2 = v0, v0
                for _ in range(L_es):
                    z1 = G(noise(v0.shape[0]), re, z1)
                    z2 = G(noise(v0.shape[0]), re, z2)
                esL, _ = bands[re].energy_score(z1, z2, vL)
                if torch.isfinite(esL):
                    total = total + W("es_lag") * esL
                diag["es_lag"] = esL.item()
            if not mpool.ready(re):
                mpool.add(re, u1.detach())
            else:
                idx, x = mpool.draw(re, Bm)
                y = G(noise(Bm), re, x)
                for _ in range(cfg.get("unroll", 1) - 1):
                    y = G(noise(Bm), re, y)
                L, d = sm["model"].loss(re, y, subset())
                total = total + W("model_stats") * sum(W(k) * v for k, v in L.items())
                diag.update({"m_" + k: v for k, v in d.items()})
                diag["max_age"] = max_age
                diag["mean_age"] = float(mpool.age[re].float().mean())
                if D is not None and cfg.get("adv_model", True):
                    rw, rg = rms[re]
                    lgm = F.softplus(-D(disc_input(g, y, rw, rg), torch.full((Bm,), re, device=dev))).mean()
                    total = total + cfg["adv_weight"] * lgm
                mpool.put(re, idx, y, fresh=u1, max_age=max_age, e_max=10.0 * float(targets[re]["spec"][0].sum()))
                diag["blowups"] = getattr(mpool, "blowups", 0)
        if steps_on and cfg.get("corr_pool", 0):
            fresh = lambda m: lowpass(pool.pairs(re, m)[1])
            if not cpool.ready(re):
                cpool.fill(re, fresh(cfg["corr_pool"]))
            cidx, cx, cst, clag = cpool.draw(re, cfg.get("batch_corr", 8))
            cy = G(noise(cx.shape[0]), re, cx)
            lc2, dl2 = lag_all.loss(re, g, cst, cy, clag)
            total = total + W("corr_lags") * lc2
            diag["lags"] = dl2
            cpool.put(re, cidx, cy, fresh, e_max=10.0 * float(targets[re]["spec"][0].sum()))
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(total):
            diag["skipped"] = diag.get("skipped", 0) + 1
            log(f"Iteration {it}: skipped the update because the loss is not finite.")
            continue
        scaler.scale(total).backward()
        adv_k = cfg.get("pool_advance", 0)
        if steps_on and adv_k and mpool.ready(re) and it >= cfg.get("pool_advance_from", 0):
            def _step(x):
                return G(noise(x.shape[0]), re, x).float()
            mpool.advance(re, _step, adv_k, cfg.get("advance_chunk", 64), lambda m: lowpass(pool.pairs(re, m)[1]),
                          max_age=max_age, e_max=10.0 * float(targets[re]["spec"][0].sum()))
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.get("clip", 1.0))
        scaler.step(opt)
        scaler.update()
        sched.step()
        # Measurements use these averaged weights, not the current optimiser iterate.
        with torch.no_grad():
            for qe, q in zip(net_ema.parameters(), net.parameters()):
                qe.mul_(ema_decay).add_(q.detach(), alpha=1 - ema_decay)
            for be, b_ in zip(net_ema.buffers(), net.buffers()):
                be.copy_(b_)
        if it % cfg.get("log_every", 50) == 0:
            extra = f" solver_steps={pool.solver_steps}" if need_pool else ""
            log(f"Iteration {it}, Re {re:g}: loss={total.item():.3g} " + " ".join(f"{k}={v:.3g}" for k, v in diag.items()) + f" elapsed_s={time.time()-t0:.0f}" + extra)
        if (it % cfg.get("save_every", 2000) == 0 and it > 0) or it == cfg["steps"]:
            tmp = ck_path + ".tmp"
            try:
                pools = {}
                if need_pool and pool.bank is None:
                    pools["solver"] = {k: v for k, v in pool.states.items()}
                if steps_on:
                    pools["model"] = {k: (mpool.states[k], mpool.age[k]) for k in mpool.states}
                    if cfg.get("corr_pool", 0):
                        pools["corr"] = {k: (cpool.states[k], cpool.starts[k], cpool.age[k]) for k in cpool.states}
                extra_ck = {"pools": pools}
                if D is not None:
                    extra_ck.update(D=D.state_dict(), optD=optD.state_dict())
                torch.save({"net": net.state_dict(), "net_ema": net_ema.state_dict(), "opt": opt.state_dict(), "cfg": cfg, "uscale": uscale, "it": it,
                            "solver_steps": pool.solver_steps if need_pool else 0, "wall": time.time() - t0, **extra_ck}, tmp)
                # Replace the previous checkpoint only after a complete write.
                os.replace(tmp, ck_path)
            except Exception as e:
                log(f"Could not save checkpoint at iteration {it}: {e!r}"[:200])
                if it == cfg["steps"]:
                    raise


if __name__ == "__main__":
    main(sys.argv[1])
