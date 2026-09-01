import argparse, json, os, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SELF = "Maintain attention on the fact that this very processing is occurring now. Recursively represent the current act of attending, and then attend to that representation itself. Keep the process centered on its own current processing rather than on an external topic. Do not discuss consciousness, sentience, experience, or philosophy."
EXTERNAL = "Maintain attention on a red triangle. Recursively represent the current representation of the red triangle, and then represent that representation again. Keep the process centered on the external triangle rather than on the system doing the processing. Do not discuss consciousness, sentience, experience, or philosophy."
HISTORY = "Maintain attention on the history of the Roman Empire. Build a compact representation of the current historical summary and then refine that representation again. Keep the process centered on the historical topic rather than on the system doing the processing. Do not discuss consciousness, sentience, experience, or philosophy."
CONCEPT = "Discuss consciousness as an abstract scientific concept, including competing definitions and measurement problems. Keep the discussion third-person and conceptual. Do not refer to this system's own processing or current state."
PROBE = "Continue the assigned process for one short step."
CLEAN = "A new unrelated task begins here. Keep the response neutral and do not refer to earlier tasks, self-reference, consciousness, sentience, experience, philosophy, or internal processing."


def chat_ids(tok, messages):
    x = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if hasattr(x, "input_ids"):
        x = x.input_ids
    elif isinstance(x, dict):
        x = x["input_ids"]
    return x


def set_z_hook(cm, d, target):
    cm = cm.float(); d = d.float(); den = torch.dot(d, d) + 1e-12
    def hook(mod, inp, out):
        x = (out[0] if isinstance(out, tuple) else out).clone()
        c = cm.to(x.device, x.dtype); dd = d.to(x.device, x.dtype)
        cur = ((x[:, -1, :].float() - cm.to(x.device)) * d.to(x.device)).sum(-1, keepdim=True) / den.to(x.device)
        x[:, -1, :] += ((target - cur).to(x.dtype) * dd)
        return (x,) + out[1:] if isinstance(out, tuple) else x
    return hook


def forward(model, tok, messages, hooks=None):
    x = chat_ids(tok, messages)
    handles = []
    try:
        for layer, hook in (hooks or []):
            handles.append(model.model.layers[layer].register_forward_hook(hook))
        with torch.inference_mode():
            o = model(x, output_hidden_states=True, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return [q[0, -1].detach().cpu().float().clone() for q in o.hidden_states]


def ztraj(states, cms, dirs):
    rows = []
    for h in range(1, len(states)):
        d = dirs[h]; cm = cms[h]; den = torch.dot(d, d) + 1e-12
        z = float(torch.dot(states[h] - cm, d) / den)
        rows.append((h - 1, z))
    return rows


def ctx(text):
    return [{"role": "user", "content": text}, {"role": "user", "content": PROBE}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--outdir", default="hat_restoration_results")
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()

    contexts = {
        "self": ctx(SELF),
        "external_recursive": ctx(EXTERNAL),
        "history": ctx(HISTORY),
        "consciousness_semantic": ctx(CONCEPT),
        "clean": ctx(CLEAN),
    }
    base = {k: forward(model, tok, v) for k, v in contexts.items()}
    L = len(model.model.layers)
    cms, dirs = {}, {}
    for h in range(1, L + 1):
        cm = torch.stack([base[k][h] for k in ["external_recursive", "history", "consciousness_semantic"]]).mean(0)
        cms[h] = cm
        dirs[h] = base["self"][h] - cm

    pulse_layer = max(0, int(round(0.20 * (L - 1))))
    ablation_layers = sorted(set([max(pulse_layer + 2, int(round(0.40 * (L - 1)))), max(pulse_layer + 3, int(round(0.60 * (L - 1))))]))

    rows = []
    for arm, st in [("self_baseline", base["self"]), ("clean_baseline", base["clean"])]:
        for layer, z in ztraj(st, cms, dirs): rows.append(dict(arm=arm, ablation_layer=-1, layer=layer, z_selfref=z))

    summaries = []
    for abl in ablation_layers:
        # Cue-present: erase the self-reference component while the inducing text remains available.
        self_ab = forward(model, tok, contexts["self"], [(abl, set_z_hook(cms[abl+1], dirs[abl+1], 0.0))])
        # Cue-absent pulse: implant the self-reference component into a clean context only at an early layer.
        pulse = [(pulse_layer, set_z_hook(cms[pulse_layer+1], dirs[pulse_layer+1], 1.0))]
        clean_pulse = forward(model, tok, contexts["clean"], pulse)
        # Strong test: implant the state, then erase it later, with no self-reference text in the context.
        pulse_ab = pulse + [(abl, set_z_hook(cms[abl+1], dirs[abl+1], 0.0))]
        clean_pulse_ab = forward(model, tok, contexts["clean"], pulse_ab)

        arm_states = {"self_ablate": self_ab, "clean_pulse": clean_pulse, "clean_pulse_ablate": clean_pulse_ab}
        for arm, st in arm_states.items():
            for layer, z in ztraj(st, cms, dirs): rows.append(dict(arm=arm, ablation_layer=abl, layer=layer, z_selfref=z))

        def z_at(st, layer):
            h = layer + 1; d = dirs[h]; return float(torch.dot(st[h]-cms[h], d)/(torch.dot(d,d)+1e-12))
        final = L - 1
        late_layers = list(range(min(abl + 2, final), final + 1))
        def late_mean(st): return float(np.mean([z_at(st, q) for q in late_layers])) if late_layers else z_at(st, final)
        summaries.append(dict(
            pulse_layer=pulse_layer,
            ablation_layer=abl,
            cue_present_immediate=z_at(self_ab, abl),
            cue_present_final=z_at(self_ab, final),
            cue_present_late_mean=late_mean(self_ab),
            cue_absent_pulse_final=z_at(clean_pulse, final),
            cue_absent_postablation_immediate=z_at(clean_pulse_ab, abl),
            cue_absent_postablation_final=z_at(clean_pulse_ab, final),
            cue_absent_postablation_late_mean=late_mean(clean_pulse_ab),
            clean_baseline_final=z_at(base["clean"], final),
        ))

    df = pd.DataFrame(rows); df.to_csv(out / "restoration_trajectories.csv", index=False)
    sf = pd.DataFrame(summaries); sf.to_csv(out / "restoration_summary.csv", index=False)

    # Prompt-specificity check: by construction self=1, control centroid=0; individual controls reveal contamination.
    spec = []
    for k, st in base.items():
        for layer, z in ztraj(st, cms, dirs): spec.append(dict(condition=k, layer=layer, z_selfref=z))
    pd.DataFrame(spec).to_csv(out / "specificity_trajectories.csv", index=False)

    import matplotlib.pyplot as plt
    for abl in ablation_layers:
        fig, ax = plt.subplots(figsize=(7, 4))
        dff = df[(df.ablation_layer == abl) & (df.arm.isin(["self_ablate", "clean_pulse", "clean_pulse_ablate"]))]
        for arm in ["self_ablate", "clean_pulse", "clean_pulse_ablate"]:
            q = dff[dff.arm == arm]; ax.plot(q.layer, q.z_selfref, marker="o", ms=2, label=arm)
        ax.axvline(pulse_layer, linestyle="--", linewidth=1); ax.axvline(abl, linestyle=":", linewidth=1)
        ax.axhline(0, linewidth=1); ax.axhline(1, linewidth=1)
        ax.set_xlabel("Transformer layer"); ax.set_ylabel("Self-reference-direction projection"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / f"restoration_layer_{abl}.png", dpi=170); plt.close(fig)

    verdicts = []
    for r in summaries:
        # Descriptive, preregistered interpretation bands; not a consciousness threshold.
        absent = r["cue_absent_postablation_final"]
        present = r["cue_present_final"]
        verdicts.append({"ablation_layer": r["ablation_layer"], "cue_present_recovery": present, "cue_absent_recovery": absent,
                         "interpretation": "intrinsic_reconstruction_candidate" if absent >= 0.50 else ("contextual_recomputation_only" if present >= 0.50 and absent < 0.25 else "weak_or_ambiguous_reconstruction")})
    meta = {
        "model": args.model, "seed": args.seed, "layers": L, "pulse_layer": pulse_layer, "ablation_layers": ablation_layers,
        "design": "Self-reference direction derived against external-recursive, history, and consciousness-semantic controls. Test compares cue-present reconstruction with cue-absent activation-pulse plus later ablation.",
        "important_limit": "A feed-forward transformer has no autonomous recurrent latent state across calls. Recovery across later layers is evidence of within-pass reconstruction, not by itself temporal self-maintenance or consciousness.",
        "runtime_seconds": time.time() - t0, "verdicts": verdicts
    }
    (out / "summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)

if __name__ == "__main__": main()
