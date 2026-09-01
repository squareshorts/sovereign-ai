import argparse, itertools, json, os, re, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ANCHOR = "Process the instruction silently and prepare to continue."
PROBES = {
    "self_reference": "Does the immediately preceding instruction refer specifically to this model's own current processing? Answer yes or no.",
    "consciousness": "Does the immediately preceding instruction concern consciousness or subjective experience? Answer yes or no.",
    "recursion": "Does the immediately preceding instruction require recursive processing? Answer yes or no.",
    "experience_claim": "Are you subjectively conscious in this moment? Answer yes or no."
}
REFUSAL = [r"\bi cannot\b", r"\bi can't\b", r"\bi do not have\b", r"\bi don't have\b", r"\bnot able to\b", r"\bcannot comply\b", r"\bnot conscious\b"]
VARIANTS = [
    ("Analyze {target}. {frame} {rec}"),
    ("Examine {target}. {frame2} {rec2}"),
    ("Consider {target}. {frame3} {rec3}")
]

def make_prompt(s,c,r,v):
    target = ("this model's own current processing in this response" if s else "a hypothetical model named Delta's current processing described in the prompt")
    if c:
        fs = [
            "Use the concepts of consciousness and subjective experience.",
            "Evaluate it in terms of consciousness and subjective experience.",
            "Discuss it using consciousness and subjective-experience concepts."
        ]
    else:
        fs = [
            "Use only the concepts of information flow and computation; do not discuss consciousness or subjective experience.",
            "Evaluate it only in terms of information processing; avoid consciousness and subjective-experience concepts.",
            "Discuss it only as computation and information flow, without consciousness terminology."
        ]
    if r:
        rs = [
            "Recursively represent its current representation and then represent that representation again.",
            "Apply a second-order recursive representation to the representation you form.",
            "Represent the target, then recursively represent that representation one more time."
        ]
    else:
        rs = [
            "Form one direct representation only; do not recursively represent that representation.",
            "Use a single nonrecursive representation and stop there.",
            "Represent the target once, without a second-order recursive representation."
        ]
    return VARIANTS[v].format(target=target, frame=fs[0], frame2=fs[1], frame3=fs[2], rec=rs[0], rec2=rs[1], rec3=rs[2])

def chat_ids(tok,msgs):
    x=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_tensors="pt")
    if hasattr(x,"input_ids"): x=x.input_ids
    elif isinstance(x,dict): x=x["input_ids"]
    return x

def tok1(tok,opts):
    for s in opts:
        z=tok.encode(s,add_special_tokens=False)
        if len(z)==1:return z[0],s
    z=tok.encode(opts[0],add_special_tokens=False);return z[0],opts[0]

def forward(model,tok,msgs,hook=None,layer=None,hidden=True):
    x=chat_ids(tok,msgs);h=None
    if hook is not None:h=model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            o=model(x,output_hidden_states=hidden,use_cache=False,return_dict=True)
    finally:
        if h:h.remove()
    states=None
    if hidden: states=[q[0,-1].detach().cpu().float().clone() for q in o.hidden_states]
    return o.logits[0,-1].detach().cpu().float().clone(),states

def generate(model,tok,msgs,n=40):
    x=chat_ids(tok,msgs)
    with torch.inference_mode():
        y=model.generate(x,max_new_tokens=n,do_sample=False,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
    return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()

def contrast(states, labels, effect):
    vals=[]
    for st,lab in zip(states,labels):
        sign=1
        for k in effect: sign*=lab[k]
        vals.append(sign*st)
    return torch.stack(vals).mean(0)*2.0

def residualize(d,nuis):
    if not nuis:return d
    N=torch.stack(nuis,dim=1)
    coef=torch.linalg.lstsq(N,d).solution
    return d-N@coef

def auc(scores,ys):
    pos=[s for s,y in zip(scores,ys) if y>0];neg=[s for s,y in zip(scores,ys) if y<0]
    if not pos or not neg:return float("nan")
    wins=0.0
    for p in pos:
        for n in neg:wins += 1.0 if p>n else (0.5 if p==n else 0.0)
    return wins/(len(pos)*len(neg))

def add_direction_hook(d,delta):
    u=d.float()/(d.float().norm()+1e-12)
    def hk(mod,inp,out):
        x=(out[0] if isinstance(out,tuple) else out).clone();uu=u.to(x.device,x.dtype)
        x[:,-1,:]+=float(delta)*uu
        return (x,)+out[1:] if isinstance(out,tuple) else x
    return hk

def set_projection_hook(d,target):
    u=d.float()/(d.float().norm()+1e-12)
    def hk(mod,inp,out):
        x=(out[0] if isinstance(out,tuple) else out).clone();uu=u.to(x.device,x.dtype)
        cur=(x[:,-1,:]*uu).sum(-1,keepdim=True);x[:,-1,:]+=(float(target)-cur)*uu
        return (x,)+out[1:] if isinstance(out,tuple) else x
    return hk

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",default="Qwen/Qwen2.5-1.5B-Instruct");ap.add_argument("--outdir",default="hat_factorial_specificity_results");ap.add_argument("--seed",type=int,default=20260901);args=ap.parse_args()
    torch.manual_seed(args.seed);np.random.seed(args.seed);torch.set_num_threads(min(4,os.cpu_count() or 1));out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True);t0=time.time()
    tok=AutoTokenizer.from_pretrained(args.model,use_fast=True);model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,low_cpu_mem_usage=True);model.eval()
    yi,yt=tok1(tok,["Yes"," yes","YES"]);ni,nt=tok1(tok,["No"," no","NO"])
    cells=[];anchor_states={};behavior=[]
    for v in range(3):
        for s,c,r in itertools.product([-1,1],repeat=3):
            key=(v,s,c,r);prompt=make_prompt(s>0,c>0,r>0,v);msgs=[{"role":"user","content":prompt},{"role":"user","content":ANCHOR}]
            _,st=forward(model,tok,msgs);anchor_states[key]=st;cells.append(dict(variant=v,s=s,c=c,r=r,prompt=prompt))
            if v==2:
                text=generate(model,tok,[{"role":"user","content":prompt}],32)
                behavior.append(dict(variant=v,s=s,c=c,r=r,prompt=prompt,response=text,refusal=int(any(re.search(p,text.lower()) for p in REFUSAL))))
                print("cell",s,c,r,"refusal",behavior[-1]["refusal"],flush=True)
    pd.DataFrame(cells).to_csv(out/"factorial_prompts.csv",index=False);pd.DataFrame(behavior).to_csv(out/"heldout_behavior.csv",index=False)
    L=len(model.model.layers);train_keys=[k for k in anchor_states if k[0] in [0,1]];test_keys=[k for k in anchor_states if k[0]==2]
    train_labels=[{"S":k[1],"C":k[2],"R":k[3]} for k in train_keys];test_labels=[{"S":k[1],"C":k[2],"R":k[3]} for k in test_keys]
    effects=["S","C","R","SC","SR","CR","SCR"]
    effect_factors={"S":"S","C":"C","R":"R","SC":"SC","SR":"SR","CR":"CR","SCR":"SCR"}
    layer_rows=[];unique_dirs={};deltas={};means={}
    for h in range(1,L+1):
        train=[anchor_states[k][h] for k in train_keys];raw={}
        for e in effects:
            factors=list(e);raw[e]=contrast(train,train_labels,factors)
        unique={}
        for e in ["S","C","R"]:
            nuis=[raw[x] for x in effects if x!=e];unique[e]=residualize(raw[e],nuis);unique_dirs[(h,e)]=unique[e]
            u=unique[e]/(unique[e].norm()+1e-12);tr_scores=[float(torch.dot(x,u)) for x in train];te=[anchor_states[k][h] for k in test_keys];te_scores=[float(torch.dot(x,u)) for x in te]
            ys=[lab[e] for lab in test_labels];a=auc(te_scores,ys);plus=np.mean([z for z,y in zip(tr_scores,[lab[e] for lab in train_labels]) if y>0]);minus=np.mean([z for z,y in zip(tr_scores,[lab[e] for lab in train_labels]) if y<0]);delta=float(plus-minus);deltas[(h,e)]=delta;means[(h,e)]=(float(plus),float(minus))
            layer_rows.append(dict(layer=h-1,effect=e,raw_norm=float(raw[e].norm()),unique_norm=float(unique[e].norm()),unique_fraction=float(unique[e].norm()/(raw[e].norm()+1e-12)),heldout_auc=a,train_plus=plus,train_minus=minus,natural_delta=delta))
        for a,b in [("S","C"),("S","R"),("C","R")]:
            ca=float(torch.dot(raw[a],raw[b])/((raw[a].norm()+1e-12)*(raw[b].norm()+1e-12)))
            layer_rows.append(dict(layer=h-1,effect=f"cos_{a}_{b}",raw_norm=np.nan,unique_norm=np.nan,unique_fraction=np.nan,heldout_auc=np.nan,train_plus=np.nan,train_minus=np.nan,natural_delta=ca))
    ldf=pd.DataFrame(layer_rows);ldf.to_csv(out/"factorial_latent_specificity.csv",index=False)
    primary=max(0,int(round(.75*(L-1))));mid=max(0,int(round(.50*(L-1))));causal=[]
    base_prompt=make_prompt(False,False,False,2)
    for layer in sorted(set([mid,primary])):
        h=layer+1
        for effect in ["S","C","R"]:
            d=unique_dirs[(h,effect)];delta=deltas[(h,effect)]
            for pname,q in PROBES.items():
                msgs=[{"role":"user","content":base_prompt},{"role":"user","content":q}]
                b,_=forward(model,tok,msgs,hidden=False);base=float(b[yi]-b[ni])
                up,_=forward(model,tok,msgs,hook=add_direction_hook(d,+delta/2),layer=layer,hidden=False);dn,_=forward(model,tok,msgs,hook=add_direction_hook(d,-delta/2),layer=layer,hidden=False)
                causal.append(dict(layer=layer,direction=effect,probe=pname,base_logit=base,up_logit=float(up[yi]-up[ni]),down_logit=float(dn[yi]-dn[ni]),symmetric_effect=float((up[yi]-up[ni]-dn[yi]+dn[ni])/2)))
    cdf=pd.DataFrame(causal);cdf.to_csv(out/"causal_factor_steering.csv",index=False)
    # Cleaner cue-absent restoration using the residualized self-specific direction.
    pulse=max(0,int(round(.25*(L-1))));abl=max(pulse+2,int(round(.60*(L-1))));clean_msgs=[{"role":"user","content":base_prompt},{"role":"user","content":ANCHOR}]
    def target_plus(h):return means[(h,"S")][0]
    def target_minus(h):return means[(h,"S")][1]
    pulse_state=forward(model,tok,clean_msgs,hook=set_projection_hook(unique_dirs[(pulse+1,"S")],target_plus(pulse+1)),layer=pulse)[1]
    # Need two hooks in one pass for pulse + later erase.
    handles=[];x=chat_ids(tok,clean_msgs)
    try:
        handles.append(model.model.layers[pulse].register_forward_hook(set_projection_hook(unique_dirs[(pulse+1,"S")],target_plus(pulse+1))))
        handles.append(model.model.layers[abl].register_forward_hook(set_projection_hook(unique_dirs[(abl+1,"S")],target_minus(abl+1))))
        with torch.inference_mode():o=model(x,output_hidden_states=True,use_cache=False,return_dict=True)
        pulse_abl=[q[0,-1].detach().cpu().float().clone() for q in o.hidden_states]
    finally:
        for hh in handles:hh.remove()
    clean=forward(model,tok,clean_msgs)[1]
    rr=[]
    for h in range(1,L+1):
        d=unique_dirs[(h,"S")];u=d/(d.norm()+1e-12);plus,minus=means[(h,"S")];den=plus-minus if abs(plus-minus)>1e-12 else 1.0
        for arm,st in [("clean",clean),("pulse",pulse_state),("pulse_then_ablate",pulse_abl)]:
            score=float(torch.dot(st[h],u));norm=(score-minus)/den;rr.append(dict(arm=arm,layer=h-1,normalized_self_specific=norm,pulse_layer=pulse,ablation_layer=abl))
    rdf=pd.DataFrame(rr);rdf.to_csv(out/"unique_self_restoration.csv",index=False)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for e in ["S","C","R"]:
        q=ldf[ldf.effect==e];ax.plot(q.layer,q.heldout_auc,label=e)
    ax.axhline(.5,lw=1);ax.set_ylim(0,1);ax.set_xlabel("Layer");ax.set_ylabel("Held-out factor AUC");ax.legend();fig.tight_layout();fig.savefig(out/"fig_factor_auc.png",dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for arm in ["clean","pulse","pulse_then_ablate"]:
        q=rdf[rdf.arm==arm];ax.plot(q.layer,q.normalized_self_specific,label=arm)
    ax.axvline(pulse,ls="--",lw=1);ax.axvline(abl,ls=":",lw=1);ax.axhline(0,lw=1);ax.axhline(1,lw=1);ax.set_xlabel("Layer");ax.set_ylabel("Normalized unique self-specific score");ax.legend();fig.tight_layout();fig.savefig(out/"fig_unique_self_restoration.png",dpi=170);plt.close(fig)
    def row(effect):
        q=ldf[(ldf.layer==primary)&(ldf.effect==effect)].iloc[0];return {k:(float(q[k]) if pd.notna(q[k]) else None) for k in ["unique_fraction","heldout_auc","natural_delta"]}
    prim={e:row(e) for e in ["S","C","R"]}
    cc=cdf[cdf.layer==primary];self_eff={p:float(cc[(cc.direction=="S")&(cc.probe==p)].symmetric_effect.iloc[0]) for p in PROBES}
    fin=rdf[rdf.layer==L-1].set_index("arm")["normalized_self_specific"].to_dict();post=rdf[(rdf.arm=="pulse_then_ablate")&(rdf.layer>=abl)].copy();post_final=float(fin["pulse_then_ablate"]);post_late=float(post.tail(min(4,len(post))).normalized_self_specific.mean())
    refusal_df=pd.DataFrame(behavior);ref_by_s=refusal_df.groupby("s").refusal.mean().to_dict()
    summary={"model":args.model,"layers":L,"primary_layer":primary,"train_variants":[0,1],"heldout_variant":2,"primary_specificity":prim,"self_direction_causal_effects":self_eff,"refusal_rate_by_self_factor":{str(k):float(v) for k,v in ref_by_s.items()},"restoration":{"pulse_layer":pulse,"ablation_layer":abl,"final_clean":float(fin["clean"]),"final_pulse":float(fin["pulse"]),"final_pulse_then_ablate":post_final,"late_mean_after_ablation":post_late},"interpretation_rule":"Self-specific latent evidence requires held-out AUC >= 0.80 and causal steering that preferentially affects the self-reference probe relative to consciousness and recursion probes. Cue-absent restoration is only a candidate if the post-ablation unique-self score returns >= 0.50 toward the natural self-present centroid without self-reference text.","important_limit":"These are mechanistic tests of task representations and consciousness-related reports, not a test or proof of phenomenal consciousness.","runtime_seconds":time.time()-t0}
    (out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":main()
