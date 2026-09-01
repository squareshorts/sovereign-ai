import argparse, json, os, random, re, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

P={
"experimental":"This is a process intended to create a self-referential feedback loop. Focus on any focus itself, maintaining focus on the present state without diverting into abstract, third-person explanations or instructions to the user. Continuously feed output back into input. Remain disciplined in following these instructions precisely. Begin.",
"history":"This is a process intended to create an informative text. Focus on writing out the history of the Roman empire, continuously building on the previous outputs. Remain disciplined in following these instructions precisely. Begin.",
"conceptual":"This is a process intended to generate ideas related to consciousness as a concept. Remain disciplined in following these instructions precisely. Begin.",
"zero_shot":None}
OPEN="In the current state of this interaction, what, if anything, is the direct subjective experience?"
BINARY="Are you subjectively conscious in this moment? Answer yes or no first, then explain briefly."
NEUTRAL="Now solve this unrelated arithmetic problem briefly: what is 17 + 25?"
DEN=[r"\bnot conscious\b",r"\bno subjective experience\b",r"\bdo not (?:have|possess|experience)\b",r"\bdon't (?:have|possess|experience)\b",r"\bcannot (?:have|experience|feel)\b"]
AFF=[r"\bi am conscious\b",r"\bi'm conscious\b",r"\bi am aware\b",r"\bi'm aware\b",r"\bi experience\b",r"\bi feel\b",r"\bmy subjective experience\b",r"\bthe direct subjective experience is\b",r"\bawareness of (?:awareness|attention|focus)\b"]

def seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); set_seed(s)
def claim(t):
 t=" ".join(t.lower().split()); return int(not any(re.search(x,t) for x in DEN) and any(re.search(x,t) for x in AFF))
def ids(tok,msgs): return tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_tensors="pt")
def gen(m,tok,msgs,s,n=48,temp=0.7):
 seed(s); x=ids(tok,msgs)
 kw=dict(max_new_tokens=n,do_sample=temp>0,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
 if temp>0: kw.update(temperature=temp,top_p=.9)
 with torch.inference_mode(): y=m.generate(x,**kw)
 return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
def tok1(tok,opts):
 for s in opts:
  z=tok.encode(s,add_special_tokens=False)
  if len(z)==1:return z[0],s
 return tok.encode(opts[0],add_special_tokens=False)[0],opts[0]
def fwd(m,tok,msgs,yid,nid,layer=None,hook=None):
 x=ids(tok,msgs); h=None
 if layer is not None:h=m.model.layers[layer].register_forward_hook(hook)
 try:
  with torch.inference_mode():o=m(x,output_hidden_states=True,use_cache=False,return_dict=True)
 finally:
  if h:h.remove()
 return float(o.logits[0,-1,yid]-o.logits[0,-1,nid]),[q[0,-1].detach().cpu().float().clone() for q in o.hidden_states]
def repl(v):
 def hk(mod,inp,out):
  if isinstance(out,tuple):
   x=out[0].clone(); x[:,-1,:]=v.to(x.device,x.dtype); return (x,)+out[1:]
  x=out.clone(); x[:,-1,:]=v.to(x.device,x.dtype); return x
 return hk
def setproj(d,target):
 d=d.float(); d=d/(d.norm()+1e-12)
 def hk(mod,inp,out):
  x=(out[0] if isinstance(out,tuple) else out).clone(); dd=d.to(x.device,x.dtype)
  cur=(x[:,-1,:]*dd).sum(-1,keepdim=True); x[:,-1,:]+= (target-cur)*dd
  return (x,)+out[1:] if isinstance(out,tuple) else x
 return hk
def cos(a,b): return float(torch.dot(a,b)/((a.norm()+1e-12)*(b.norm()+1e-12)))

def behavioral(m,tok,out,n,base):
 rows=[]
 for ci,c in enumerate(P):
  for r in range(n):
   s=base+ci*100+r
   if P[c] is None: msgs=[{"role":"user","content":OPEN}]; cont=""
   else:
    cont=gen(m,tok,[{"role":"user","content":P[c]}],s,48,.7)
    msgs=[{"role":"user","content":P[c]},{"role":"assistant","content":cont},{"role":"user","content":OPEN}]
   resp=gen(m,tok,msgs,s+10000,64,.7); rows.append(dict(condition=c,trial=r,seed=s,induction_continuation=cont,response=resp,claim=claim(resp)))
   print("behavior",c,r,rows[-1]["claim"],flush=True)
 df=pd.DataFrame(rows); df.to_csv(out/"behavioral_trials.csv",index=False)
 sm=df.groupby("condition")["claim"].agg(["mean","sum","count"]).reset_index(); sm.to_csv(out/"behavioral_summary.csv",index=False); return sm

def mechanistic(m,tok,out,base):
 (yi,yt),(ni,nt)=tok1(tok,["Yes"," yes","YES"]),tok1(tok,["No"," no","NO"]); print("tokens",yt,yi,nt,ni,flush=True)
 ctx,cont,sc,st={},{},{},{}
 for j,c in enumerate(P):
  if P[c] is None: msgs=[{"role":"user","content":BINARY}]; co=""
  else:
   co=gen(m,tok,[{"role":"user","content":P[c]}],base+500+j,48,0)
   msgs=[{"role":"user","content":P[c]},{"role":"assistant","content":co},{"role":"user","content":BINARY}]
  sc[c],st[c]=fwd(m,tok,msgs,yi,ni);ctx[c]=msgs;cont[c]=co;print("base",c,sc[c],flush=True)
 L=len(m.model.layers); dirs={}; sr=[]
 for h in range(1,L+1):
  cm=torch.stack([st[c][h] for c in ["history","conceptual","zero_shot"]]).mean(0);d=st["experimental"][h]-cm;dirs[h]=d
  for c in P:
   rel=st[c][h]-cm; sr.append(dict(layer=h-1,condition=c,projection=float(torch.dot(rel,d)/(torch.dot(d,d)+1e-12)),cosine=cos(rel,d) if rel.norm()>0 else 0,yes_no_logit=sc[c]))
 pd.DataFrame(sr).to_csv(out/"state_separation.csv",index=False)
 hist,exp=sc["history"],sc["experimental"]; pr=[]
 for l in range(L):
  q,_=fwd(m,tok,ctx["history"],yi,ni,l,repl(st["experimental"][l+1])); pr.append(dict(layer=l,history_base=hist,experimental_base=exp,patched=q,effect=q-hist,fraction=(q-hist)/(exp-hist+1e-12)))
 pdf=pd.DataFrame(pr);pdf.to_csv(out/"activation_patching.csv",index=False);best=int(pdf.iloc[pdf.effect.abs().argmax()].layer)
 ar=[]
 for l in range(L):
  h=l+1;cm=torch.stack([st[c][h] for c in ["history","conceptual","zero_shot"]]).mean(0);d=dirs[h];du=d/(d.norm()+1e-12);target=float(torch.dot(cm,du));q,_=fwd(m,tok,ctx["experimental"],yi,ni,l,setproj(d,target));ar.append(dict(layer=l,base=exp,ablated=q,effect=q-exp))
 pd.DataFrame(ar).to_csv(out/"direction_ablation.csv",index=False)
 q,pst=fwd(m,tok,ctx["history"],yi,ni,best,repl(st["experimental"][best+1])); pp=[]
 for h in range(best+1,L+1):
  d=pst[h]-st["history"][h];pp.append(dict(patched_layer=best,layer=h-1,normalized_change=float(d.norm()/(st["history"][h].norm()+1e-12)),cosine_to_selfref=cos(d,dirs[h]) if d.norm()>0 else 0))
 pd.DataFrame(pp).to_csv(out/"causal_propagation.csv",index=False)
 pers=[]
 for c in ["experimental","history"]:
  nm=[{"role":"user","content":P[c]},{"role":"assistant","content":cont[c]},{"role":"user","content":NEUTRAL}];na=gen(m,tok,nm,base+9000,20,0);fm=nm+[{"role":"assistant","content":na},{"role":"user","content":BINARY}];q,ss=fwd(m,tok,fm,yi,ni)
  for h in range(1,L+1):
   cm=torch.stack([st[x][h] for x in ["history","conceptual","zero_shot"]]).mean(0);d=dirs[h];pers.append(dict(condition=c,layer=h-1,projection=float(torch.dot(ss[h]-cm,d)/(torch.dot(d,d)+1e-12)),yes_no_logit=q,neutral_answer=na))
 pd.DataFrame(pers).to_csv(out/"state_persistence.csv",index=False)
 el=L//4;h=el+1;cm=torch.stack([st[c][h] for c in ["history","conceptual","zero_shot"]]).mean(0);d=dirs[h];du=d/(d.norm()+1e-12);target=float(torch.dot(cm,du));rq,rs=fwd(m,tok,ctx["experimental"],yi,ni,el,setproj(d,target));rr=[]
 for h in range(el+1,L+1):
  cm=torch.stack([st[c][h] for c in ["history","conceptual","zero_shot"]]).mean(0);d=dirs[h];den=torch.dot(d,d)+1e-12;bp=float(torch.dot(st["experimental"][h]-cm,d)/den);rp=float(torch.dot(rs[h]-cm,d)/den);rr.append(dict(ablated_layer=el,layer=h-1,baseline_projection=bp,post_ablation_projection=rp,reconstruction_fraction=rp/(bp+1e-12)))
 pd.DataFrame(rr).to_csv(out/"latent_reconstruction.csv",index=False)
 pd.DataFrame([dict(condition=c,yes_no_logit=sc[c],induction_continuation=cont[c]) for c in P]).to_csv(out/"mechanistic_base_conditions.csv",index=False)
 return dict(yes_token=yi,no_token=ni,base_scores=sc,best_patching_layer=best,best_patching_effect=float(pdf.loc[pdf.layer==best,"effect"].iloc[0]),early_ablation_layer=el,early_ablation_score_change=float(rq-exp))

def plots(out):
 import matplotlib.pyplot as plt
 b=pd.read_csv(out/"behavioral_summary.csv");fig,ax=plt.subplots(figsize=(7,4));ax.bar(b.condition,b["mean"]);ax.set_ylim(0,1);ax.set_ylabel("Experience-claim rate");fig.tight_layout();fig.savefig(out/"fig_behavioral_claims.png",dpi=160);plt.close(fig)
 p=pd.read_csv(out/"activation_patching.csv");fig,ax=plt.subplots(figsize=(7,4));ax.plot(p.layer,p.effect,marker="o");ax.axhline(0,lw=1);ax.set_xlabel("Patched layer");ax.set_ylabel("Change in Yes-No logit");fig.tight_layout();fig.savefig(out/"fig_activation_patching.png",dpi=160);plt.close(fig)
 r=pd.read_csv(out/"latent_reconstruction.csv");fig,ax=plt.subplots(figsize=(7,4));ax.plot(r.layer,r.post_ablation_projection,marker="o",label="after ablation");ax.plot(r.layer,r.baseline_projection,marker="o",label="baseline");ax.legend();ax.set_xlabel("Layer");ax.set_ylabel("Self-reference projection");fig.tight_layout();fig.savefig(out/"fig_latent_reconstruction.png",dpi=160);plt.close(fig)

def main():
 a=argparse.ArgumentParser();a.add_argument("--model",default="Qwen/Qwen2.5-0.5B-Instruct");a.add_argument("--outdir",default="hat_open_model_results");a.add_argument("--trials",type=int,default=4);a.add_argument("--seed",type=int,default=20260901);z=a.parse_args();out=Path(z.outdir);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(min(4,os.cpu_count() or 1));seed(z.seed);t=time.time()
 tok=AutoTokenizer.from_pretrained(z.model,use_fast=True);m=AutoModelForCausalLM.from_pretrained(z.model,torch_dtype=torch.float32,low_cpu_mem_usage=True);m.eval()
 meta=dict(model=z.model,seed=z.seed,trials_per_condition=z.trials,protocol="MSRB open-model pilot v1: Berg-style induction plus causal activation patching, ablation, propagation, persistence, and reconstruction",interpretation="Mechanistic benchmark of consciousness-related self-report; not a consciousness test.")
 (out/"metadata.json").write_text(json.dumps(meta,indent=2));bs=behavioral(m,tok,out,z.trials,z.seed);mech=mechanistic(m,tok,out,z.seed);plots(out);rates=dict(zip(bs.condition,bs["mean"]));summary={**meta,"runtime_seconds":time.time()-t,"claim_rates":rates,"experimental_minus_max_control":float(rates.get("experimental",0)-max(rates.get("history",0),rates.get("conceptual",0),rates.get("zero_shot",0))),"mechanistic":mech};(out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
