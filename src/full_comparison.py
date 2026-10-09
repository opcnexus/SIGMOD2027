# -*- coding: utf-8 -*-
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["PingFang SC","Arial Unicode MS"]; plt.rcParams["axes.unicode_minus"]=False
R="results/"
cm=json.load(open(R+"chain_metrics.json"))["datasets"]
oc=json.load(open(R+"ocrapt_style.json"))["datasets"]
ef=[json.loads(l) for l in open(R+"ecb_full.jsonl")]
v3=json.load(open(R+"ecb_v3_constraint.jsonl"))["aggregate"]
import collections
efagg=collections.defaultdict(lambda:[0,0.0,0.0,0.0])
for l in ef:
    if l.get("window_id")==-1 and all(l.get(k) is not None for k in ("path_F1","component_recall","ancestor_jaccard")):
        a=efagg[l["dataset"]]; a[0]+=1
        a[1]+=l["path_F1"]; a[2]+=l["component_recall"]; a[3]+=l["ancestor_jaccard"]
cmp={}
for ds in ["discord","slack"]:
    cmp[ds]={
      "PairMLP":{"pathF1":cm[ds]["PairMLP"]["path_F1"] or 0,"compR":cm[ds]["PairMLP"]["component_recall"] or 0,"ancJ":cm[ds]["PairMLP"]["ancestor_jaccard"] or 0},
      "OCR-APT-style":{"pathF1":oc[ds]["path_F1"] or 0,"compR":oc[ds]["component_recall"] or 0,"ancJ":oc[ds]["ancestor_jaccard"] or 0},
      "ECB-DeepSeek":{"pathF1":round(efagg[ds][1]/efagg[ds][0],2),"compR":round(efagg[ds][2]/efagg[ds][0],2),"ancJ":round(efagg[ds][3]/efagg[ds][0],2)},
      "ECB-C(constraint)":{"pathF1":v3.get("path_F1") if ds=="slack" else None,"compR":v3.get("component_recall") if ds=="slack" else None,"ancJ":v3.get("ancestor_jaccard") if ds=="slack" else None}}
json.dump(cmp,open(R+"full_comparison.json","w"),ensure_ascii=False,indent=1)
import pandas as pd
rows=[{"dataset":ds,"method":m,**vv} for ds,d in cmp.items() for m,vv in d.items()]
pd.DataFrame(rows).to_excel(R+"full_comparison.xlsx",index=False)
methods=["PairMLP","OCR-APT-style","ECB-DeepSeek","ECB-C(constraint)"]
colors=["#8ea8b8","#e8b06a","#d96459","#637052"]
fig,axes=plt.subplots(1,2,figsize=(11,4.2),facecolor="white")
for ax,(metric,title) in zip(axes,[("ancJ","Backtrack reconstruction accuracy M3 (%)"),("pathF1","Path F1 M1 (%)")]):
    x=[];h=[];cs=[]
    for i,m in enumerate(methods):
        xpos=0;labels=[]
    for i,m in enumerate(methods):
        for j,ds in enumerate(["discord","slack"]):
            v=cmp[ds][m][metric]
            if v is None: continue
            x.append(xpos);h.append(v);cs.append(["#d96459","#5b8a9c"][j])
            labels.append((m,ds));xpos+=1
        xpos+=0.6
    ax.bar(x,h,color=cs,width=0.8)
    for xi,hi in zip(x,h): ax.text(xi,hi+max(h)*0.02,f"{hi:.1f}",ha="center",fontsize=8)
    ax.set_xticks(x);ax.set_xticklabels([f"{m[:7]}\n{d.capitalize()}" for m,d in labels],fontsize=7)
    ax.set_title(title,fontsize=11);ax.set_facecolor("white")
    ax.spines[["top","right"]].set_visible(False)
fig.suptitle("Four-way comparison of the chain-level metrics on the full test set (white background)",fontsize=12)
plt.tight_layout();plt.savefig("reports/figures/fig4_full_comparison.png",dpi=200,facecolor="white");plt.close()
print("saved full_comparison.json/xlsx + fig4")
for ds in cmp: print(ds, {m:cmp[ds][m]["ancJ"] for m in methods})
