import matplotlib.pyplot as plt
plt.rcParams["font.family"]="DejaVu Sans"
S="#fcfcfb"; T1="#0b0b0b"; T2="#52514e"; GRID="#e4e3df"
k=[1,2,3,4,6]
serial=[38.23,42.85,47.53,52.08,61.25]; tog=[35.95,38.37,42.26,46.80,56.05]
ideal=[33.67,33.76,33.90,33.91,34.03]
BW=15.23/33.91*1000  # GB/s the target achieves alone
floor=[(15.23+0.988*x)/BW*1000 for x in k]
fig,ax=plt.subplots(figsize=(7.4,4.7),dpi=200); fig.patch.set_facecolor(S); ax.set_facecolor(S)
series=[("Serial (draft, then target)",serial,"#eb6834","o","-"),
        ("Together, measured",tog,"#2a78d6","o","-"),
        ("Memory-bandwidth floor",floor,"#52514e","s","--"),
        ("Perfect overlap (target alone)",ideal,"#1baf7a","^",":")]
for name,y,c,m,ls in series:
    ax.plot(k,y,color=c,lw=2,ls=ls,marker=m,ms=7,mec=S,mew=2,label=name)
    ax.text(6.15,y[-1],f"{y[-1]:.1f}",va="center",fontsize=9,color=T1)
ax.set_xticks(k); ax.set_xlabel("Draft tokens per round (k)",fontsize=10,color=T2)
ax.set_ylabel("Time for one round (ms)",fontsize=10,color=T2)
ax.set_xlim(0.7,6.6); ax.set_ylim(30,64)
ax.yaxis.grid(True,color=GRID,lw=.8); ax.set_axisbelow(True)
for s in ["top","right","left"]: ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#bdbcb7"); ax.tick_params(colors=T2,length=0)
ax.legend(loc="upper left",frameon=False,fontsize=9,labelcolor=T1)
fig.suptitle("Running draft and target together sits near the bandwidth floor for small k",x=0.02,ha="left",fontsize=12,fontweight="bold",color=T1,y=0.975)
ax.set_title("The gap above the floor grows with k: that is the room isolation could recover",loc="left",fontsize=9.5,color=T2,pad=8)
fig.text(0.02,0.012,"A10G · 7B verify (k+1 tokens) + 0.5B draft (k passes) · CUDA graphs on separate streams\nFloor = (target + k × draft weight bytes) / 449 GB/s, the bandwidth the target reaches alone",fontsize=7.5,color=T2)
fig.tight_layout(rect=(0,0.06,1,0.97))
fig.savefig("figures/overlap_vs_bandwidth_floor.png",facecolor=S)
