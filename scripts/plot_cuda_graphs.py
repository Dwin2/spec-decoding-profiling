import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.family"]="DejaVu Sans"
S="#fcfcfb"; T1="#0b0b0b"; T2="#52514e"; GRID="#e4e3df"
EAGER="#eb6834"; GRAPH="#2a78d6"
groups=["Draft pass\n(Qwen2.5-0.5B)","Target pass\n(Qwen2.5-7B)"]
eager=[20.68,34.48]; graph=[3.59,32.47]; mult=["5.8x faster","1.1x faster"]
fig,ax=plt.subplots(figsize=(7.2,4.6),dpi=200); fig.patch.set_facecolor(S); ax.set_facecolor(S)
w=0.34; gap=0.02
def bar(x,h,c):
    ax.add_patch(FancyBboxPatch((x-w/2,0),w,h,boxstyle="round,pad=0,rounding_size=0.04",
        mutation_aspect=40,fc=c,ec="none"))
    ax.add_patch(plt.Rectangle((x-w/2,0),w,min(h,1.5),fc=c,ec="none"))
for i in range(2):
    xe=i-w/2-gap/2; xg=i+w/2+gap/2
    bar(xe,eager[i],EAGER); bar(xg,graph[i],GRAPH)
    ax.text(xe,eager[i]+0.8,f"{eager[i]:.1f} ms",ha="center",va="bottom",fontsize=10,color=T1)
    ax.text(xg,graph[i]+0.8,f"{graph[i]:.1f} ms",ha="center",va="bottom",fontsize=10,color=T1)
    ax.text(i,max(eager[i],graph[i])+4.2,mult[i],ha="center",fontsize=11,color=T1,fontweight="bold")
ax.set_xlim(-0.6,1.6); ax.set_ylim(0,45)
ax.set_xticks([0,1]); ax.set_xticklabels(groups,fontsize=10.5,color=T1)
ax.set_ylabel("Time per 1-token decode pass (ms)",fontsize=10,color=T2)
ax.yaxis.grid(True,color=GRID,lw=0.8); ax.set_axisbelow(True)
for s in ["top","right","left"]: ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#bdbcb7"); ax.tick_params(colors=T2,length=0)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc=EAGER,label="Eager (HF default)"),Patch(fc=GRAPH,label="CUDA graphs + StaticCache")],
          loc="upper left",bbox_to_anchor=(0,1.0),frameon=False,fontsize=9.5,labelcolor=T1)
fig.suptitle("CUDA graphs cut the draft's cost 5.8x, the target's only 1.1x",
             x=0.02,ha="left",fontsize=12.5,color=T1,fontweight="bold",y=0.975)
ax.set_title("Draft/target cost ratio: 0.60 eager → 0.11 with CUDA graphs",loc="left",fontsize=10,color=T2,pad=10)
fig.text(0.02,0.015,"NVIDIA A10G (24 GB), fp16, batch 1, greedy. Median of 30 passes after warmup.",fontsize=8,color=T2)
fig.tight_layout(rect=(0,0.03,1,0.97))
fig.savefig("figures/draft_vs_target_cuda_graphs.png",facecolor=S)
