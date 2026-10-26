"""Where does the draft lose time when it runs alongside the target?
Usage: nsys export --type sqlite step7_k4.nsys-rep && python scripts/analyze_step7_nsys.py step7_k4.sqlite"""
import statistics as st, collections, bisect
import sys, sqlite3
con=sqlite3.connect(sys.argv[1]); S=dict(con.execute("select id,value from StringIds"))
nv=con.execute("select start,end,coalesce(text,(select value from StringIds where id=textId)) from NVTX_EVENTS order by start").fetchall()
K=[(a,b,s,S.get(n,n),g,bl,rg,sm,gid) for a,b,s,n,g,bl,rg,sm,gid in con.execute("select start,end,streamId,shortName,gridX*gridY*gridZ,blockX*blockY*blockZ,registersPerThread,staticSharedMemory+dynamicSharedMemory,graphId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]; R={n:(a,b) for a,b,n in nv}
rng0=lambda n:[k for k in K if k[0]>=R[n][0] and k[1]<=R[n][1]]
DS=max(set(k[2] for k in rng0("alone: draft")), key=[k[2] for k in rng0("alone: draft")].count)
TS=max(set(k[2] for k in rng0("alone: target verify")), key=[k[2] for k in rng0("alone: target verify")].count)
rng=lambda n:[k for k in K if k[0]>=R[n][0] and k[1]<=R[n][1]]
dA=[k for k in rng("alone: draft") if k[2]==DS]; tA=[k for k in rng("alone: target verify") if k[2]==TS]
tog=rng("together"); dT=[k for k in tog if k[2]==DS]; tT=[k for k in tog if k[2]==TS]
# merged target-busy intervals
iv=[]
for k in tT:
    if iv and k[0]<=iv[-1][1]: iv[-1][1]=max(iv[-1][1],k[1])
    else: iv.append([k[0],k[1]])
starts=[a for a,b in iv]
def active(t):
    i=bisect.bisect_right(starts,t)-1
    return i>=0 and iv[i][1]>t
tstart,tend=tT[0][0],tT[-1][1]
print("together window: draft %.2f-%.2f ms, target %.2f-%.2f ms (from first kernel)"%(0,(dT[-1][1]-dT[0][0])/1e6,(tstart-dT[0][0])/1e6,(tend-dT[0][0])/1e6))
n=len(dA)  # alone sequence length for all 5 replays (same order)
ov=[];no=[]
for i,k in enumerate(dT):
    ka=dA[i]; assert ka[3]==k[3]
    mid=(k[0]+k[1])//2
    (ov if active(mid) else no).append((ka,k, (dT[i+1][0]-k[1]) if i+1<len(dT) else None, (dA[i+1][0]-ka[1]) if i+1<len(dA) else None))
for lab,xs in (("while target runs",ov),("target idle",no)):
    if not xs: continue
    da=sum(a[1]-a[0] for a,_,_,_ in xs); dt=sum(b[1]-b[0] for _,b,_,_ in xs)
    gt=[g for *_,g,_ in xs if g is not None]; ga=[g for *_,_,g in xs if g is not None]
    print(f"{lab:18s}: {len(xs):5d} draft kernels | kernel time {da/1e6:6.2f} -> {dt/1e6:6.2f} ms ({dt/da:.2f}x) | gap after kernel median {st.median(ga)/1e3:.1f} -> {st.median(gt)/1e3:.1f} us, total {sum(ga)/1e6:.2f} -> {sum(gt)/1e6:.2f} ms")
# per kernel type, while target runs
by=collections.defaultdict(lambda:[0,0,0,None])
for a,b,_,_ in ov:
    x=by[a[3][:55]]; x[0]+=a[1]-a[0]; x[1]+=b[1]-b[0]; x[2]+=1; x[3]=(a[4],a[5])
print("\ndraft kernel types while target runs: alone-equivalent ms -> co-run ms  (grid,block)")
for nm,(x,y,c,g) in sorted(by.items(), key=lambda z:-(z[1][1]-z[1][0]))[:8]:
    print(f"  {nm:55s} {x/1e6:6.2f} -> {y/1e6:6.2f} ({y/x:4.1f}x, n={c}) {g}")
tb=collections.defaultdict(lambda:[0,0,None])
for k in tA[:len(tA)//5]:
    x=tb[k[3][:55]]; x[0]+=k[1]-k[0]; x[1]+=1; x[2]=(k[4],k[5],k[6],k[7])
print("\ntarget kernels per verify pass (ms, n, (grid,block,regs,smem)):")
for nm,(x,c,g) in sorted(tb.items(), key=lambda z:-z[1][0])[:6]: print(f"  {nm:55s} {x/1e6:6.2f} n={c} {g}")
print("target kernel time alone vs together: %.2f -> %.2f ms total"%(sum(k[1]-k[0] for k in tA)/1e6, sum(k[1]-k[0] for k in tT)/1e6))
