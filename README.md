# Assymetric GPU micro-isolation

Run PEARL speculative decoding on the same GPU and measuring draft-target model/data dependencies.Built with
Python, PyTorch, CUDA, NVML and
Nsight Systems. 

**Results, findings and figures:** [figures/figures.md](figures/figures.md)

**Candidate primitives:** [primitives/README.md](primitives/README.md)

## Run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/step1.py 2>&1 | tee step1.txt     # likewise for step2 ... step7
```

Needs a CUDA GPU with ≥ 24 GB (results in [results/a10g/](results/a10g/) are from an A10G).
`step2.py` writes `trace.json` (open at https://ui.perfetto.dev). `step5.py` and `step6.py`
compile CUDA graphs on first use, which takes a few minutes.

Nsight Systems timeline for step7:

```bash
nsys profile -t cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi \
     -o step7_k4 python src/step7.py --nsys
nsys export --type sqlite step7_k4.nsys-rep && python scripts/analyze_step7_nsys.py step7_k4.sqlite
```
