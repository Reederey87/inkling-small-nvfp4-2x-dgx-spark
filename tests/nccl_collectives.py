# Two-node NCCL collectives test (all_reduce + all_gather) — run on both nodes
# simultaneously with RANK=0/1. Run INSIDE the serving container (it carries the
# deployment's NCCL env): RANK=0 on head, RANK=1 on worker, same command line.
# init_method uses the rail-1 head IP — edit if your fabric IPs differ.
import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
dist.init_process_group("nccl", rank=rank, world_size=2,
                        init_method="tcp://192.168.177.10:25999")
t = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
dist.all_reduce(t)
print(f"[rank {rank}] all_reduce OK sum={t.sum().item()}", flush=True)
out = [torch.empty_like(t) for _ in range(2)]
dist.all_gather(out, t)
torch.cuda.synchronize()
print(f"[rank {rank}] all_gather OK {[o.sum().item() for o in out]}", flush=True)
dist.destroy_process_group()
