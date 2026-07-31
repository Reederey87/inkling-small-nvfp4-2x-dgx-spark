# Networking — the 2x DGX Spark QSFP fabric

The two Sparks connect through a direct QSFP DAC (no switch). On DGX OS each
QSFP port surfaces as **two PCIe-twin NICs**; NCCL merges them for the full
link bandwidth (~23 GB/s measured all-reduce). This is the same fabric design
as the reference DeepSeek deployments on this hardware.

## Interfaces and static IPs

| Rail | Interface (head/worker) | Head IP | Worker IP | MTU |
|---|---|---|---|---|
| 1 | `enp1s0f1np1` | `192.168.177.10/24` | `192.168.177.11/24` | 9000 |
| 2 | `enP2p1s0f1np1` | `192.168.178.10/24` | `192.168.178.11/24` | 9000 |

Netplan sketch (per node — adjust interface names if yours differ,
`ip link | grep enp`):

```yaml
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses: [192.168.177.10/24]   # .11 on the worker
      mtu: 9000
    enP2p1s0f1np1:
      addresses: [192.168.178.10/24]   # .11 on the worker
      mtu: 9000
```

## NCCL environment (already in cluster.env — don't "optimize")

- `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1` + `NCCL_IB_MERGE_NICS=1` — the merge is
  what gets you dual-rail bandwidth. Single-HCA configs halve it.
- `NCCL_SOCKET_IFNAME=enp1s0f1np1` — bootstrap/control on rail 1.
- `NCCL_CROSS_NIC=1`, `NCCL_IB_ADDR_FAMILY=AF_INET`,
  `NCCL_IB_ROCE_VERSION_NUM=2` — RoCEv2 over IPv4.
- `MASTER_ADDR` = head's rail-1 IP. The rendezvous crosses the fabric, not the LAN.

## Verify

```
ping -c2 192.168.177.11                       # head -> worker rail 1
ping -c2 192.168.178.11                       # head -> worker rail 2
# full collectives test with the deployment's exact env (both nodes):
tests/nccl_collectives.py
```

Expected: all_reduce and all_gather complete cross-node; anything else is a
cable/IP/netplan issue, not vLLM.
