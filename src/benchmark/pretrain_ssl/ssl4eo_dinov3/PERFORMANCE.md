# Training performance notes (data-loading I/O)

Hard-won notes on keeping the GPUs fed during DINOv3 pretraining on the SSL4EO-S12
LMDB. Read this first if a run is slow or `nvidia-smi` shows the GPUs idling.

## TL;DR

- The bottleneck is almost never compute — it's **reading ~3.6 MB samples out of a
  ~624 GB LMDB on a network PersistentDisk**.
- The single biggest win was `readahead=True` in `datasets.py`'s `lmdb.open`
  (see below). It took a 2× A100-40GB run from **~4.1 s/iter (ETA ~3 days) to
  ~0.9 s/iter (ETA ~1 day)** with no other changes.
- After that fix you are **disk-bandwidth-bound** (~0.76 GB/s sustained, the
  practical ceiling for this VM/disk). Further large gains require a **local SSD**,
  not config tweaks.

## How to tell where the bottleneck is

Run these while training:

```bash
# 1. Are the GPUs actually busy? Sample over time, not a single shot.
for i in $(seq 1 15); do \
  nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits; \
  sleep 1; done
#   GPUs flat at 0% / power near idle (~60W on A100) / low mem  => data-starved.

# 2. What does the per-iteration breakdown say?
tail -3 /data/out/<run>/training_metrics.json   # look at iter_time vs data_time
#   data_time ~= iter_time  => I/O-bound. data_time << iter_time => compute-bound.

# 3. What is the disk doing?
iostat -x 2 3 | grep -E 'Device|sda'
#   rareq-sz ~4 KB  => tiny random reads, NO readahead (the bad case).
#   rareq-sz tens-hundreds of KB => readahead working, you're bandwidth-bound.

# 4. Dataloader worker state (lots of 'D' = uninterruptible sleep = blocked on I/O)
ps aux | grep train_ssl4eo | awk '$8 ~ /D/'
```

## The fix that mattered: LMDB readahead

`datasets.py` opens the LMDB env. Upstream DINOv3 uses `readahead=False`
(`MDB_NORDAHEAD`) because ImageNet samples are tiny (~100 KB JPEGs) and readahead
would waste bandwidth pulling unrelated neighboring data.

**Our samples are the opposite case**: each value is a ~3.6 MB contiguous cube
(`seasons × 13 bands × H × W`). With `readahead=False`, every sample was faulted in
as ~900 individual 4 KB page reads — pure latency/IOPS-bound access. `iostat`
showed exactly that: 25,000 reads/s × 4 KB = ~100 MB/s, pinned at the disk's
IOPS cap (pd-balanced ≈ 6 read-IOPS/GB × 4 TB ≈ 24.5k IOPS).

Setting `readahead=True` lets the kernel coalesce each sample into a few large
sequential reads. IOPS per sample drops ~30×, throughput jumped from ~100 MB/s to
~760–980 MB/s, and the workload flipped from IOPS-bound to bandwidth-bound.

Reinforce it at the block layer (non-persistent; re-apply after reboot, needs sudo):

```bash
echo 2048 | sudo tee /sys/block/sda/queue/read_ahead_kb   # default is 128
```

## Measured results (2× A100-40GB, a2-highgpu-2g, ViT-S/16, global batch 128)

| state                         | iter_time | data_time | disk read | ETA      |
| ----------------------------- | --------- | --------- | --------- | -------- |
| `readahead=False` (original)  | ~4.1 s    | ~3.0 s    | 4 KB reads, ~100 MB/s | ~3–5 days |
| `readahead=True` + read_ahead_kb=2048 | ~0.9 s | ~0.4 s | ~80–250 KB reads, ~0.76 GB/s | ~1 day |

## Why ~1 day is the floor on this hardware

Per iteration you read `128 samples × 3.6 MB ≈ 461 MB`. At the VM's sustained
~0.76 GB/s that's ~0.6 s of reading vs ~0.5 s of GPU compute, so the disk can't
*quite* keep the GPUs saturated — hence the residual `data_time` and the periodic
dips to 0% util. That ~0.76 GB/s is near the per-VM PersistentDisk throughput
ceiling (it scales with vCPU count; this VM has 24), so you cannot tune your way
past it.

### Levers, in order of value

1. **Local SSD (only thing that breaks the bandwidth wall).** Striped NVMe local
   SSD does multiple GB/s and would make the run compute-bound (~0.5 s/iter,
   ~12 h). Caveats: local SSD is **creation-time only** (needs a new VM), is
   **ephemeral** (wiped on stop/delete — and A2 GPU VMs are *terminated*, not
   live-migrated, on host maintenance, which wipes it), needs ≥2× 375 GB disks
   striped for the 624 GB LMDB, and requires a one-time ~624 GB copy from durable
   storage. **Checkpoints must go to the boot/persistent disk or GCS, never the
   local SSD.**
2. **`num_workers` bump (cheap, marginal).** Disk `%util` sat ~47%, so queue depth
   was low; raising `num_workers` 10 → ~14 may pull sustained bandwidth a bit
   closer to the cap. Restart required.
3. **Things that DON'T help while disk-bound:**
   - Larger `batch_size_per_gpu` — only helps once the GPU is the bottleneck. VRAM
     is wide open (~13/40 GB used), so it's available *after* the I/O is fixed.
   - `pd-ssd` — its throughput ceiling is barely above pd-balanced; not worth a
     snapshot+recreate.

## Restart gotchas

- **`EADDRINUSE` on port 29500**: a previous run is still alive and holding the
  rendezvous port (and the GPUs). Kill it first — relaunching does *not* take over:
  ```bash
  pkill -f train_ssl4eo.py && sleep 5
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader  # confirm freed
  ```
  Wait until GPU memory is released before relaunching, or you'll hit CUDA
  out-of-memory / device-busy instead. (Or pass `MASTER_PORT=29501` for a second
  concurrent run.)
- On resume, training picks up the latest checkpoint in `/data/out/<run>/ckpt`.
  Clear it if you want a clean start.

## Launch reference

```bash
# read_ahead_kb tweak (optional, needs sudo) then launch:
DATASET_SIZE=172348 bash ssl4eo_dinov3/scripts/train_gcp.sh \
  /data/ssl4eo_s2c_uint8.lmdb /data/out/vits16 \
  configs/ssl4eo_s2_vits16.yaml 64 --skip-errors
```
