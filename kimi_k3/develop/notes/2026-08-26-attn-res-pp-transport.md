# 2026-08-26 — AttnRes pipeline transport

`decision` · supersedes the incoming plan's "the schedule carries both
`prefix_sum` and `block_residual` (and their gradients)" design.

## The question

Kimi K3's decoder carries two tensors, not one: `prefix_sum` (`[S,B,H]`) and
`block_residual` (`[S,B,K,H]`, `K` growing by one slot every 12 layers). Both
must cross every pipeline boundary, and both must receive gradients. How do we
do that without touching `megatron/**`?

## What core actually does (verified at `a1b00d4`)

1. `forward_backward_pipelining_without_interleaving` computes one recv-shape
   list and one send-shape list, then hands them to an optional
   `adjust_tensor_shapes_fn(recv, send)` — `schedules.py:1975-2004, 2019,
   2164-2183`. **The function runs on each rank**, so recv and send shapes may
   legitimately differ from stage to stage.
2. The hook is asserted `None` by the no-pipelining schedule
   (`schedules.py:631-632`) and by the interleaved/VPP schedule (`:949-950`).
3. `train_step` only ever binds the hook for NVIDIA-modelopt distillation
   (`training.py:1815-1822, 1853`).
4. **`backward_step` back-props only `output_tensor[0]` with
   `output_tensor_grad[0]`** (`schedules.py:451-493`), with a comment saying it
   "can handle at most one skip connection". Extra tensors in the payload list
   are forward-only — their received gradients are dropped on the floor.

## The consequences

- (4) rules out a two-tensor payload. It would train, the loss would fall, and
  the model would be wrong: `block_residual`'s contribution to the loss would
  never reach the stages that produced it. **This is the failure mode the design
  must be built to exclude, and it has to be tested for explicitly** — nothing
  in core will complain.
- (1) means we do **not** need a custom schedule. The incoming plan proposed
  wrapping or replacing `get_forward_backward_func()`; that is hundreds of lines
  of duplicated 1F1B that breaks on every IFU, to obtain something core already
  offers.
- (2) confirms the VPP descope, and adds a second constraint the plan missed:
  the hook must not be bound at `PP == 1` either, or every single-GPU test
  asserts.
- (3) means the binding is ours to install, from our own entry point.

## The design

**One packed tensor**, concatenated along the sequence axis so the payload stays
a `(seq, mbs, hidden)` triple that `p2p_communication` already handles:

```
packed = cat([prefix_sum,                       # [S, B, H]
              block_residual.permute(2,0,1,3)   # [K, S, B, H]
                            .reshape(K*S, B, H)], dim=0)   # -> [(1+K)*S, B, H]
```

`cat` / `view` / `permute` are differentiable, so a single
`torch.autograd.backward(packed, grad)` delivers gradients to `prefix_sum` and
to every slot. Unpacking is the exact inverse and equally transparent.

Per-stage shapes come from the local layer range:

```
slots_before(l) = ceil(l / block_size)          # slots visible on entry to layer l
recv_multiplier = 1 + slots_before(first_local_layer)
send_multiplier = 1 + slots_before(last_local_layer + 1)
```

Neighbour consistency is automatic because stage `s`'s
`last_local_layer + 1` is stage `s+1`'s `first_local_layer`.

## Costs (analytic; to be confirmed by gate G6)

At `S=8192, B=1, H=7168, bf16` the unit tensor is 117 MB, so boundary payloads
run 235 MB → 1.06 GB as `K` goes 1 → 8. Under 1F1B the in-flight count at stage
`s` is `PP − s`, which roughly cancels the growth: ≈ 2 GB/GPU of in-flight
payload at PP=8 (≈ 4 GB counting saved inputs and outputs). Context parallelism
divides `S` and therefore divides this directly.

Placing stage boundaries at 0-indexed layer ≡ 11 (mod 12) — i.e. just *before* an
append layer — costs one slot less per boundary than the alternative, which is
why the recommended 93-layer PP=8 layout is `12×7 + 9`.

## Test that encodes this decision

`kimi_k3/tests/test_k3_p5_payload_grad.py` (gate **G20**): perturb a slot of
`block_residual` on the later stage and assert a non-zero gradient arrives at the
earlier stage. Validate the test itself by temporarily reverting to a two-tensor
payload and confirming it goes red — a gate that cannot fail is not a gate.

## Status

Prototyped in P0-T0.7 (gate G7), shipped in P5-T5.2/T5.5.
