# Stress / production-level test cases

These attack the runtime the way a hostile or careless input would in production:
integer overflow, resource exhaustion, hardware limits, size-contract violations,
adversarial text, and large valid workloads. The goal is to FIND GAPS, not to
confirm known behaviour.

## How to run

Same runner as `../tests/`. Point `run_all.sh` at this folder, or copy it here:

```
cd ../tests && RUNNER=../../../test_runner ./run_all.sh   # runs ../tests
```

To run this folder, copy the script and run from here:

```
cp ../tests/run_all.sh .
RUNNER=../../../test_runner ./run_all.sh
```

## IMPORTANT — read before trusting PASS/FAIL

The runner judges by the file name: `err_*` must fail, `ok_*` must pass. For
many stress tests the correct outcome is NOT known in advance. The name is my
BEST GUESS. So on this folder:

- A `FAIL` line may mean the VM has a gap, OR it may mean my guess in the name
  was wrong. Both are findings. Check each against the code before you "fix" it.
- Treat every row below marked **PROBE** as an open question, not a known bug.

## Confident expectations (name should match real behaviour)

| File | Attack | Why it should fail |
|------|--------|--------------------|
| `err_int_max_alloc.txt` | `ALLOC 2147483647` | far past the 256 MB global mem |
| `err_alloc_over_global_mem.txt` | `ALLOC 300 MB` | over the 256 MB global mem (the device is created with capacity 0, which means the 256 MB default) |
| `err_alloc_exhaust_mem.txt` | 600 x 512 KB allocs (300 MB) | exhausts the 256 MB global mem partway |
| `err_launch_after_free.txt` | LAUNCH on a freed buffer | `isValidHandle` sees `freed` |
| `err_bare_register_as_opcode.txt` | `@5` as an opcode | unknown opcode |
| `err_reg_huge_id.txt` | `SET 2000000000 1` | `registers.resize` to ~2e9 ints = ~8 GB |
| `ok_matmul_16x16_tile.txt` | real 16x16 MMA tile | valid, exercises the tensor core. No upload: the runner's input slots hold only 16 bytes, so the operands are not initialised. This checks the launch, not the values. |
| `ok_br_to_eof.txt` | `BR` to the EOF line | lands on EOF, halts clean |
| `ok_leading_whitespace.txt` | spaces before opcode | trimmed, parses |
| `ok_trailing_whitespace.txt` | spaces after args | ignored by `>>` |
| `ok_tabs_between_tokens.txt` | tabs between tokens | `>>` splits on any ws |

## PROBE — outcome UNCERTAIN (the interesting ones)

| File | Open question |
|------|---------------|
| `err_int_overflow_alloc.txt` | `strtol` on a value > LONG_MAX sets ERANGE and clamps. `stringToInteger` does NOT check ERANGE -> is the overflow caught, or silently clamped to LONG_MAX? **Likely a GAP.** |
| `err_digit_flood.txt` | 5000-digit number -> ERANGE again. Same gap. |
| `err_reg_int_min_underflow.txt` | `DEC` at INT_MIN. Signed overflow is UB. Does it crash, wrap, or error? **Likely a GAP (no overflow guard).** |
| `err_br_int_max.txt` | `BR 2147483647` -> `next_index` check is `> opcode_size`, so caught. But the value is stored in an `int`; fine. Probably err. |
| `err_condbr_int_max.txt` | same as above via COND_BR. |
| `err_copy_exceeds_buffer.txt` | upload 64 B into a 16 B buffer. It fails, but for the WRONG reason: the runner's slot holds 16 B, so the slot-size check stops it first. The device check is never reached. emunpu itself does reject an oversize upload (`OUT_OF_RANGE`); CUDA would not. |
| `err_download_exceeds_buffer.txt` | download 64 B from a 16 B buffer. emunpu returns `OUT_OF_RANGE`, so this fails correctly. The VM itself does not check it, so a CUDA run would read past the buffer. |
| `err_copy_zero_bytes.txt` | is a 0-byte copy rejected anywhere? **Probe.** |
| `err_copy_negative_size.txt` | negative copy size. `size` is `int`; emunpu takes it as a count. Caught or not? **Probe.** |
| `err_line_too_many_args.txt` | 10000 args on one line -- parser stress, memory. Should parse ok (then run), so `err_` may be the WRONG label. **Probe.** |
| `err_crlf_line_endings.txt` | `\r` stuck on the last token (`16\r`). Does `strtol` reject `16\r`? (trailing junk) -> probably err, but confirm. |
| `err_embedded_nul.txt` | a NUL byte mid-line. `getline` keeps it; `>>` and `c_str()` behaviour is the question. **Probe.** |
| `err_condbr_self_loop.txt` | `COND_BR 0 1 1`, reg stays 0 -> infinite self-loop -> step limit. SLOW (1e8 steps). Ends as err, but takes seconds. |
| `ok_download_uninitialised.txt` | download a buffer never written. Does emunpu return zeros (ok) or error? **Probe.** If it errors, the `ok_` label is wrong. |

## What to do with the results

1. Run the folder. Note every `FAIL`.
2. For each `FAIL`, decide: is the VM wrong, or was my name-guess wrong?
   - VM wrong -> a real gap. Add support (likely: ERANGE check in
     `stringToInteger`; copy-size vs buffer-size check in the VM; INC/DEC
     overflow guard).
   - Name-guess wrong -> rename the file so the convention matches real,
     correct behaviour.
3. The GAP candidates I would bet on, in order:
   - `stringToInteger` ignores `strtol` ERANGE (overflow silently clamped).
   - COPY size is never checked against the device buffer's allocated size.
   - No INC/DEC overflow guard.

## Correction (2026-10-31)

Earlier versions of this file said emunpu global memory is 4 MB. That is wrong for our runtime. `emunpu_device_create(0, ...)` passes capacity 0, which means `IREE_EMUNPU_DEFAULT_GLOBAL_MEM_BYTES` = **256 MB**. The 4 MB value (`IREE_EMUNPU_GLOBAL_MEM_BYTES`) applies only to emunpu's IREE HAL path. The memory tests were changed to sizes above 256 MB.

`err_copy_over_dma_limit.txt` was removed. A host upload is a `memcpy` into global memory. The 64 KB DMA limit applies to DMA instructions inside the device, not to `COPY_H2D`. The test only failed because the runner's slot is 16 bytes.
