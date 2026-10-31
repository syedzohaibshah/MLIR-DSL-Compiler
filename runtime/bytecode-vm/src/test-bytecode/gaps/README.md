# Gap tests

Each file here describes the behaviour the runtime **should** have for one open issue. Most of them **fail today**. A `FAIL` here is a known open issue, not a broken test. When the issue is fixed, its test changes to `PASS`.

The naming rule is the same as in `../tests/`: `err_*` must return an Error, `ok_*` must return ok.

## How to run

From `runtime/bytecode-vm/`:

```
cmake --build build
src/test-bytecode/gaps/run_all.sh
```

One file:

```
./build/test_runner src/test-bytecode/gaps/ok_free_indirect.txt
```

## The cases

Indices in the explanations are 0-based instruction indices, the same as `BR` and `COND_BR` use.

### A. `@N` does not work for every operand

`dispatch()` reads a register for some operands only. These files use `@N` where it is ignored today.

| File | Operand | Today | After the fix |
|---|---|---|---|
| `ok_free_indirect.txt` | `FREE @0` | frees buffer 0 (the register number), then `FREE 0` is a double free | frees buffer 1, then buffer 0 |
| `ok_copy_h2d_size_indirect.txt` | `COPY_H2D` size | size becomes 5 (the register number), size mismatch | size is 16 |
| `ok_copy_d2h_size_indirect.txt` | `COPY_D2H` size | size becomes 100, emunpu returns out of range | size is 16 |
| `ok_br_indirect_target.txt` | `BR` target | target becomes 50, out of bounds | jumps to index 3 (`EOF`) |
| `ok_condbr_indirect_target.txt` | `COND_BR` target | target becomes 50, out of bounds | jumps to index 4 (`EOF`) |
| `ok_set_value_from_register.txt` | `SET` value | writes 0, then `ALLOC @1` has size 0 | copies 16 from register 0 |

`ok_set_value_from_register.txt` depends on a design decision: does `SET 1 @0` mean "copy register 0 into register 1"? If the team decides `@` is not allowed in `SET`, change this file to an `err_` test.

### B. Buffer lifecycle

| File | Issue | Today | After the fix |
|---|---|---|---|
| `ok_cleanup_loop_frees_all.txt` | the cleanup loop of `bytecode12*.txt`, in small form | `FREE @4` frees buffer 4, which does not exist | frees buffers 2, 1, 0 |
| `err_leak_buffer_not_freed.txt` | a buffer is never freed | runs ok, the leak is silent | **depends on the ownership policy** (see `runtime/context.md`). If a leak is an error, this passes. If a leak is a warning, rename it to `ok_`. |

### C. Host I/O

| File | Issue | Today | After the fix |
|---|---|---|---|
| `err_copy_d2h_huge_output_slot.txt` | output slot id 2000000000 | `prepareOutputSlot` resizes to ~2e9 entries and the process crashes (`std::bad_alloc`), so no verdict line prints | clean Error: output slot out of range |

### D. LAUNCH dimensions are not checked against buffer sizes

Every buffer below is 16 bytes (4 floats) unless stated. emunpu does not compare the dimensions with the tensor size, so the kernel reads and writes past the end of the tensor, into other tensors in the same global memory. The run returns ok. That is silent memory corruption.

| File | Launch | Needs | Has |
|---|---|---|---|
| `err_launch_matmul_dims_exceed_buffers.txt` | MATMUL M=N=K=16 | 1024 bytes per operand | 16 |
| `err_launch_relu_count_exceeds_buffer.txt` | RELU count=1000 | 4000 bytes | 16 |
| `err_launch_vec_add_count_exceeds_buffer.txt` | VEC_ADD count=1000 | 4000 bytes | 16 |
| `err_launch_bias_period_exceeds_bias.txt` | BIAS_ADD out_features=100 | bias of 400 bytes | 8 |
| `ok_launch_matmul_nonsquare.txt` | MATMUL 2x4 by 4x3 | 32, 48, 24 bytes | exactly that |

`ok_launch_matmul_nonsquare.txt` is a **guard**. It should pass today, and it must still pass after the dimension checks are added. It stops a check that is too strict.

## Tests that are not here

Some issues cannot be reached from a bytecode file, because `test_runner` does not call `getOutput` or `checkInputSize`. Those are in `src/unit_tests.cpp`. Run them with `./build/unit_tests`.
