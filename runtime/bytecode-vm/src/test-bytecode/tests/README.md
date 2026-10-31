# Runtime test cases

Small bytecode files. Each one targets one or two things. Together they test the
whole runtime: the parser, the VM control flow, the registers, the device
boundary, and real kernel compute. The goal is not only to confirm the error
handling we added. The goal is to run the whole VM and find where it lacks.

## How to run

Build the runner once, from `runtime/bytecode-vm`:

```
clang++ -std=c++14 -I src -I src/hal -I ../../emunpu-sim/include \
  src/bytecode.cpp src/io_handle.cpp src/vm_core.cpp \
  src/hal/emunpu_device.cpp src/test_runner.cpp \
  -L ../../emunpu-sim/build -lemunpu -lm -o test_runner
```

Run all tests from this folder:

```
./run_all.sh
```

The summary prints one line per test: `PASS` / `FAIL`, the name, and a short
detail. The noisy device logs go to `detail.log`, so the table stays readable.

Run one test directly:

```
../../../test_runner err_parse_bad_number.txt
```

## The naming rule

The runner decides the expected result from the file name:

- `err_*` -> the VM MUST return an Error. PASS means it failed as intended.
- `ok_*`  -> the VM MUST return ok. PASS means it ran clean.

So `PASS` always means "behaved as the name promised", for both kinds.

## Groups

- **A. Parser** — bad number, empty register, trailing junk, double `@`, float,
  comments and blanks.
- **B. Opcodes** — unknown name, case sensitivity, wrong argument count.
- **C. Program end** — no `EOF`, `EOF` mid-file, `EOF` with an argument, empty file.
- **D. Alloc / free** — negative/zero/odd size, double free, free unallocated,
  use-after-free, many allocs.
- **E. Copy** — size mismatch, bad input slot, bad buffer, valid round trip.
- **F. Registers** — read before set, negative id, set/inc/dec, `@reg` indirection.
- **G. Control flow** — `BR` out of range, `COND_BR` target/register, forward
  skip, a real countdown loop, and an infinite loop (must hit the step limit).
- **H. LAUNCH / kernels** — unknown op, wrong shape, bad buffer, truncated
  header, negative count, plus real RELU / VEC_ADD / MATMUL runs.
- **I. End to end** — a small matmul -> relu -> download program.

## Probe tests (expected result UNCERTAIN -- read these first)

These were written to find gaps, not to confirm known behaviour. If one does not
behave as guessed, that is a finding, not a broken test. Check each against the
code and decide what the VM SHOULD do.

| File | Question it probes | Guess |
|------|--------------------|-------|
| `err_parse_float_value.txt` | Does `strtol` reject `1.5`? (It reads `1`, leaves `.5` -> trailing junk) | FAIL |
| `err_parse_double_at.txt` | `@@3` -- does the empty-register / strtol guard catch it? | FAIL |
| `err_op_lowercase.txt` | Are opcode names case sensitive? `alloc` should be unknown | FAIL |
| `err_op_alloc_two_args.txt` | `ALLOC 16 32` -- arity is checked by `args.size()==1` | FAIL |
| `err_op_set_too_many.txt` | `SET 0 1 2` -- too many args | FAIL |
| `ok_end_no_eof.txt` | No `EOF` line. Does the loop fall off the end cleanly? | ok |
| `ok_end_empty_file.txt` | Empty file -- load ok, execute nothing? | ok |
| `err_end_eof_with_arg.txt` | `EOF 1` -- dispatch wants `args.size()==0` | FAIL |
| `err_alloc_zero.txt` | Is `ALLOC 0` rejected? VM checks `size < 0` only, not `<= 0` | UNCERTAIN |
| `err_use_after_free.txt` | Copy into a freed buffer -- is it caught? | FAIL |
| `err_condbr_bad_target.txt` | Is the `COND_BR` jump target range-checked in dispatch? | FAIL |
| `ok_reg_indirect_alloc.txt` | Does `@reg` resolve as the ALLOC size argument? | UNCERTAIN |
| `err_loop_infinite.txt` | `BR 0` forever -- does the step limit stop it? | FAIL (slow) |

Note on `err_loop_infinite.txt`: the step limit is 1e8, so this test runs for a
while before it stops. Expect a short wait.

## A design note (learned earlier)

Parse and execute are two phases. `loadFromFile` parses the WHOLE file into
opcodes first; `execute` runs them after. So `EOF` mid-file only stops EXECUTION
of later lines -- every line is still PARSED. A syntax error after `EOF` still
fails at parse time. `ok_end_eof_in_middle.txt` therefore uses valid syntax after
`EOF` (a double free that would fail only if executed).
