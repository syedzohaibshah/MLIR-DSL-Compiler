# Error-handling test cases

Small bytecode files. Each one targets one or two edge cases of the runtime.
They test the `Error` infrastructure, the parser, and the VM control flow.

## How to run

Build the test runner once, from `runtime/bytecode-vm`:

```
clang++ -std=c++14 -I src -I src/hal -I ../../emunpu-sim/include \
  src/bytecode.cpp src/io_handle.cpp src/vm_core.cpp \
  src/hal/emunpu_device.cpp src/test_runner.cpp \
  -L ../../emunpu-sim/build -lemunpu -lm -o test_runner
```

Run one test (the runner takes the file path):

```
./test_runner src/test-bytecode/tests/err_bad_number.txt
```

Or run all tests with the helper script (from this folder):

```
./run_all.sh
```

## What the runner does

- Loads the file.
- Sets input slots 0..4, each 16 bytes (four floats).
- Runs the VM on the emunpu device.
- Prints `RESULT: ok` or `RESULT: FAIL - <message> (line N)`.

The runner does NOT check a numeric result. The thing under test is the
`Error` (or `ok`) that comes back.

## The cases

A name that starts with `err_` must FAIL. A name that starts with `ok_` must
pass with `RESULT: ok`.

| File | Case tested | Expected |
|------|-------------|----------|
| `err_bad_number.txt` | a non-number token (`abc`) in an argument | FAIL (invalid integer) |
| `err_empty_register.txt` | `@` with no register number | FAIL (register argument is empty) |
| `err_number_trailing_junk.txt` | trailing junk on a number (`16x`) | FAIL (invalid integer) |
| `err_unknown_opcode.txt` | an opcode name that does not exist | FAIL (unknown opcode) |
| `err_wrong_arg_count.txt` | `SET` with one argument, not two | FAIL (unknown opcode or wrong arg count) |
| `err_br_negative.txt` | `BR` to a negative index | FAIL (BR target out of bounds) |
| `err_br_too_big.txt` | `BR` past the last instruction | FAIL (BR target out of bounds) |
| `err_condbr_unset_register.txt` | `COND_BR` reads a register never set | FAIL (out of range register id) |
| `err_register_negative.txt` | `INC` on a negative register id | FAIL (out of range register id) |
| `err_alloc_negative_size.txt` | `ALLOC` with a negative size | FAIL (negative size, VM check) |
| `err_alloc_not_multiple_of_4.txt` | `ALLOC 7` (not a multiple of 4) | FAIL (emunpu size check) |
| `err_copy_size_mismatch.txt` | `COPY_H2D` size does not match the input slot | FAIL (size mismatch) |
| `err_copy_from_bad_buffer.txt` | `COPY_D2H` from a buffer never allocated | FAIL (invalid buffer id) |
| `err_free_bad_buffer.txt` | `FREE` a buffer never allocated | FAIL (invalid buffer id) |
| `err_double_free.txt` | `FREE` the same buffer twice | FAIL (buffer already freed) |
| `err_launch_unknown_op.txt` | `LAUNCH` with an op id that does not exist | FAIL (unknown op_id) |
| `err_launch_matmul_wrong_shape.txt` | `LAUNCH` MATMUL with wrong buffer/scalar count | FAIL (wrong number of buffers/scalars) |
| `ok_comments_and_blanks.txt` | comment and blank lines are skipped | ok |
| `ok_minimal.txt` | alloc, copy a valid input, free, eof | ok |
| `ok_eof_in_middle.txt` | `EOF` mid-file halts before later (bad) lines run | ok |

## A note learned while writing these

Parsing and execution are two separate phases. `loadFromFile` parses the WHOLE
file into opcodes first. `execute` runs them after. So `EOF` in the middle only
stops EXECUTION of later lines -- every line is still PARSED at load time.
A syntax error after `EOF` therefore still fails at parse time. `ok_eof_in_middle.txt`
uses valid syntax after `EOF`, with a double-free that would fail only if executed.
