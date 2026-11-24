// Unit tests for the parts that bytecode test files cannot reach:
//   - the host API (setInput, getOutput, and their signature checks), and
//   - the NUMERIC results of the kernels (the bytecode tests only check
//     whether a run returns an Error, never what it computed).
//
// Each test builds its program in C++ (no .txt file), runs it on a new emunpu
// device, and prints PASS or FAIL with a reason.
//
// Build: the CMake target `unit_tests`.
// Run (from runtime/bytecode-vm/):   ./build/unit_tests
//
// The LAST test (setInput with a negative slot) used to CRASH, because
// IOHandler::setInput wrote input_slots[-1]. The signature check now rejects
// the slot first. It stays last, so a crash would not hide the other results.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <string>
#include <vector>
#include "bytecode.h"
#include "vm_core.h"
#include "hal/emunpu_device.h"
#include "error.h"

static int g_pass = 0;
static int g_fail = 0;

static void result(const char* name, bool passed, const std::string& why){
    if(passed){ std::printf("PASS  %s\n", name); g_pass++; }
    else      { std::printf("FAIL  %s  ->  %s\n", name, why.c_str()); g_fail++; }
    std::fflush(stdout);
}

// Builds one instruction whose arguments are all literals (no @N).
static opcode op(const std::string& name, std::initializer_list<int> values, int line){
    opcode o;
    o.name = name;
    for(int v : values) o.args.push_back(Arg{v, false});
    o.line_number = line;
    return o;
}

// Encodes a float as the bit pattern a FLOAT_BITS scalar carries. memcpy, not a
// cast: a cast would convert the VALUE (e.g. (int)0.2f == 0), losing it. This
// copies the raw 4 bytes instead, so the device's own memcpy decodes them back
// to the exact same float.
static int floatBits(float f){
    int bits;
    std::memcpy(&bits, &f, sizeof(float));
    return bits;
}

// Compares two float arrays with a small tolerance.
static bool sameFloats(const float* got, const float* want, int n, std::string& why){
    for(int i = 0; i < n; i++){
        if(std::fabs(got[i] - want[i]) > 1e-4f){
            why = "element " + std::to_string(i) + ": got " + std::to_string(got[i])
                + ", want " + std::to_string(want[i]);
            return false;
        }
    }
    return true;
}

// Binds a signature to the VM. `header` holds the .input and .output lines.
// The helper writes a small program with that header to a temp file, loads it,
// and binds it. It is the same path the real programs take.
static Error bindHeader(VM& vm, const std::string& header){
    const char* path = "unit_tests_tmp.txt";
    {
        std::ofstream f(path);
        f << ".version 1\n" << header << "EOF\n";
    }
    Bytecode bc;
    Error load = bc.loadFromFile(path);
    std::remove(path);
    if(!load.ok) return load;
    return vm.bindSignature(bc);
}

// ---------------------------------------------------------------------------
// Host API
// ---------------------------------------------------------------------------

static void test_round_trip(){
    const char* name = "round_trip_returns_the_same_bytes";
    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 16 in\n.output 0 16 out\n").ok){
        result(name, false, "could not bind signature"); return;
    }

    float in[4]  = {1.5f, -2.0f, 3.25f, 0.0f};
    float out[4] = {0, 0, 0, 0};
    std::vector<opcode> prog = {
        op("ALLOC",    {16},       1),
        op("COPY_H2D", {0, 0, 16}, 2),
        op("COPY_D2H", {0, 0, 16}, 3),
        op("FREE",     {0},        4),
        op("EOF",      {},         5),
    };
    Error e = vm.setInput(0, in, sizeof(in));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, out, sizeof(out));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(out, in, 4, why), why);
}

// Issue: "When the device download fails, getOutput still reports success".
static void test_failed_download_blocks_getOutput(){
    const char* name = "failed_download_makes_getOutput_fail";
    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".output 0 16 out\n").ok){
        result(name, false, "could not bind signature"); return;
    }

    std::vector<opcode> prog = {
        op("ALLOC",    {16},       1),
        op("COPY_D2H", {0, 5, 16}, 2),   // buffer 5 does not exist -> download fails
        op("EOF",      {},         3),
    };
    Error run = vm.execute(prog);
    if(run.ok){ result(name, false, "execute should fail on buffer 5"); return; }

    float out[4];
    Error get = vm.getOutput(0, out, sizeof(out));
    result(name, !get.ok, "getOutput returned ok for a slot whose download failed");
}

static void test_getOutput_wrong_size(){
    const char* name = "getOutput_with_wrong_size_fails";
    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 16 in\n.output 0 16 out\n").ok){
        result(name, false, "could not bind signature"); return;
    }

    float in[4] = {1, 2, 3, 4};
    std::vector<opcode> prog = {
        op("ALLOC",    {16},       1),
        op("COPY_H2D", {0, 0, 16}, 2),
        op("COPY_D2H", {0, 0, 16}, 3),
        op("FREE",     {0},        4),
        op("EOF",      {},         5),
    };
    Error e = vm.setInput(0, in, sizeof(in));
    if(e.ok) e = vm.execute(prog);
    if(!e.ok){ result(name, false, "setup failed: " + e.msg); return; }

    float out[2];
    Error get = vm.getOutput(0, out, sizeof(out));   // slot holds 16, ask for 8
    result(name, !get.ok, "getOutput accepted 8 bytes for a 16-byte slot");
}

static void test_getOutput_never_written(){
    const char* name = "getOutput_on_a_slot_never_written_fails";
    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    // Slot 3 is declared, so the failure must come from "never written", not "not declared".
    if(!bindHeader(vm, ".output 3 16 out\n").ok){
        result(name, false, "could not bind signature"); return;
    }

    float out[4];
    Error get = vm.getOutput(3, out, sizeof(out));
    result(name, !get.ok, "getOutput returned ok for slot 3, which no COPY_D2H wrote");
}

// Issue: "An input size mismatch is detected, but the program runs anyway".
// The signature now checks sizes in setInput, so the VM must refuse a wrong size.
static void test_setInput_wrong_size(){
    const char* name = "setInput_rejects_a_size_that_differs_from_the_signature";
    emunpuDevice npu;
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 16 x\n").ok){
        result(name, false, "could not bind signature"); return;
    }

    // Positive control: the declared size must still be accepted.
    float right[4] = {0, 0, 0, 0};
    Error good = vm.setInput(0, right, sizeof(right));
    if(!good.ok){ result(name, false, "16 bytes rejected, but the signature declares 16: " + good.msg); return; }

    float wrong[2] = {0, 0};
    Error bad = vm.setInput(0, wrong, sizeof(wrong));
    result(name, !bad.ok, "setInput accepted 8 bytes for a slot the signature declares as 16");
}

// Issue: "Failed input allocations ... aren't handled". setInput accepts any
// pointer and any size today.
static void test_setInput_null_data(){
    const char* name = "setInput_rejects_null_data";
    emunpuDevice npu;
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 16 in\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    Error e = vm.setInput(0, nullptr, 16);
    result(name, !e.ok, "setInput accepted a null pointer");
}

static void test_setInput_negative_size(){
    const char* name = "setInput_rejects_negative_size";
    emunpuDevice npu;
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 16 in\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    float x[4] = {0, 0, 0, 0};
    Error e = vm.setInput(0, x, -16);
    result(name, !e.ok, "setInput accepted size -16");
}

// ---------------------------------------------------------------------------
// Kernel numerics on emunpu (compare with a CPU reference)
// ---------------------------------------------------------------------------

// Issue: different-size (non-square) matmul. A is 2x4, B is 4x3, C is 2x3.
static void test_matmul_nonsquare(){
    const char* name = "matmul_2x4_by_4x3_matches_cpu";
    const int M = 2, N = 3, K = 4;
    float A[M * K] = {1, 2, 3, 4,
                      5, 6, 7, 8};
    float B[K * N] = {1, 0, 2,
                      0, 1, 0,
                      1, 1, 1,
                      2, 0, 1};
    float want[M * N];
    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++){
            float s = 0;
            for(int k = 0; k < K; k++) s += A[i * K + k] * B[k * N + j];
            want[i * N + j] = s;
        }

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 32 A\n.input 1 48 B\n.output 0 24 C\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {32},                          1),   // A: 8 floats
        op("ALLOC",    {48},                          2),   // B: 12 floats
        op("ALLOC",    {24},                          3),   // C: 6 floats
        op("COPY_H2D", {0, 0, 32},                    4),
        op("COPY_H2D", {1, 1, 48},                    5),
        op("LAUNCH",   {0, 3, 0, 1, 2, 3, M, N, K},   6),
        op("COPY_D2H", {0, 2, 24},                    7),
        op("FREE",     {0},                           8),
        op("FREE",     {1},                           9),
        op("FREE",     {2},                          10),
        op("EOF",      {},                           11),
    };
    float got[M * N] = {0};
    Error e = vm.setInput(0, A, sizeof(A));
    if(e.ok) e = vm.setInput(1, B, sizeof(B));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, M * N, why), why);
}

// M*K*4 bytes overflows int64 for M = K = INT_MAX. The size check must still reject it.
static void test_matmul_huge_dims_rejected(){
    const char* name = "matmul_huge_dims_rejected_on_small_buffers";
    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    DeviceHandle a, b, c;
    if(!npu.allocate(16, 1, a).ok || !npu.allocate(16, 1, b).ok || !npu.allocate(16, 1, c).ok){
        result(name, false, "could not allocate 16-byte buffers"); return;
    }
    const int big = 2147483647;
    Error e = npu.dispatch({0, 3, a, b, c, 3, big, 1, big}, 1);   // M = 2^31-1, N = 1, K = 2^31-1
    result(name, !e.ok, "MATMUL with M = K = INT_MAX on 16-byte buffers was accepted");
}

static void test_relu_values(){
    // Exercises RELU through ACTIVATION (op_id 7, kind=0) rather than op_id 2.
    // Both call the same runActivation body; see do_relu's own op_id 2 LAUNCH
    // in the bytecode .txt tests for the other spelling.
    const char* name = "relu_matches_cpu";
    float in[4]   = {-1.0f, 2.0f, -3.0f, 4.0f};
    float want[4] = { 0.0f, 2.0f,  0.0f, 4.0f};

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 16 in\n.output 0 16 out\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {16},                   1),
        op("ALLOC",    {16},                   2),
        op("COPY_H2D", {0, 0, 16},             3),
        op("LAUNCH",   {7, 2, 0, 1, 4, 4, 0, 0, 0},  4),   // op_id=7, n_buf=2 (a,c), n_scalar=4 (count=4, kind=0 RELU, alpha_bits=0, beta_bits=0)
        op("COPY_D2H", {0, 1, 16},             5),
        op("FREE",     {0},                    6),
        op("FREE",     {1},                    7),
        op("EOF",      {},                     8),
    };
    float got[4] = {0};
    Error e = vm.setInput(0, in, sizeof(in));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, 4, why), why);
}

static void test_silu_values(){
    // SiLU (kind=8): x * sigmoid(x). Ignores alpha/beta, same as RELU, but a
    // different kind index and a non-trivial formula -- catches a kind that is
    // read/cast wrong in a way RELU (kind=0) could hide.
    const char* name = "silu_matches_cpu";
    float in[4] = {-2.0f, 0.0f, 1.0f, 3.0f};
    float want[4];
    for(int i = 0; i < 4; i++){
        float sigmoid = 1.0f / (1.0f + std::exp(-in[i]));
        want[i] = in[i] * sigmoid;
    }

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 16 in\n.output 0 16 out\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {16},                                 1),
        op("ALLOC",    {16},                                 2),
        op("COPY_H2D", {0, 0, 16},                           3),
        op("LAUNCH",   {7, 2, 0, 1, 4, 4, 8, 0, 0},          4),   // count=4, kind=8 SILU, alpha_bits=0, beta_bits=0
        op("COPY_D2H", {0, 1, 16},                           5),
        op("FREE",     {0},                                  6),
        op("FREE",     {1},                                  7),
        op("EOF",      {},                                   8),
    };
    float got[4] = {0};
    Error e = vm.setInput(0, in, sizeof(in));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, 4, why), why);
}

static void test_leaky_relu_values(){
    // LEAKY_RELU (kind=1): x > 0 ? x : alpha * x. Only alpha is used -- this is
    // the FLOAT_BITS round trip's first real exercise with a NON-ZERO value;
    // every test before this only ever sent bits for 0.0f, which could not
    // catch a memcpy direction or sizeof mistake.
    const char* name = "leaky_relu_matches_cpu";
    const float alpha = 0.1f;
    float in[4]   = {-2.0f,  2.0f, -5.0f, 0.0f};
    float want[4] = {-0.2f,  2.0f, -0.5f, 0.0f};

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 16 in\n.output 0 16 out\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {16},                                            1),
        op("ALLOC",    {16},                                            2),
        op("COPY_H2D", {0, 0, 16},                                      3),
        op("LAUNCH",   {7, 2, 0, 1, 4, 4, 1, floatBits(alpha), 0},     4),   // count=4, kind=1 LEAKY_RELU, alpha=0.1, beta unused
        op("COPY_D2H", {0, 1, 16},                                      5),
        op("FREE",     {0},                                             6),
        op("FREE",     {1},                                             7),
        op("EOF",      {},                                              8),
    };
    float got[4] = {0};
    Error e = vm.setInput(0, in, sizeof(in));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, 4, why), why);
}

static void test_clamp_values(){
    // CLAMP (kind=3): min(max(x, alpha), beta). The only kind that uses BOTH
    // FLOAT_BITS scalars at once -- exercises beta_bits, which LEAKY_RELU and
    // ELU never touch.
    const char* name = "clamp_matches_cpu";
    const float lo = -1.0f, hi = 1.0f;
    float in[4]   = {-5.0f, -0.5f, 0.5f, 5.0f};
    float want[4] = {-1.0f, -0.5f, 0.5f, 1.0f};

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 16 in\n.output 0 16 out\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {16},                                                      1),
        op("ALLOC",    {16},                                                      2),
        op("COPY_H2D", {0, 0, 16},                                                3),
        op("LAUNCH",   {7, 2, 0, 1, 4, 4, 3, floatBits(lo), floatBits(hi)},      4),   // count=4, kind=3 CLAMP, alpha=-1, beta=1
        op("COPY_D2H", {0, 1, 16},                                                5),
        op("FREE",     {0},                                                       6),
        op("FREE",     {1},                                                       7),
        op("EOF",      {},                                                       8),
    };
    float got[4] = {0};
    Error e = vm.setInput(0, in, sizeof(in));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, 4, why), why);
}

static void test_elu_values(){
    // ELU (kind=9): x > 0 ? x : alpha * (exp(x) - 1). The third and last
    // alpha-only kind -- together with LEAKY_RELU this confirms alpha is read
    // the same way regardless of which kind uses it, not special-cased per kind.
    const char* name = "elu_matches_cpu";
    const float alpha = 0.5f;
    float in[4] = {-2.0f, -0.1f, 0.0f, 3.0f};
    float want[4];
    for(int i = 0; i < 4; i++){
        want[i] = in[i] > 0.0f ? in[i] : alpha * (std::exp(in[i]) - 1.0f);
    }

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 16 in\n.output 0 16 out\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {16},                                          1),
        op("ALLOC",    {16},                                          2),
        op("COPY_H2D", {0, 0, 16},                                    3),
        op("LAUNCH",   {7, 2, 0, 1, 4, 4, 9, floatBits(alpha), 0},   4),   // count=4, kind=9 ELU, alpha=0.5, beta unused
        op("COPY_D2H", {0, 1, 16},                                    5),
        op("FREE",     {0},                                           6),
        op("FREE",     {1},                                           7),
        op("EOF",      {},                                            8),
    };
    float got[4] = {0};
    Error e = vm.setInput(0, in, sizeof(in));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, 4, why), why);
}

static void test_sigmoid_values(){
    // SIGMOID (kind=4): 1 / (1 + exp(-x)). No alpha/beta, like RELU/SiLU, but
    // this one is used directly for Llama-3's gating -- worth its own test
    // rather than relying on SiLU alone to cover "no-param kinds."
    const char* name = "sigmoid_matches_cpu";
    float in[4] = {-4.0f, -1.0f, 0.0f, 2.0f};
    float want[4];
    for(int i = 0; i < 4; i++){
        want[i] = 1.0f / (1.0f + std::exp(-in[i]));
    }

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 16 in\n.output 0 16 out\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {16},                         1),
        op("ALLOC",    {16},                         2),
        op("COPY_H2D", {0, 0, 16},                   3),
        op("LAUNCH",   {7, 2, 0, 1, 4, 4, 4, 0, 0}, 4),   // count=4, kind=4 SIGMOID, alpha_bits=0, beta_bits=0
        op("COPY_D2H", {0, 1, 16},                   5),
        op("FREE",     {0},                          6),
        op("FREE",     {1},                          7),
        op("EOF",      {},                           8),
    };
    float got[4] = {0};
    Error e = vm.setInput(0, in, sizeof(in));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, 4, why), why);
}

// Error-path tests below use npu.dispatch() directly (no VM/bytecode file),
// the same pattern test_matmul_huge_dims_rejected() uses -- these only check
// that a bad LAUNCH is rejected, not what it computes.

static void test_activation_kind_out_of_range_rejected(){
    // checkScalar's ENUM case must reject kind >= EMUNPU_ACTIVATION_KIND_COUNT
    // (24). This is the direct regression test for the stale "20" bug found
    // and fixed earlier: kind=24 is one past the real last kind (23, ERF).
    const char* name = "activation_kind_24_rejected";
    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    DeviceHandle a, c;
    if(!npu.allocate(16, 1, a).ok || !npu.allocate(16, 1, c).ok){
        result(name, false, "could not allocate 16-byte buffers"); return;
    }
    Error e = npu.dispatch({7, 2, a, c, 4, 4, 24, 0, 0}, 1);   // kind=24, one past ERF=23
    result(name, !e.ok, "ACTIVATION with kind=24 (out of range) was accepted");
}

static void test_activation_count_exceeds_buffer_rejected(){
    // requireElements must still catch a count that overflows the allocated
    // buffer, the same guarantee every other kernel gives -- ACTIVATION should
    // not be an exception just because it is the newest kernel.
    const char* name = "activation_count_exceeds_buffer_rejected";
    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    DeviceHandle a, c;
    if(!npu.allocate(16, 1, a).ok || !npu.allocate(16, 1, c).ok){   // 16 bytes = 4 floats each
        result(name, false, "could not allocate 16-byte buffers"); return;
    }
    Error e = npu.dispatch({7, 2, a, c, 4, 1000, 0, 0, 0}, 1);   // count=1000 on 4-float buffers
    result(name, !e.ok, "ACTIVATION with count=1000 on 16-byte buffers was accepted");
}

// VECTOR_OP (op_id 8), binary kind. EW_MUL = kind 2, count=4, bcast_period=0
// (no broadcast, equal shape -- the common case VECTOR_OP_SCALARS's PARAM fix
// exists for). This is the kind the "hardcoded b=0" bug (fixed this session)
// would have silently gotten wrong: EW_MUL needs a real B operand.
static void test_vector_op_mul_values(){
    const char* name = "vector_op_mul_matches_cpu";
    float a[4]    = {1, 2, 3, 4};
    float b[4]    = {10, 20, 30, 40};
    float want[4] = {10, 40, 90, 160};

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 16 a\n.input 1 16 b\n.output 0 16 c\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {16},                         1),
        op("ALLOC",    {16},                         2),
        op("ALLOC",    {16},                         3),
        op("COPY_H2D", {0, 0, 16},                   4),
        op("COPY_H2D", {1, 1, 16},                   5),
        op("LAUNCH",   {8, 3, 0, 1, 2, 3, 4, 2, 0},  6),   // op_id=8, n_buf=3 (a,b,c), count=4, kind=2 MUL, bcast_period=0
        op("COPY_D2H", {0, 2, 16},                   7),
        op("FREE",     {0},                          8),
        op("FREE",     {1},                          9),
        op("FREE",     {2},                         10),
        op("EOF",      {},                          11),
    };
    float got[4] = {0};
    Error e = vm.setInput(0, a, sizeof(a));
    if(e.ok) e = vm.setInput(1, b, sizeof(b));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, 4, why), why);
}

// VECTOR_OP, unary kind. EW_RELU = kind 6, n_buf=2 (a,c), no B buffer at all --
// exercises the has_b=false path, distinct from ACTIVATION's own RELU (op_id 7
// kind 0), which goes through a completely different function (runActivation).
static void test_vector_op_relu_values(){
    const char* name = "vector_op_relu_matches_cpu";
    float in[4]   = {-1.0f, 2.0f, -3.0f, 4.0f};
    float want[4] = { 0.0f, 2.0f,  0.0f, 4.0f};

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 16 in\n.output 0 16 out\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {16},                      1),
        op("ALLOC",    {16},                      2),
        op("COPY_H2D", {0, 0, 16},                3),
        op("LAUNCH",   {8, 2, 0, 1, 3, 4, 6, 0},  4),   // op_id=8, n_buf=2 (a,c), count=4, kind=6 EW_RELU, bcast_period=0
        op("COPY_D2H", {0, 1, 16},                5),
        op("FREE",     {0},                       6),
        op("FREE",     {1},                       7),
        op("EOF",      {},                        8),
    };
    float got[4] = {0};
    Error e = vm.setInput(0, in, sizeof(in));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, 4, why), why);
}

// VECTOR_OP with a real (non-zero) bcast_period: B is a 2-element "row" that
// repeats every 2 elements of a 4-element A, like BIAS_ADD's per-column bias.
// Confirms bcast_period is actually forwarded to emunpu_vector_op, not just
// accepted and ignored -- a test with bcast_period=0 could never catch that.
static void test_vector_op_broadcast_values(){
    const char* name = "vector_op_broadcast_add_matches_cpu";
    const uint32_t count = 4, period = 2;
    float a[4]    = {1, 2, 3, 4};
    float b[2]    = {100, 200};
    float want[4];
    for(uint32_t i = 0; i < count; i++) want[i] = a[i] + b[i % period];

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 16 a\n.input 1 8 b\n.output 0 16 c\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {16},                                1),
        op("ALLOC",    {8},                                 2),
        op("ALLOC",    {16},                                3),
        op("COPY_H2D", {0, 0, 16},                          4),
        op("COPY_H2D", {1, 1, 8},                           5),
        op("LAUNCH",   {8, 3, 0, 1, 2, 3, (int)count, 0, (int)period},  6),   // kind=0 EW_ADD, bcast_period=2
        op("COPY_D2H", {0, 2, 16},                          7),
        op("FREE",     {0},                                 8),
        op("FREE",     {1},                                 9),
        op("FREE",     {2},                                10),
        op("EOF",      {},                                 11),
    };
    float got[4] = {0};
    Error e = vm.setInput(0, a, sizeof(a));
    if(e.ok) e = vm.setInput(1, b, sizeof(b));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, 4, why), why);
}

// Error path: a binary kind (MUL) given only 2 buffers. do_vector_op's own
// cross-check (is_binary vs has_b) must reject this -- dispatch()'s table
// alone cannot, since min_buffers/max_buffers don't know about kind.
static void test_vector_op_binary_kind_missing_b_rejected(){
    const char* name = "vector_op_binary_kind_with_2_buffers_rejected";
    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    DeviceHandle a, c;
    if(!npu.allocate(16, 1, a).ok || !npu.allocate(16, 1, c).ok){
        result(name, false, "could not allocate 16-byte buffers"); return;
    }
    Error e = npu.dispatch({8, 2, a, c, 3, 4, 2, 0}, 1);   // kind=2 MUL (binary), only 2 buffers
    result(name, !e.ok, "VECTOR_OP MUL with 2 buffers (no B) was accepted");
}

// Error path: a unary kind (RELU) given 3 buffers (an unused B). Symmetric to
// the test above -- both directions of the mismatch must be caught.
static void test_vector_op_unary_kind_extra_b_rejected(){
    const char* name = "vector_op_unary_kind_with_3_buffers_rejected";
    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    DeviceHandle a, b, c;
    if(!npu.allocate(16, 1, a).ok || !npu.allocate(16, 1, b).ok || !npu.allocate(16, 1, c).ok){
        result(name, false, "could not allocate 16-byte buffers"); return;
    }
    Error e = npu.dispatch({8, 3, a, b, c, 3, 4, 6, 0}, 1);   // kind=6 EW_RELU (unary), 3 buffers
    result(name, !e.ok, "VECTOR_OP EW_RELU with 3 buffers (extra B) was accepted");
}

// REDUCE_OP (op_id 9), kind=SUM. A is 2 rows x 3 width: row0=[1,2,3], row1=[10,20,30].
// c[r] = sum(A[r]). Non-square, distinct values per the same convention as
// test_transpose_values (catches a row/width mixup a square or repeated-value
// input would hide).
static void test_reduce_sum_values(){
    const char* name = "reduce_sum_matches_cpu";
    const uint32_t rows = 2, width = 3;
    float a[6] = {1, 2, 3, 10, 20, 30};
    float want[2];
    for(uint32_t r = 0; r < rows; r++){
        float s = 0;
        for(uint32_t i = 0; i < width; i++) s += a[r*width+i];
        want[r] = s;
    }

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 24 a\n.output 0 8 c\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {24},                     1),
        op("ALLOC",    {8},                      2),
        op("COPY_H2D", {0, 0, 24},               3),
        op("LAUNCH",   {9, 2, 0, 1, 3, 2, 3, 0}, 4),   // op_id=9, n_buf=2 (a,c), n_scalar=3 (rows=2, width=3, kind=0 SUM)
        op("COPY_D2H", {0, 1, 8},                5),
        op("FREE",     {0},                      6),
        op("FREE",     {1},                      7),
        op("EOF",      {},                       8),
    };
    float got[2] = {0};
    Error e = vm.setInput(0, a, sizeof(a));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, rows, why), why);
}

// REDUCE_OP, kind=MEAN. Same shape as SUM but divides by width -- catches a
// kind that is read/cast wrong in a way SUM (kind=0) could hide, since SUM and
// MEAN differ only in one extra division inside the device.
static void test_reduce_mean_values(){
    const char* name = "reduce_mean_matches_cpu";
    const uint32_t rows = 2, width = 4;
    float a[8] = {2, 4, 6, 8, 1, 1, 1, 1};
    float want[2];
    for(uint32_t r = 0; r < rows; r++){
        float s = 0;
        for(uint32_t i = 0; i < width; i++) s += a[r*width+i];
        want[r] = s / width;
    }

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 32 a\n.output 0 8 c\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {32},                     1),
        op("ALLOC",    {8},                      2),
        op("COPY_H2D", {0, 0, 32},               3),
        op("LAUNCH",   {9, 2, 0, 1, 3, 2, 4, 3}, 4),   // op_id=9, n_buf=2 (a,c), n_scalar=3 (rows=2, width=4, kind=3 MEAN)
        op("COPY_D2H", {0, 1, 8},                5),
        op("FREE",     {0},                      6),
        op("FREE",     {1},                      7),
        op("EOF",      {},                       8),
    };
    float got[2] = {0};
    Error e = vm.setInput(0, a, sizeof(a));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, rows, why), why);
}

// REDUCE_OP, kind=MAX. Different again from SUM/MEAN's pure accumulation --
// exercises the comparison branch of the device's internal switch.
static void test_reduce_max_values(){
    const char* name = "reduce_max_matches_cpu";
    const uint32_t rows = 2, width = 3;
    float a[6] = {3, 1, 2, -5, -1, -9};
    float want[2] = {3.0f, -1.0f};

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 24 a\n.output 0 8 c\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {24},                     1),
        op("ALLOC",    {8},                      2),
        op("COPY_H2D", {0, 0, 24},               3),
        op("LAUNCH",   {9, 2, 0, 1, 3, 2, 3, 1}, 4),   // op_id=9, n_buf=2 (a,c), n_scalar=3 (rows=2, width=3, kind=1 MAX)
        op("COPY_D2H", {0, 1, 8},                5),
        op("FREE",     {0},                      6),
        op("FREE",     {1},                      7),
        op("EOF",      {},                       8),
    };
    float got[2] = {0};
    Error e = vm.setInput(0, a, sizeof(a));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, rows, why), why);
}

// REDUCE_OP, kind=ARGMAX. Output is the column INDEX of the row's max, not the
// max value itself (core.c: "ARG* reductions emit the position of the
// extremum, not its value") -- stored through the same fp32 buffer as every
// other kind (device.h/PROGRESS.md's known dtype gap), but exactly
// representable as a float for widths this small, so the comparison below is
// still an exact match, not a tolerance one.
static void test_reduce_argmax_values(){
    const char* name = "reduce_argmax_matches_cpu";
    const uint32_t rows = 2, width = 4;
    float a[8] = {1, 5, 3, 2,   // max at index 1
                  9, 2, 8, 20}; // max at index 3
    float want[2] = {1.0f, 3.0f};

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 32 a\n.output 0 8 c\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {32},                     1),
        op("ALLOC",    {8},                      2),
        op("COPY_H2D", {0, 0, 32},               3),
        op("LAUNCH",   {9, 2, 0, 1, 3, 2, 4, 5}, 4),   // op_id=9, n_buf=2 (a,c), n_scalar=3 (rows=2, width=4, kind=5 ARGMAX)
        op("COPY_D2H", {0, 1, 8},                5),
        op("FREE",     {0},                      6),
        op("FREE",     {1},                      7),
        op("EOF",      {},                       8),
    };
    float got[2] = {0};
    Error e = vm.setInput(0, a, sizeof(a));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, rows, why), why);
}

// Error path: kind out of range. checkScalar's ENUM case must reject
// kind >= EMUNPU_REDUCE_KIND_COUNT (7) -- same regression-test shape as
// test_activation_kind_out_of_range_rejected, for the newest ENUM scalar.
static void test_reduce_kind_out_of_range_rejected(){
    const char* name = "reduce_kind_7_rejected";
    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    DeviceHandle a, c;
    if(!npu.allocate(16, 1, a).ok || !npu.allocate(16, 1, c).ok){
        result(name, false, "could not allocate 16-byte buffers"); return;
    }
    Error e = npu.dispatch({9, 2, a, c, 3, 2, 2, 7}, 1);   // kind=7, one past ARGMIN=6
    result(name, !e.ok, "REDUCE_OP with kind=7 (out of range) was accepted");
}

// Error path: rows/width exceeding the allocated input buffer. requireElements
// must still catch this, the same guarantee every other kernel gives.
static void test_reduce_count_exceeds_buffer_rejected(){
    const char* name = "reduce_count_exceeds_buffer_rejected";
    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    DeviceHandle a, c;
    if(!npu.allocate(16, 1, a).ok || !npu.allocate(16, 1, c).ok){   // 16 bytes = 4 floats each
        result(name, false, "could not allocate 16-byte buffers"); return;
    }
    Error e = npu.dispatch({9, 2, a, c, 3, 100, 100, 0}, 1);   // rows=100, width=100 on 4-float buffers
    result(name, !e.ok, "REDUCE_OP with rows=width=100 on 16-byte buffers was accepted");
}

static void test_vec_add_values(){
    const char* name = "vec_add_matches_cpu";
    float a[4]    = {1, 2, 3, 4};
    float b[4]    = {10, 20, 30, 40};
    float want[4] = {11, 22, 33, 44};

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 16 a\n.input 1 16 b\n.output 0 16 c\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {16},                   1),
        op("ALLOC",    {16},                   2),
        op("ALLOC",    {16},                   3),
        op("COPY_H2D", {0, 0, 16},             4),
        op("COPY_H2D", {1, 1, 16},             5),
        op("LAUNCH",   {1, 3, 0, 1, 2, 1, 4},  6),
        op("COPY_D2H", {0, 2, 16},             7),
        op("FREE",     {0},                    8),
        op("FREE",     {1},                    9),
        op("FREE",     {2},                   10),
        op("EOF",      {},                    11),
    };
    float got[4] = {0};
    Error e = vm.setInput(0, a, sizeof(a));
    if(e.ok) e = vm.setInput(1, b, sizeof(b));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, 4, why), why);
}

// x is 2x2, bias has one value per column: z[r][c] = x[r][c] + bias[c].
static void test_bias_add_values(){
    const char* name = "bias_add_matches_cpu";
    float x[4]    = {1, 2, 3, 4};
    float bias[2] = {10, 20};
    float want[4] = {11, 22, 13, 24};

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 16 x\n.input 1 8 bias\n.output 0 16 z\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {16},                      1),
        op("ALLOC",    {8},                       2),
        op("ALLOC",    {16},                      3),
        op("COPY_H2D", {0, 0, 16},                4),
        op("COPY_H2D", {1, 1, 8},                 5),
        op("LAUNCH",   {3, 3, 0, 1, 2, 2, 4, 2},  6),
        op("COPY_D2H", {0, 2, 16},                7),
        op("FREE",     {0},                       8),
        op("FREE",     {1},                       9),
        op("FREE",     {2},                      10),
        op("EOF",      {},                       11),
    };
    float got[4] = {0};
    Error e = vm.setInput(0, x, sizeof(x));
    if(e.ok) e = vm.setInput(1, bias, sizeof(bias));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, 4, why), why);
}

// RMS norm, no gamma. A is 2 rows x 2 width: row0=[3,4], row1=[6,8].
// c[r][i] = a[r][i] / sqrt(mean(a[r]^2) + eps). Both rows share the same
// ratio 3:4 and 6:8 = 3:4, so both rows normalize to the same two values.
static void test_rmsnorm_no_gamma_values(){
    const char* name = "rmsnorm_no_gamma_matches_cpu";
    const uint32_t rows = 2, width = 2;
    float a[4] = {3, 4, 6, 8};
    float want[4];
    const double eps = 1e-5;
    for(uint32_t r = 0; r < rows; r++){
        double mean_sq = 0;
        for(uint32_t i = 0; i < width; i++) mean_sq += (double)a[r*width+i] * a[r*width+i];
        mean_sq /= width;
        double rms = std::sqrt(mean_sq + eps);
        for(uint32_t i = 0; i < width; i++) want[r*width+i] = (float)(a[r*width+i] / rms);
    }

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 16 a\n.output 0 16 c\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {16},                   1),
        op("ALLOC",    {16},                   2),
        op("COPY_H2D", {0, 0, 16},             3),
        op("LAUNCH",   {4, 2, 0, 1, 2, 2, 2},  4),   // op_id=4, n_buf=2 (a,c), n_scalar=2 (rows,width)
        op("COPY_D2H", {0, 1, 16},             5),
        op("FREE",     {0},                    6),
        op("FREE",     {1},                    7),
        op("EOF",      {},                     8),
    };
    float got[4] = {0};
    Error e = vm.setInput(0, a, sizeof(a));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, 4, why), why);
}

// RMS norm, with gamma. Same A as above; gamma = [2, 0.5] scales column 0 by
// 2x and column 1 by 0.5x after normalizing, same broadcast shape as
// BIAS_ADD's bias (one value per column, repeated across every row).
static void test_rmsnorm_with_gamma_values(){
    const char* name = "rmsnorm_with_gamma_matches_cpu";
    const uint32_t rows = 2, width = 2;
    float a[4] = {3, 4, 6, 8};
    float gamma[2] = {2.0f, 0.5f};
    float want[4];
    const double eps = 1e-5;
    for(uint32_t r = 0; r < rows; r++){
        double mean_sq = 0;
        for(uint32_t i = 0; i < width; i++) mean_sq += (double)a[r*width+i] * a[r*width+i];
        mean_sq /= width;
        double rms = std::sqrt(mean_sq + eps);
        for(uint32_t i = 0; i < width; i++) want[r*width+i] = (float)((a[r*width+i] / rms) * gamma[i]);
    }

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 16 a\n.input 1 8 gamma\n.output 0 16 c\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {16},                         1),
        op("ALLOC",    {8},                          2),
        op("ALLOC",    {16},                         3),
        op("COPY_H2D", {0, 0, 16},                   4),
        op("COPY_H2D", {1, 1, 8},                    5),
        op("LAUNCH",   {4, 3, 0, 1, 2, 2, 2, 2},     6),   // op_id=4, n_buf=3 (a,gamma,c), n_scalar=2 (rows,width)
        op("COPY_D2H", {0, 2, 16},                   7),
        op("FREE",     {0},                          8),
        op("FREE",     {1},                          9),
        op("FREE",     {2},                         10),
        op("EOF",      {},                          11),
    };
    float got[4] = {0};
    Error e = vm.setInput(0, a, sizeof(a));
    if(e.ok) e = vm.setInput(1, gamma, sizeof(gamma));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, 4, why), why);
}

// Softmax, plain (log_softmax=0). A is 2 rows x 3 width: row0=[1,2,3],
// row1=[0,0,0]. c[r][i] = exp(a[r][i]-max) / sum(exp(a[r]-max)), computed
// with the max-subtraction stability trick so it matches a numerically
// stable device implementation, not just the textbook formula.
static void test_softmax_values(){
    const char* name = "softmax_matches_cpu";
    const uint32_t rows = 2, width = 3;
    float a[6] = {1, 2, 3, 0, 0, 0};
    float want[6];
    for(uint32_t r = 0; r < rows; r++){
        double max_val = a[r*width+0];
        for(uint32_t i = 1; i < width; i++) max_val = std::max(max_val, (double)a[r*width+i]);
        double sum = 0;
        for(uint32_t i = 0; i < width; i++) sum += std::exp((double)a[r*width+i] - max_val);
        for(uint32_t i = 0; i < width; i++) {
            want[r*width+i] = (float)(std::exp((double)a[r*width+i] - max_val) / sum);
        }
    }

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 24 a\n.output 0 24 c\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {24},                      1),
        op("ALLOC",    {24},                      2),
        op("COPY_H2D", {0, 0, 24},                3),
        op("LAUNCH",   {5, 2, 0, 1, 3, 2, 3, 0},  4),   // op_id=5, n_buf=2 (a,c), n_scalar=3 (rows,width,log_softmax=0)
        op("COPY_D2H", {0, 1, 24},                5),
        op("FREE",     {0},                       6),
        op("FREE",     {1},                       7),
        op("EOF",      {},                        8),
    };
    float got[6] = {0};
    Error e = vm.setInput(0, a, sizeof(a));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, 6, why), why);
}

// Transpose. A is 2 rows x 3 cols, C is 3 rows x 2 cols.
// The shape is deliberately NOT square: a square input would still pass if the
// kernel swapped rows and cols, so it would hide the most likely bug. Every
// value is distinct for the same reason.
//   a = [1 2 3]      c = [1 4]
//       [4 5 6]          [2 5]
//                        [3 6]
static void test_transpose_values(){
    const char* name = "transpose_matches_cpu";
    const uint32_t rows = 2, cols = 3;
    float a[6] = {1, 2, 3, 4, 5, 6};
    float want[6];
    for(uint32_t r = 0; r < rows; r++){
        for(uint32_t i = 0; i < cols; i++){
            want[i*rows + r] = a[r*cols + i];
        }
    }

    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 24 a\n.output 0 24 c\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC",    {24},                   1),
        op("ALLOC",    {24},                   2),
        op("COPY_H2D", {0, 0, 24},             3),
        op("LAUNCH",   {6, 2, 0, 1, 2, 2, 3},  4),   // op_id=6, n_buf=2 (a,c), n_scalar=2 (rows,cols)
        op("COPY_D2H", {0, 1, 24},             5),
        op("FREE",     {0},                    6),
        op("FREE",     {1},                    7),
        op("EOF",      {},                     8),
    };
    float got[6] = {0};
    Error e = vm.setInput(0, a, sizeof(a));
    if(e.ok) e = vm.execute(prog);
    if(e.ok) e = vm.getOutput(0, got, sizeof(got));
    if(!e.ok){ result(name, false, e.msg); return; }

    std::string why;
    result(name, sameFloats(got, want, 6, why), why);
}

// ---------------------------------------------------------------------------
// LAST: a negative slot used to write input_slots[-1] and crash.
// ---------------------------------------------------------------------------
static void test_leak_warn_and_free_lets_run_finish(){
    const char* name = "leak_warn_and_free_policy_lets_run_finish";
    emunpuDevice npu;
    if(!npu.status().ok){ result(name, false, "device not created"); return; }
    VM vm(&npu);
    vm.setLeakPolicy(LeakPolicy::WARN_AND_FREE);
    if(!bindHeader(vm, "").ok){
        result(name, false, "could not bind signature"); return;
    }
    std::vector<opcode> prog = {
        op("ALLOC", {16}, 1),   // never freed
        op("EOF",   {},   2),
    };
    Error e = vm.execute(prog);
    result(name, e.ok, e.msg);
}

static void test_setInput_negative_slot(){
    const char* name = "setInput_rejects_negative_slot";
    emunpuDevice npu;
    VM vm(&npu);
    if(!bindHeader(vm, ".input 0 16 in\n").ok){
        result(name, false, "could not bind signature"); return;
    }
    float x[4] = {0, 0, 0, 0};
    Error e = vm.setInput(-1, x, sizeof(x));
    result(name, !e.ok, "setInput accepted slot -1");
}

int main(){
    std::printf("Unit tests: host API + kernel numerics\n");
    std::printf("--------------------------------------\n");

    test_round_trip();
    test_failed_download_blocks_getOutput();
    test_getOutput_wrong_size();
    test_getOutput_never_written();
    test_setInput_wrong_size();
    test_setInput_null_data();
    test_setInput_negative_size();

    test_matmul_nonsquare();
    test_matmul_huge_dims_rejected();
    test_relu_values();
    test_silu_values();
    test_leaky_relu_values();
    test_clamp_values();
    test_elu_values();
    test_sigmoid_values();
    test_activation_kind_out_of_range_rejected();
    test_activation_count_exceeds_buffer_rejected();
    test_vector_op_mul_values();
    test_vector_op_relu_values();
    test_vector_op_broadcast_values();
    test_vector_op_binary_kind_missing_b_rejected();
    test_vector_op_unary_kind_extra_b_rejected();
    test_reduce_sum_values();
    test_reduce_mean_values();
    test_reduce_max_values();
    test_reduce_argmax_values();
    test_reduce_kind_out_of_range_rejected();
    test_reduce_count_exceeds_buffer_rejected();
    test_vec_add_values();
    test_bias_add_values();
    test_rmsnorm_no_gamma_values();
    test_rmsnorm_with_gamma_values();
    test_softmax_values();
    test_transpose_values();

    test_leak_warn_and_free_lets_run_finish();

    test_setInput_negative_slot();   // keep last

    std::printf("--------------------------------------\n");
    std::printf("TOTAL: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
