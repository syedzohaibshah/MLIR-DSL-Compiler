#pragma once
#include <map>
#include <string>
#include <vector>
#include "bytecode.h"
#include "error.h"
#include "runtime_limits.h"
#include "hal/device.h"

// What an operand means. See doc/validation-design.md section 5.
//   VALUE:  data (a size, a buffer id, a slot, a number). Literal or @N.
//   REG_ID: names a register. Literal only.
//   TARGET: an instruction number. Literal only.
enum class OperandKind { VALUE, REG_ID, TARGET };

// Which declared-slot table a COPY opcode's slot argument must be in.
// NONE means "this opcode has no slot argument to check".
enum class SlotCheck { NONE, INPUT, OUTPUT };

// One row of the opcode table.
// `kinds` lists the operands in order. For a variadic opcode (LAUNCH), the
// last kind repeats for every argument after the fixed ones.
//
// `slot_check`, `slot_arg` and `size_arg` describe a COPY-shaped opcode: the
// argument at `slot_arg` must name a slot declared in ctx.inputs/outputs, and
// the argument at `size_arg` must equal that slot's declared byte count.
// Leave slot_check as NONE for every other opcode.
struct OpcodeSpec {
    std::string name;
    std::vector<OperandKind> kinds;
    bool variadic;
    int min_args;
    SlotCheck slot_check = SlotCheck::NONE;
    int slot_arg = -1;
    int size_arg = -1;
};

// Everything the verifier reads. VM::execute builds it.
struct VerifyContext {
    const RuntimeLimits& runtime_limits;
    const DeviceLimits& device;
    const std::map<int, SlotDecl>& inputs;
    const std::map<int, SlotDecl>& outputs;
};

// Checks the whole program once, before any device call. Returns the first error.
Error verifyProgram(const std::vector<opcode>& program, const VerifyContext& ctx);
