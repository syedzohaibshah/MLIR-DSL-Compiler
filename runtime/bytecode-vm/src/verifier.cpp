#include "verifier.h"

// One row per opcode in contract section 4. Argument order matches the
// bytecode text format exactly (see bytecode-contract.md section 4).
static const std::vector<OpcodeSpec>& opcodeTable(){
    static const std::vector<OpcodeSpec> table = {
        // name        kinds                                                               variadic  min_args  slot_check          slot_arg  size_arg
        {"ALLOC",      {OperandKind::VALUE},                                                  false,    1},
        {"FREE",       {OperandKind::VALUE},                                                  false,    1},
        {"COPY_H2D",   {OperandKind::VALUE, OperandKind::VALUE, OperandKind::VALUE},          false,    3,  SlotCheck::INPUT,   1,        2},
        {"COPY_D2H",   {OperandKind::VALUE, OperandKind::VALUE, OperandKind::VALUE},          false,    3,  SlotCheck::OUTPUT,  0,        2},
        {"LAUNCH",     {OperandKind::VALUE, OperandKind::VALUE},                                  true,     2},
        {"SET",        {OperandKind::REG_ID, OperandKind::VALUE},                             false,    2},
        {"INC",        {OperandKind::REG_ID},                                                 false,    1},
        {"DEC",        {OperandKind::REG_ID},                                                 false,    1},
        {"BR",         {OperandKind::TARGET},                                                 false,    1},
        {"COND_BR",    {OperandKind::REG_ID, OperandKind::TARGET, OperandKind::TARGET},       false,    3},
        {"EOF",        {},                                                                    false,    0},
    };
    return table;
}

// Returns the table row for `name`, or nullptr if the opcode is unknown.
static const OpcodeSpec* findSpec(const std::string& name){
    for(const auto& spec : opcodeTable()){
        if(spec.name == name){
            return &spec;
        }
    }
    return nullptr;
}

//if the requested index exists in kinds, use that type. Otherwise, reuse the last type.(for variadic(LAUNCH) opcodes)(we have to review it)
static OperandKind kindAt(const OpcodeSpec& spec, size_t i){
    return (i < spec.kinds.size()) ? spec.kinds[i] : spec.kinds.back();
}

// REG_ID and TARGET are literal-only (rule R8). REG_ID must name a real
// register; TARGET must name a real instruction. VALUE may be a register
// (checked at run time, not here) or a literal (no range check applies).
static Error checkOperand(const opcode& op, size_t i, OperandKind kind,
                           int program_size, const VerifyContext& ctx){
    const Arg& arg = op.args[i];
    if(kind == OperandKind::VALUE){
        return ok();
    }
    if(arg.is_register){//check if the id or target is in a  @n register, if so return an error because it should be a literal
        return makeError("opcode " + op.name + " argument " + std::to_string(i) +
                          " must be a literal, but got a register", op.line_number);
    }
    if(kind == OperandKind::REG_ID && (arg.value < 0 || arg.value >= ctx.runtime_limits.MAX_REGISTERS)){
        return makeError("opcode " + op.name + " argument " + std::to_string(i) +
                          " must be in [0, " + std::to_string(ctx.runtime_limits.MAX_REGISTERS) +
                          "), but got " + std::to_string(arg.value), op.line_number);
    }
    if(kind == OperandKind::TARGET && (arg.value < 0 || arg.value >= program_size)){
        return makeError("opcode " + op.name + " argument " + std::to_string(i) +
                          " must be in [0, " + std::to_string(program_size) +
                          "), but got " + std::to_string(arg.value), op.line_number);
    }
    return ok();
}

// ALLOC's only argument, when a literal, must fit the device's single-buffer cap.
static Error checkAllocSize(const opcode& op, const VerifyContext& ctx){
    const Arg& size_arg = op.args[0];
    if(size_arg.is_register){
        return ok(); // only the run can know a register's value
    }
    int64_t size = size_arg.value;
    if(size <= 0 || size > ctx.device.max_allocation_bytes){
        return makeError("opcode ALLOC argument 0 must be in (0, " +
                          std::to_string(ctx.device.max_allocation_bytes) +
                          "], but got " + std::to_string(size), op.line_number);
    }
    return ok();
}

// A COPY opcode's slot argument, when a literal, must be declared, and its
// size argument, when a literal, must equal the slot's declared byte count.
static Error checkSlotArgs(const opcode& op, const OpcodeSpec& spec, const VerifyContext& ctx){
    //get both these from table
    const Arg& slot_arg = op.args[spec.slot_arg];
    const Arg& size_arg = op.args[spec.size_arg];
    if(slot_arg.is_register){
        return ok(); // only the run can know which slot a register names
    }
    const std::map<int, SlotDecl>& decls = (spec.slot_check == SlotCheck::INPUT) ? ctx.inputs : ctx.outputs;
    const char* kind = (spec.slot_check == SlotCheck::INPUT) ? "input" : "output";
    auto it = decls.find(slot_arg.value);
    if(it == decls.end()){
        return makeError("opcode " + op.name + " argument " + std::to_string(spec.slot_arg) +
                          " refers to undeclared " + kind + " slot " + std::to_string(slot_arg.value),
                          op.line_number);
    }

    if(!size_arg.is_register && size_arg.value != it->second.bytes){
        return makeError("opcode " + op.name + " argument " + std::to_string(spec.size_arg) +
                          " size " + std::to_string(size_arg.value) + " does not match declared " +
                          kind + " slot bytes " + std::to_string(it->second.bytes), op.line_number);
    }
    return ok();
}

// Every LAUNCH argument is a value: ids, counts, dimensions and op_id. None may be negative.
static Error checkLaunchLiterals(const opcode& op){
    for(size_t i = 0; i < op.args.size(); ++i){
        const Arg& arg = op.args[i];
        if(!arg.is_register && arg.value < 0){
            return makeError("opcode LAUNCH argument " + std::to_string(i) +
                              " cannot be negative, but got " + std::to_string(arg.value), op.line_number);
        }
    }
    return ok();
}

// Checks one instruction: its name, its argument count, each argument's
// kind, and any opcode-specific rule on its literal VALUE arguments.
static Error verifyInstruction(const opcode& op, int program_size, const VerifyContext& ctx){

    //1.check if opcode even exist in our table
    const OpcodeSpec* spec = findSpec(op.name); 
    if(spec == nullptr){
        return makeError("unknown opcode: " + op.name, op.line_number);
    }

    //2.check arity of opcode arguments.
    size_t expected = spec->kinds.size();
    bool count_ok = spec->variadic ? op.args.size() >= (size_t)spec->min_args //for launch, the number of arguments can be greater than the minimum required
                                    : op.args.size() == expected;
    if(!count_ok){
        return makeError("opcode " + op.name + " expects " +
                          (spec->variadic ? "at least " + std::to_string(spec->min_args)
                                          : std::to_string(expected)) +
                          " arguments, but got " + std::to_string(op.args.size()), op.line_number);
    }
    
    for(size_t i = 0; i < op.args.size(); ++i){
        RETURN_IF_ERROR(checkOperand(op, i, kindAt(*spec, i), program_size, ctx));
    }

    if(op.name == "ALLOC"){
        RETURN_IF_ERROR(checkAllocSize(op, ctx));
    }
    if(spec->slot_check != SlotCheck::NONE){
        RETURN_IF_ERROR(checkSlotArgs(op, *spec, ctx));
    }
    if(op.name == "LAUNCH"){
        RETURN_IF_ERROR(checkLaunchLiterals(op));
    }

    return ok();
}

Error verifyProgram(const std::vector<opcode>& program, const VerifyContext& ctx){
    // Rule R10: the program must end with EOF. An empty program has no last
    // instruction to check, so it fails the same way, at line 0.
    if(program.empty() || program.back().name != "EOF"){
        int line_number = program.empty() ? 0 : program.back().line_number;
        return makeError("program must end with an EOF instruction", line_number);
    }

    for(const auto& op : program){
        RETURN_IF_ERROR(verifyInstruction(op, (int)program.size(), ctx));
    }
    return ok();
}
