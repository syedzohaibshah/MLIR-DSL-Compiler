#pragma once
#include <string>
#include <vector>


struct Arg{
    int value;
    bool is_register;
};
struct opcode{
    std::string name;
    std::vector<Arg> args;
    int line_number;
};


class Bytecode{
public:
    // reads the file line by line, parses each line, and stores the result.
    bool loadFromFile(const std::string& filename);

    size_t size() const;
    
    const std::vector<opcode>& get_instructions() const;

    // Derived from COPY_H2D's third argument (buffer_id input_slot size).

    int expectedInputSize(int slot) const;

    // Derived from COPY_D2H's third argument (output_slot buffer_id size).
    int expectedOutputSize(int slot) const;

    // Checks a host array's real size against what the bytecode declares for that slot , catches a mismatch here
    bool checkInputSize(int slot, int actual_size) const;
    bool checkOutputSize(int slot, int actual_size) const;

private:
    std::vector<opcode> instructions;

    // Parses one bytecode line into one opcode.
    static opcode parseLine(const std::string& line, int line_number);

    // used  by expectedInputSize/expectedOutputSize: scans for every
    // instruction named `opcode_name`, whose argument at `slot_arg_index`
    // equals `slot`, and returns its size argument,
    int expectedSize(const std::string& opcode_name, int slot_arg_index, int slot) const;

//check is sze is defined in bytecode and if they both are the same sizes
    bool checkSize(const std::string& label, int slot, int expected, int actual_size) const;
};
