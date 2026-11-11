#pragma once
#include <string>
#include <vector>
#include "error.h"
#include <cstdint>
#include <climits>
#include <map>
#include <sstream>

// The only bytecode format version this runtime reads.
static const int SUPPORTED_VERSION = 1;

struct Arg{
    int value;
    bool is_register;
};

struct opcode{
    std::string name;
    std::vector<Arg> args;
    int line_number;
};

struct SlotDecl {
    int slot;
    int bytes;
    std::string name;
    int line_number;     // for "already declared on line N" errors
};



class Bytecode{
public:

    int version = 0;
const std::map<int, SlotDecl>& inputs() const;   // getters
const std::map<int, SlotDecl>& outputs() const;

    // reads the file line by line, parses each line, and stores the result.
    Error loadFromFile(const std::string& filename);
    
    size_t size() const;
    
    const std::vector<opcode>& get_instructions() const;

  
  
private:
    std::vector<opcode> instructions;

    std::map<int, SlotDecl> input_decls;
    std::map<int, SlotDecl> output_decls;

    // Dispatches one header line (".version", ".input" or ".output") to its handler.
    // it writes version, input_decls and output_decls.
    Error parseDirective(const std::string& line, int line_number, bool& seen_version);
    Error parseVersionDirective(std::istringstream& ss, int line_number, bool& seen_version);
    Error parseSlotDirective(std::istringstream& ss, int line_number,
                              const std::string& keyword, std::map<int, SlotDecl>& decls);

    // Parses one bytecode line into one opcode.
    static Error parseLine(const std::string& line, int line_number,opcode& out_opcode);
    static Error stringToInteger(const std::string& str, int& out_value, int line_number);

    static bool isSkipable(const std::string& line);


};
