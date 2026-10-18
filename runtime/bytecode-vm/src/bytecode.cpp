#include "bytecode.h"
#include "error.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include<string>
#include <cstdlib>

Error Bytecode::parseLine(const std::string& line, int line_number,opcode& out_opcode){
    std::stringstream ss(line);

    std::string name;
    ss>>name;

    std::vector<Arg> args;
    std::string arg;
    while(ss>>arg){
        if(arg[0]=='@'){
           char* endptr;
            std::string reg_no= arg.substr(1);
            if(reg_no.empty()){
                return makeError("register argument is empty", line_number);
            }
            long reg_value;
            RETURN_IF_ERROR(stringToInteger(reg_no, reg_value, line_number));
             args.push_back({reg_value,true}); //ignore @ consider the value as register  after that
        }else{
             long reg_value;
             RETURN_IF_ERROR(stringToInteger(arg, reg_value, line_number));
         
    
             args.push_back({reg_value,false});
        }
       
    }

    
    out_opcode.name = name;
    out_opcode.args = args;
    out_opcode.line_number = line_number;

    return ok();
}

Error Bytecode::loadFromFile(const std::string& filename){
    std::ifstream file(filename);
    if(!file.is_open()){
    
        return makeError("Failed to open the file", 0);
    }

    instructions.clear();
    int line_number = 0;
    std::string line;
    while(std::getline(file, line)){
        line_number++;
        if(isSkipable(line)) continue;  // Skip empty lines and comments

        opcode op;
        RETURN_IF_ERROR(parseLine(line, line_number, op));
        instructions.push_back(op);
    }
    return ok();
}


size_t Bytecode::size() const{
    return instructions.size();
}

const std::vector<opcode>& Bytecode::get_instructions() const{
    return instructions;
}

Error Bytecode::stringToInteger(const std::string& str, long& out_value, int line_number){
    char* endptr;
    out_value = std::strtol(str.c_str(), &endptr, 10);
    if (endptr == str.c_str() || *endptr != '\0') {
        return makeError("Invalid integer: " + str, line_number);
    }
    return ok();
}

// Line 1: COPY_H2D 0 0 16
// Line 2: COPY_H2D 0 0 32   <- conflict Same slot, different size
// 



// the slot i am providing along the sizes is refered in the bytecode "COPY_H2D 0 0 16"(copy from input slot 0 to buffer 0)
// now this function  will check if size provided by the opcode matched is consistent throughout the opcode and if the 
// size provided by the opcode match with the  host input size


// bytecode.checkInputSize(0, sizeof(A));
// vm.setInput(0, A, sizeof(A));


int Bytecode::expectedSize(const std::string& opcode_name, int slot_arg_index, int slot) const{
    int found_size = -1;

    for(const auto& op : instructions){

        if( op.name == opcode_name && op.args.size() == 3 ){

            // This slot is a register - we can't know its value statically,
            // so skip this instruction.
            if(op.args[slot_arg_index].is_register){
                continue;
            }

            if(op.args[slot_arg_index].value == slot){
                int size = op.args[2].value;
                if(found_size != -1 && found_size != size){   //checking for the repeated conflicting size,
                    std::cerr<<"Error at line "<<op.line_number<<": slot "<<slot
                              <<" is referenced with conflicting sizes ("<<found_size
                              <<" and "<<size<<")"<<std::endl;
                }
                found_size = size;
            }
        }
    }
    return found_size;
}

int Bytecode::expectedInputSize(int slot) const{
    return expectedSize("COPY_H2D", /*slot_arg_index=*/1, slot);
}

int Bytecode::expectedOutputSize(int slot) const{
    return expectedSize("COPY_D2H", /*slot_arg_index=*/0, slot);
}

Error Bytecode::checkSize(const std::string& label, int slot, int expected, int actual_size) const{
    if(expected == -1){
        return makeError("no instruction references "+label+" slot "+std::to_string(slot), 0);
    }
    if(expected != actual_size){  //here it will check for conflict in bytecode size 
        std::cerr<<"check"<<label<<"Size: "<<label<<" slot "<<slot<<" expects "<<expected
                  <<" bytes per the bytecode, but "<<actual_size<<" bytes were provided"<<std::endl;
        return makeError("size mismatch", 0);
    }
    return ok();
}

Error Bytecode::checkInputSize(int slot, int actual_size) const{
    return checkSize("Input", slot, expectedInputSize(slot), actual_size);
}

Error Bytecode::checkOutputSize(int slot, int actual_size) const{
    return checkSize("Output", slot, expectedOutputSize(slot), actual_size);
}

bool Bytecode::isSkipable(const std::string& line)  {
    std::string trimmed_line = line;

    // Trim leading and trailing whitespace
    size_t start = trimmed_line.find_first_not_of(" \t\n\r\f\v");

    if (start == std::string::npos) {
        return true;  // Ignore empty or whitespace-only lines
    }

    trimmed_line.erase(0, start);
    trimmed_line.erase(
        trimmed_line.find_last_not_of(" \t\n\r\f\v") + 1
    );

    // Ignore comment lines
    if (trimmed_line.substr(0, 2) == "//") {
        return true;
    }

  //consisdering normal instructions, if the line is not empty and not a comment, we will parse it
    return false;
}