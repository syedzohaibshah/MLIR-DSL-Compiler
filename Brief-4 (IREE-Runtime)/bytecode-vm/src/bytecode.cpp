#include "bytecode.h"
#include "error.h"
#include <fstream>
#include <sstream>
#include <iostream>



opcode Bytecode::parseLine(const std::string& line, int line_number){
    std::stringstream ss(line);

    std::string name;
    ss>>name;

    std::vector<Arg> args;
    std::string arg;
    while(ss>>arg){
        if(arg[0]=='@'){
             args.push_back({std::stoi(arg.substr(1)),true}); //ignore @ consider the value as register  after that
        }else{
             args.push_back({std::stoi(arg),false});
        }
       
    }

    opcode op;
    op.name = name;
    op.args = args;
    op.line_number = line_number;

    return op;
}

bool Bytecode::loadFromFile(const std::string& filename){
    std::ifstream file(filename);
    if(!file.is_open()){
        std::cerr<<"Failed to open the file\n";
        return false;
    }

    instructions.clear();
    int line_number = 0;
    std::string line;
    while(std::getline(file, line)){
        line_number++;
        instructions.push_back(parseLine(line, line_number));
    }
    return true;
}


size_t Bytecode::size() const{
    return instructions.size();
}

const std::vector<opcode>& Bytecode::get_instructions() const{
    return instructions;
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

bool Bytecode::checkSize(const std::string& label, int slot, int expected, int actual_size) const{
    if(expected == -1){
        std::cerr<<"check"<<label<<"Size: no instruction references "<<label<<" slot "<<slot<<std::endl;
        return false;
    }
    if(expected != actual_size){  //here it will check for conflict in bytecode size 
        std::cerr<<"check"<<label<<"Size: "<<label<<" slot "<<slot<<" expects "<<expected
                  <<" bytes per the bytecode, but "<<actual_size<<" bytes were provided"<<std::endl;
        return false;
    }
    return true;
}

bool Bytecode::checkInputSize(int slot, int actual_size) const{
    return checkSize("Input", slot, expectedInputSize(slot), actual_size);
}

bool Bytecode::checkOutputSize(int slot, int actual_size) const{
    return checkSize("Output", slot, expectedOutputSize(slot), actual_size);
}
