#pragma once
#include <string>
#include <iostream>

inline void fail(const std::string& msg, int line_number){
    std::cerr<<"Error at line "<<line_number<<": "<<msg<<std::endl;
}
