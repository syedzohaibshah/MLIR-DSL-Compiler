#include "bytecode.h"
#include "error.h"
#include <fstream>
#include <sstream>
#include<string>
#include <cstdlib>
#include <cerrno>
#include <cctype>

Error Bytecode::parseLine(const std::string& line, int line_number,opcode& out_opcode){
    std::stringstream ss(line);

    std::string name;
    ss>>name;

    std::vector<Arg> args;
    std::string arg;
    while(ss>>arg){
        if(arg[0]=='@'){
         
            std::string reg_no= arg.substr(1);
            if(reg_no.empty()){
                return makeError("register argument is empty", line_number);
            }
            int reg_value;
            RETURN_IF_ERROR(stringToInteger(reg_no, reg_value, line_number));
             args.push_back({reg_value,true}); //ignore @ consider the value as register  after that
        }else{
             int reg_value;
             RETURN_IF_ERROR(stringToInteger(arg, reg_value, line_number));
         
    
             args.push_back({reg_value,false});
        }
       
    }

    
    out_opcode.name = name;
    out_opcode.args = args;
    out_opcode.line_number = line_number;

    return ok();
}

// Fails if any token is left on the directive line.
static Error expectEndOfLine(std::istringstream& ss, const std::string& keyword, int line_number){
    std::string extra;
    if(ss >> extra){
        return makeError("unexpected '" + extra + "' after " + keyword, line_number);
    }
    return ok();
}

// A name uses only letters, digits, '_', '.' and '-'. It is never empty.
static bool isValidName(const std::string& name){
    if(name.empty()) return false;
    for(unsigned char c : name){  //Using unsigned char avoids undefined behavior for characters with negative char values.
        if(!std::isalnum(c) && c != '_' && c != '.' && c != '-') return false; //reason behind it?
    }
    return true;
}



// Handles ".version n". Rejects a repeat, a non-number, trailing tokens,
// and any version this runtime does not read.
Error Bytecode::parseVersionDirective(std::istringstream& ss, int line_number, bool& seen_version){
    if(seen_version){
        return makeError("multiple .version directives", line_number);
    }
    std::string version_token;
    if(!(ss >> version_token)){
        return makeError("invalid .version directive: expected a number", line_number);
    }
    int version_value;
    RETURN_IF_ERROR(stringToInteger(version_token, version_value, line_number));
    if(version_value != SUPPORTED_VERSION){
        return makeError("unsupported bytecode version " + version_token +
                         ", this runtime reads version " + std::to_string(SUPPORTED_VERSION), line_number);
    }
    RETURN_IF_ERROR(expectEndOfLine(ss, ".version", line_number));

    version = version_value;
    seen_version = true;
    return ok();
}

// Handles ".input <slot> <bytes> <name>" and ".output <slot> <bytes> <name>".
// `keyword` is ".input" or ".output"; `decls` is that directive's own map.
Error Bytecode::parseSlotDirective(std::istringstream& ss, int line_number,
                                   const std::string& keyword, std::map<int, SlotDecl>& decls){
    std::string slot_token, bytes_token, name;
    if(!(ss >> slot_token >> bytes_token >> name)){
        return makeError("invalid " + keyword + " directive: expected <slot> <bytes> <name>", line_number);
    }

    int slot, bytes;
    RETURN_IF_ERROR(stringToInteger(slot_token, slot, line_number));
    RETURN_IF_ERROR(stringToInteger(bytes_token, bytes, line_number));

    if(!isValidName(name)){
        return makeError("invalid name '" + name + "' in " + keyword + " directive", line_number);
    }
    RETURN_IF_ERROR(expectEndOfLine(ss, keyword, line_number));

    auto found = decls.find(slot);
    if(found != decls.end()){
        return makeError(keyword.substr(1) + " slot " + std::to_string(slot) +
                         " already declared on line " + std::to_string(found->second.line_number), line_number);
    }
    decls[slot] = SlotDecl{slot, bytes, name, line_number};
    return ok();
}

// Dispatches one header line to its handler. `.version` must come first;
// every other directive needs it already seen.
Error Bytecode::parseDirective(const std::string& line, int line_number, bool& seen_version){
    std::istringstream ss(line);
    std::string keyword;
    ss >> keyword;

    if(keyword == ".version"){
        return parseVersionDirective(ss, line_number, seen_version);
    }
    if(!seen_version){
        return makeError(keyword + " appears before .version, and .version must be the first line", line_number);
    }
    if(keyword == ".input"){
        return parseSlotDirective(ss, line_number, keyword, input_decls);
    }
    if(keyword == ".output"){
        return parseSlotDirective(ss, line_number, keyword, output_decls);
    }
    return makeError("unknown directive " + keyword, line_number);
}


Error Bytecode::loadFromFile(const std::string& filename){
    std::ifstream file(filename);
    if(!file.is_open()){
    
        return makeError("Failed to open the file", 0);
    }

   instructions.clear();
input_decls.clear();
output_decls.clear();
version = 0;

bool in_header = true;  //tells whether we are still reading the header
bool seen_version = false;
int line_number = 0;
std::string line;

   while (std::getline(file, line)) {
    line_number++;
    
    if (line.find('\r') != std::string::npos) { //handle CRLF line endings
    return makeError("carriage return character is not allowed", line_number);
}
    // Remove inline comments
    size_t comment_pos = line.find("//");

    if (comment_pos != std::string::npos) {
        line.erase(comment_pos);
    }

    // Skip empty lines and comment-only lines
    if (isSkipable(line)) continue;

    // A directive starts with '.' after any leading spaces. isSkipable already removed blank lines.
    size_t first = line.find_first_not_of(" \t\r\f\v");
    bool is_directive = (first != std::string::npos && line[first] == '.');

    if (is_directive) {
        if (!in_header) { //check whether we are still in the header
            return makeError("header line after the first instruction", line_number);
        }
        RETURN_IF_ERROR(parseDirective(line, line_number, seen_version));
        continue;
    }

    // First instruction line. The header ends here.
    if (in_header) {
        
        if(!seen_version){
            return makeError("missing .version directive in header", line_number);
        }
        in_header = false;
    }
//how does it handle the header and instruction mix up into eachother?
    opcode op;
     
    RETURN_IF_ERROR(parseLine(line, line_number, op));

    instructions.push_back(op);
}

    // A file with no instructions is a header only, or it is empty. Both are errors.
    if(instructions.empty()){
        return makeError(seen_version ? "program has no instructions" : "missing .version directive", line_number);
    }
    return ok();
}


size_t Bytecode::size() const{
    return instructions.size();
}

const std::vector<opcode>& Bytecode::get_instructions() const{
    return instructions;
}


Error Bytecode::stringToInteger(const std::string& str, int& out_value, int line_number) {
    char* endptr;
    errno = 0;

    long value = std::strtol(str.c_str(), &endptr, 10);

    if (errno == ERANGE) {
        return makeError("Integer out of range: " + str, line_number);
    }
    else if (endptr == str.c_str() || *endptr != '\0') {
        return makeError("Invalid integer: " + str, line_number);
    }
    else if (value < INT_MIN || value > INT_MAX) {
        return makeError("Integer out of int range: " + str, line_number);
    }

    out_value = static_cast<int>(value);
    return ok();
}



const std::map<int, SlotDecl>& Bytecode::outputs() const {
    return output_decls;
}

const std::map<int, SlotDecl>& Bytecode::inputs() const {
    return input_decls;
}

bool Bytecode::isSkipable(const std::string& line)  {
    std::string trimmed_line = line;

    // Trim leading and trailing whitespace
    size_t start = trimmed_line.find_first_not_of(" \t\n\r\f\v");

    if (start == std::string::npos) {
        return true;  // Ignore empty or whitespace-only lines
    }
  trimmed_line.erase(0, start);

//trim the trailing whitespace
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