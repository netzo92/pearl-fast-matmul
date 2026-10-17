// Compile/run without CUDA to compare the actual C++ planning code to fixtures.
#include "plan.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

template<class T> void array(const std::vector<T>& v) {
    std::cout << '[';
    for (size_t i=0;i<v.size();++i) { if (i) std::cout << ','; std::cout << v[i]; }
    std::cout << ']';
}
int main(int argc,char** argv) {
    try {
        if (argc!=2) throw std::runtime_error("usage: plan-check PAIRS_U32_LE");
        std::ifstream input(argv[1],std::ios::binary);
        std::vector<uint32_t> pairs(2*2048);
        for (auto& word:pairs) {
            unsigned char bytes[4];
            if (!input.read(reinterpret_cast<char*>(bytes),4)) throw std::runtime_error("short input");
            word=uint32_t(bytes[0]) | uint32_t(bytes[1])<<8 | uint32_t(bytes[2])<<16 | uint32_t(bytes[3])<<24;
            if (word>=pfm::R) throw std::runtime_error("vertex out of bounds");
        }
        if (input.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("trailing input");
        auto p=pfm::make_plan(pairs,2048);
        std::cout << "{\"offsets\":"; array(p.offsets);
        std::cout << ",\"edges\":"; array(p.edges);
        std::cout << ",\"vertices\":"; array(p.vertices);
        std::cout << ",\"anchors\":"; array(p.anchors);
        std::cout << ",\"dims\":"; array(p.dims); std::cout << "}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
