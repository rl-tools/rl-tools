#include <rl_tools/rl_tools.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv){
    int seed = argc > 1 ? std::atoi(argv[1]) : 0;
    std::vector<int> values(seed, 1);
    int sum = 0;
    for(int i = 0; i < seed; ++i){
        sum += values[i] + i;
    }
    std::cout << "seed " << seed << ", sum " << sum << '\n';
    std::ofstream output("result.txt");
    output << sum << '\n';
    return output ? 0 : 1;
}
