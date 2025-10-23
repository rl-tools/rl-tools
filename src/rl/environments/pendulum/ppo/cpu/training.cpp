#include "config.h"

auto run(TI seed, bool verbose){
    DEVICE device;
    if(verbose){
        rlt::log(device, LOOP_TIMING_CONFIG{});
    }
    LOOP_STATE ts;
    rlt::malloc(device, ts);
    rlt::malloc(device, eval_buffer);
    rlt::init(device, ts, seed);
    while(!rlt::step(device, ts)){
    }
    rlt::rl::utils::evaluation::Result<EVAL_SPEC> result;
    auto actor = rlt::get_actor(ts);
    evaluate(device, ts.envs[0], ts.ui, actor, eval_buffer, result, ts.rng, rlt::Mode<rlt::mode::Evaluation<>>{});
    rlt::log(device, device.logger, "Final return: ", result.returns_mean);
    rlt::log(device, device.logger, "              mean: ", result.returns_mean);
    rlt::log(device, device.logger, "              std : ", result.returns_std);
    rlt::free(device, ts);
    rlt::free(device, eval_buffer);
    return result;
}


int main(int argc, char** argv) {
    bool verbose = true;
    std::vector<decltype(run(0, verbose))> returns;
    for (TI seed=0; seed < 1; seed++){
        auto return_stats = run(seed, verbose);
        returns.push_back(return_stats);
    }
    T sum = 0;
    T sum_squared = 0;
    for(auto& return_stats: returns){
        sum += return_stats.returns_mean;
        sum_squared += return_stats.returns_mean * return_stats.returns_mean;
    }
    T mean = sum / returns.size();
    T std = std::sqrt(sum_squared / returns.size() - mean * mean);
    // median
    std::sort(returns.begin(), returns.end(), [](auto& a, auto& b){
        return a.returns_mean < b.returns_mean;
    });
    T median = returns[returns.size() / 2].returns_mean;
    std::cout << "Mean return: " << mean << std::endl;
    std::cout << "Std return: " << (returns.size() > 1 ? std::to_string(std) : "-")  << std::endl;
    std::cout << "Median return: " << median << std::endl;
#ifdef RL_TOOLS_ENABLE_JSON
    nlohmann::json j;
    for(auto& return_stats: returns){
        j.push_back(return_stats.returns);
    }
    std::ofstream file("pendulum_ppo_returns.json");
    file << j.dump(4);
#endif
    return 0;
}

// Should take ~ 0.3s on M3 Pro in BECHMARK mode
// - tested @ 1118e19f904a26a9619fac7b1680643a0afcb695)
// - tested @ 361c2f5e9b14d52ee497139a3b82867fce0404a7