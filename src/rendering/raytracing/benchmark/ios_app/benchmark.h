#pragma once
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif
int rl_tools_paper_benchmark_run(const char* scene_path, const char* output_directory, bool smoke_test);
#ifdef __cplusplus
}
#endif
