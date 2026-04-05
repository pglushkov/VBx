add_executable(run_benchmarks benchmarks.cpp)
target_link_libraries(run_benchmarks PRIVATE vbx benchmark::benchmark)
