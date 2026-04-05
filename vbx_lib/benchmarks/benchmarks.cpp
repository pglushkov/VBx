#include <benchmark/benchmark.h>

#include "vbx/clustering.h"
#include "vbx/types.h"

#include <vector>
#include <random>
#include <algorithm>
#include <type_traits>

template<typename T>
void fillRandom(std::vector<T>& v, T min = T{0}, T max = T{1}) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<T> dist(min, max);
    std::generate(v.begin(), v.end(), [&]() { return dist(rng); });
}

static void BM_BenchCosineSimilarityDouble(benchmark::State& state) {

    // vbx::MatrixD input(state.range(0), state.range(0)); // or however you construct it
    const int size = state.range(0);
    std::vector<double> input_data(size*size);
    // fill with some data
    fillRandom(input_data);
    auto mat_view = vbx::MutableMatrixViewD{input_data.data(), size, size, 0};

    for (auto _ : state) {
        vbx::cosine_similarity<double>(mat_view);
    }
}

BENCHMARK(BM_BenchCosineSimilarityDouble)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024);

BENCHMARK_MAIN();
