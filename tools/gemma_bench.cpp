// gemma-bench — run with sudo. Spawns macOS `powermetrics` (GPU sampler, needs
// root) and a real Gemma-2 generation workload concurrently, then prints one
// combined report: prefill/decode tokens-per-second, peak GPU memory, and the
// GPU active-residency / power / frequency averaged over the run.
//
//   sudo ./gemma-bench <model-dir> [seconds]
//
// Pure C++/MLX (no Python). Built with ORAMA_BACKEND_MLX. powermetrics inherits
// root from your sudo; without sudo the GPU samples are skipped (tok/s still
// reported).
#include <unistd.h>

#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <mlx/mlx.h>

#include "model/gemma.hpp"

namespace mx = mlx::core;

namespace {

// Parse the first number appearing after `key` in `line` (-1 if absent).
double num_after(const std::string& line, const char* key) {
    auto p = line.find(key);
    if (p == std::string::npos) return -1.0;
    p += std::strlen(key);
    while (p < line.size() && !(std::isdigit((unsigned char)line[p]) || line[p] == '-' || line[p] == '.')) ++p;
    if (p >= line.size()) return -1.0;
    return std::atof(line.c_str() + p);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: sudo %s <model-dir> [seconds]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const int secs = argc > 2 ? std::atoi(argv[2]) : 6;
    const bool root = (geteuid() == 0);
    if (!root)
        std::fprintf(stderr, "[warn] not root — powermetrics skipped. Re-run with sudo for GPU stats.\n");

    std::fprintf(stderr, "loading %s ...\n", dir.c_str());
    auto g = model::Gemma::load(dir);
    if (!g) { std::fprintf(stderr, "load failed: %s\n", g.error().message.c_str()); return 1; }
    g->generate("hi", 8);  // warm up (Metal kernel compile, etc.)

    // ---- spawn powermetrics in a reader thread (root only) ----
    const int interval_ms = 500;
    const int n_samples = std::max(2, secs * 1000 / interval_ms);
    std::string pm;
    std::thread pmt;
    if (root) {
        pmt = std::thread([&] {
            const std::string cmd = "/usr/bin/powermetrics --samplers gpu_power -i " +
                std::to_string(interval_ms) + " -n " + std::to_string(n_samples) + " 2>/dev/null";
            if (FILE* f = popen(cmd.c_str(), "r")) {
                char buf[8192];
                while (std::fgets(buf, sizeof buf, f)) pm += buf;
                pclose(f);
            }
        });
    }

    // ---- generation workload for ~secs (overlaps the sampling window) ----
    long long ptok = 0, gtok = 0;
    double psum = 0.0, dsum = 0.0;
    int iters = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
        model::GenStats st;
        auto r = g->generate("Write a detailed essay about databases and search engines.", 200, {}, &st);
        if (!r) { std::fprintf(stderr, "generate failed: %s\n", r.error().message.c_str()); break; }
        ptok += st.prompt_tokens; gtok += st.gen_tokens;
        psum += st.prefill_s; dsum += st.decode_s; ++iters;
    }
    if (pmt.joinable()) pmt.join();

    // ---- parse powermetrics GPU samples ----
    double res = 0, pow = 0, frq = 0;
    int rc = 0, pc = 0, fc = 0;
    std::string line;
    auto flush = [&](const std::string& l) {
        if (l.find("active residency") != std::string::npos) {  // utilization %
            double v = num_after(l, "active residency"); if (v >= 0) { res += v; ++rc; }
        } else if (l.find("GPU Power") != std::string::npos) {
            double v = num_after(l, "GPU Power"); if (v >= 0) { pow += v; ++pc; }
        } else if (l.find("active frequency") != std::string::npos) {
            double v = num_after(l, "active frequency"); if (v >= 0) { frq += v; ++fc; }
        }
    };
    for (char ch : pm) { if (ch == '\n') { flush(line); line.clear(); } else line += ch; }
    flush(line);

    // ---- report ----
    const double pmem = static_cast<double>(mx::get_peak_memory()) / 1e6;
    std::printf("\n================ gemma-bench (pure C++/MLX, no Python) ================\n");
    std::printf("model dir        : %s\n", dir.c_str());
    std::printf("window           : %d s   (%d generation runs)\n", secs, iters);
    std::printf("prefill          : %lld tok in %.3f s  =  %.1f tok/s\n", ptok, psum, psum > 0 ? ptok / psum : 0);
    std::printf("decode           : %lld tok in %.3f s  =  %.1f tok/s\n", gtok, dsum, dsum > 0 ? gtok / dsum : 0);
    std::printf("peak GPU memory  : %.0f MB\n", pmem);
    if (rc) std::printf("GPU active resid : %.1f %%   (avg over %d samples)\n", res / rc, rc);
    if (pc) std::printf("GPU power        : %.0f mW  (avg)\n", pow / pc);
    if (fc) std::printf("GPU frequency    : %.0f MHz (avg)\n", frq / fc);
    if (!root) std::printf("GPU samples      : (skipped — run with sudo)\n");
    std::printf("=======================================================================\n");
    return 0;
}
