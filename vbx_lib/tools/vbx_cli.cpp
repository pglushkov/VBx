#include <cstdio>
#include "vbx/vbx.h"

int main() {
    std::printf("vbx_cli %s\n", vbx::get_version().c_str());
    return 0;
}
