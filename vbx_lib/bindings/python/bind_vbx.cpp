#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "vbx/vbx.h"

namespace nb = nanobind;

NB_MODULE(vbx_native, m) {
    m.def("get_version", &vbx::get_version);
}
