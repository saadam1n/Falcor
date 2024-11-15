#pragma once

#include "Falcor.h"

struct GlobalParameters {
    int32_t patchWidth = 1920;
    int32_t patchHeight = 1061; // NOT 1080!
};

extern GlobalParameters sParams;
