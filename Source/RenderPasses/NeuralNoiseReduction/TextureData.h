#pragma once

#include "Falcor.h"

using namespace Falcor;

#include <map>

class TextureData
{
public:
    ref<Texture> getTexture(const std::string& name) const;
    void setTexture(const std::string& name, ref<Texture> tex, bool force = false);

private:
    std::map<std::string, ref<Texture>> mTextureMapping;
};
