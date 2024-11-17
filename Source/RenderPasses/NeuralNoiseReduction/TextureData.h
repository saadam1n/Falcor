#pragma once

#include "Falcor.h"

using namespace Falcor;

#include <map>
#include <string>

#define ASSERT_TEXTURE_IS_OUTPUT(textureData, tex, name) \
    if (tex != textureData.getTexture(name))             \
    FALCOR_THROW("Error! " + std::string(name) + " is not in output");

class TextureData
{
public:
    ref<Texture> getTexture(const std::string& name) const;
    // force indicates us to knowingly override the default value
    void setTexture(const std::string& name, ref<Texture> tex, bool force = false, bool dynamicEntry = false);

private:
    struct TextureInfo
    {
        ref<Texture> ptr;
        bool manuallyOverriden = false;
        bool dynamicEntry = false;
    };

    std::map<std::string, TextureInfo> mTextureMapping;
};
