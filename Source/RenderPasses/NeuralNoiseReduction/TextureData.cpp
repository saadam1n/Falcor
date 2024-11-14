#include "TextureData.h"

ref<Texture> TextureData::getTexture(const std::string& name) const
{
    auto iter = mTextureMapping.find(name);

    if (iter == mTextureMapping.end())
    {
        FALCOR_THROW("Texture " + name + " has NOT been added to texture mapping!");
    }

    if (iter->second == nullptr)
    {
        FALCOR_THROW("Texture " + name + " was tried to be fetched from the Texture Data but was not set!");
    }

    return iter->second;
}

void TextureData::setTexture(const std::string& name, ref<Texture> tex, bool force)
{
    auto [it, success] = mTextureMapping.insert(std::make_pair(name, tex));

    if (!success)
    {
        // if texture already exists and is not equal to current tex we are adding
        if (!force && (it->second && it->second != tex))
        {
            FALCOR_THROW("Texture " + name + " has ALREADY been added to texture mapping!");
        }
        else
        {
            // texture is nullptr (uninitialized)
            it->second = tex;
        }
    }
}
