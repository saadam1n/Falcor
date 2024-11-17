#include "TextureData.h"

ref<Texture> TextureData::getTexture(const std::string& name) const
{
    auto iter = mTextureMapping.find(name);

    if (iter == mTextureMapping.end())
    {
        FALCOR_THROW("Texture " + name + " has NOT been added to texture mapping!");
    }

    if (iter->second.ptr == nullptr)
    {
        FALCOR_THROW("Texture " + name + " was tried to be fetched from the Texture Data but was not set!");
    }

    return iter->second.ptr;
}

void TextureData::setTexture(const std::string& name, ref<Texture> tex, bool force, bool dynamicEntry)
{
    TextureInfo info;
    info.ptr = tex;
    info.manuallyOverriden = force;
    info.dynamicEntry = dynamicEntry;

    // attempt to insert into structure
    auto [it, success] = mTextureMapping.insert(std::make_pair(name, info));

    if (!success)
    {
        // not success, it must be added already to the map
        // if the previous texture is null or not manually overriden, we can override it

        // just a quick check to make sure this is not a malformed entry (overriden but null)
        if (!it->second.ptr && it->second.manuallyOverriden)
        {
            FALCOR_THROW("Malformed texture data entry! Name is " + name);
        }

        if (it->second.dynamicEntry != dynamicEntry && it->second.ptr) // throw only if texture is not null
        {
            FALCOR_THROW("Texture has changed dynamic entry states!");
        }
        else if (it->second.manuallyOverriden)
        {
            FALCOR_THROW("Trying to override an already overriden entry in texture data. Dynamic entry state: " + std::to_string(dynamicEntry));
        } 
        else
        {
            // previous doesn't really matter, but warn in case
            if (it->second.ptr)
            {
                std::cout << "Warn: trying to override field " << name
                          << " in texture data which (before override) points to a non-null texture at " << it->second.ptr;
            }

            it->second = info;
        }
        
    }
}
