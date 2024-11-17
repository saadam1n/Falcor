#include "TextureReflecter.h"

TextureVertex& TextureVertex::setPredefinedLocation(ref<Texture> location)
{
    predefinedLocation = location;
    return *this;
}

TextureVertex& TextureReflecter::addInput(const std::string& name)
{
    throwIfAlreadyPresent(mInputList, name);

    TextureVertex tv;
    tv.name = name;

    mInputList.push_back(tv);

    return mInputList.back();
}

TextureVertex& TextureReflecter::addOutput(const std::string& name)
{
    throwIfAlreadyPresent(mOutputList, name);

    TextureVertex tv;
    tv.name = name;

    mOutputList.push_back(tv);

    return mOutputList.back();
}

const std::vector<TextureVertex>& TextureReflecter::getInputs()
{
    return mInputList;
}

const std::vector<TextureVertex>& TextureReflecter::getOutputs()
{
    return mOutputList;
}

void TextureReflecter::throwIfAlreadyPresent(const std::vector<TextureVertex>& v, const std::string& s)
{
    for (const auto& tv : v)
    {
        if (s == tv.name)
        {
            FALCOR_THROW("Texture name " + s + " was already added to reflecter!");
        }
    }
}

