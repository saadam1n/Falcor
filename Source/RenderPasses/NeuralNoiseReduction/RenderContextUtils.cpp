#include "RenderContextUtils.h"

void blitTextures(RenderContext* pRenderContext, ref<Texture> src, ref<Texture> dst)
{
    pRenderContext->blit(src->getSRV(), dst->getRTV());
}
