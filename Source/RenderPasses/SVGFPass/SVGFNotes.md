# SVGF Notes

# Falcor SVGF Outline
Here's roughly how Falcor does SVGF:
1. `SVGFPackLinearZAndNormal` packs data
2. `SVGFReproject` demodulates and does temporal accumulation of luminance 
3. `SVGFFilterMoments` computes the moments and variance.
4. `SVGFAtrous` does the actual filtering
5. `SVGFFinalModulate` remodulates the albedo

It appears Falcor only uses 1 Atrous filter pass. 

# Things we can learn

## The Atrous filtering
These are filter parameters we can learn, ordered (roughly) by the amount of impact learning can have here:
* The atrous kernel weights
* The kernel weights used in computing variance 
* The power used in the normal, illumination, and depth weighting functions

## Other variables

Here's some stuff we can try outside of filtering:
* The filter step size. This kinda does deviate from Atrous filtering, but it'd be a nice experiment to see how filter sizes affect filtering.
* Temporal comparison thresholds 