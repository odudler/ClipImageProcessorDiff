# ImageProcessorDiff
Differentiable implementation of the CLIPImageProcessor and SigLIPImageProcessor. Useful for running adversarial attacks on models using these visual backbones.
Using the default parameters, the Imageprocessors work for CLIP ViT-L/14@366px and siglip-so400m-patch14-384, but can easily be adapted for different versions.

The Implementation is imperfect. The output tensor differs from the original ClipImageProcessor by about 0.01 per value after normalization.
