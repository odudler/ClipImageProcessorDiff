# ClipImageProcessorDiff
Differentiable implementation of the ClipImageProcessor

Config used: 
https://huggingface.co/Intel/llava-gemma-2b/blob/main/preprocessor_config.json

The Implementation is imperfect. The output tensor differs from the original ClipImageProcessor by about 0.01 per value after normalization.
