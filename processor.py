class _ClipImageProcessor:
    """
        Imperfect differentiable reconstruction of ClipImageProcessor
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/image_processing_clip.py

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to 224):
            Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """
    def __init__(
        self,
        do_resize = True,
        size = 336, #Simplified to Int in this case, for the shortest_edge attribute
        resample = 'bicubic',
        do_center_crop = True,
        crop_size = None,
        do_rescale = True,
        rescale_factor = 1 / 255,
        do_normalize = True,
        image_mean = [0.48145466, 0.4578275, 0.40821073],
        image_std = [0.26862954, 0.26130258, 0.27577711],
        do_convert_rgb = True
    ):
        self.size = size
        self.std = image_std
        self.mean = image_mean
        self.rescale_factor = rescale_factor
        self.do_resize = do_resize
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb

        self.device = DEVICE
        

    def __call__(self, img):
        if self.do_resize:
            #Resize to smaller dimension equalling required dimension, and aspect ratio being conserved
            _, height, width = img.size()
            short, long = (width, height) if width <= height else (height, width)
            new_short, new_long = self.size, int(self.size * long / short)

            if width <= height:
                height, width = new_long, new_short
            else:
                height, width = new_short, new_long
            
            img = img.unsqueeze(0)
            img = F.interpolate(img, size=(height, width), mode=self.resample, antialias=True, align_corners=False)
            img = img.squeeze(0)

        #do_center_crop
        if self.do_center_crop:
            if width < height:
                new_size = width
                left = 0
                top = (height - new_size) // 2
            else:
                new_size = height
                left = (width - new_size) // 2
                top = 0

            right = left + new_size
            bottom = top + new_size

            img = img[:, top:bottom, left:right]

        img = img.clamp(min=0, max=255)

        #rescale
        if self.do_rescale:
            img = img * self.rescale_factor

        #normalize
        if self.do_normalize:
            mean = torch.tensor(self.mean).view(3, 1, 1).to(self.device)
            std = torch.tensor(self.std).view(3, 1, 1).to(self.device)
            img = (img - mean) / std

        return img
