# SIPSA

**SIPSA-Net: Shift-Invariant Pan Sharpening with Moving Object Alignment for Satellite Imagery**
**CVPR2021 Oral Accepted.**

This is the official repository of "SIPSA-Net: Shift-Invariant Pan Sharpening with Moving Object Alignment for Satellite Imagery", CVPR 2021 oral paper.

We provide the training code without the WorldView-3 dataset. If you find this repository useful, please consider citing [our paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_SIPSA-Net_Shift-Invariant_Pan_Sharpening_With_Moving_Object_Alignment_for_Satellite_CVPR_2021_paper.pdf).


# Reference
Jaehyup Lee, Soomin Seo, and Munchurl Kim, "SIPSA-Net: Shift-Invariant Pan Sharpening with Moving Object Alignment for Satellite Imagery", CVPR 2021 oral paper.

Abstract: 
Pan-sharpening is a process of merging a highresolution (HR) panchromatic (PAN) image and its corresponding low-resolution (LR) multi-spectral (MS) image to create an HR-MS and pan-sharpened image. However, due to the different sensors’ locations, characteristics and acquisition time, PAN and MS image pairs often tend to have various amounts of misalignment. Conventional deeplearning-based methods that were trained with such misaligned PAN-MS image pairs suffer from diverse artifacts such as double-edge and blur artifacts in the resultant PANsharpened images. In this paper, we propose a novel framework called shift-invariant pan-sharpening with moving object alignment (SIPSA-Net) which is the first method to take into account such large misalignment of moving object regions for PAN sharpening. The SISPA-Net has a feature alignment module (FAM) that can adjust one feature to be aligned to another feature, even between the two different PAN and MS domains. For better alignment in pansharpened images, a shift-invariant spectral loss is newly
designed, which ignores the inherent misalignment in the original MS input, thereby having the same effect as optimizing the spectral loss with a well-aligned MS image. Extensive experimental results show that our SIPSA-Net can generate pan-sharpened images with remarkable improvements in terms of visual quality and alignment, compared to
the state-of-the-art methods.


Please check the attached **network figure**. 
I attched the wrong network figure on **the arxiv and CVPR reposit** by my mistake. 
![Revised_Camera_ready_network_figure](https://user-images.githubusercontent.com/68861685/123036547-38b37f00-d428-11eb-90bb-85a32bd28365.png)

# Requirements :
numpy, PIL, tensorflow 1.13, time, tifffile, datetime, socket

I have never run this codes with other version of tensorflow. 

The codes are created by Jaehyup Lee, Jaeseok Choi, and Soomin Seo.




# Contact
Please contact us via any of the following emails: woguq365@kaist.ac.kr or leave a note in the issues tab.
