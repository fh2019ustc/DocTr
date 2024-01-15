🚀 **Exciting update! We have created a demo for our paper on Hugging Face Spaces, showcasing the capabilities of our DocTr. [Check it out here!](https://huggingface.co/spaces/HaoFeng2019/DocTr)**

🔥 **Good news! Our new work [DocTr++: Deep Unrestricted Document Image Rectification](https://github.com/fh2019ustc/DocTr-Plus) comes out, capable of rectifying various distorted document images in the wild.**

🔥 **Good news! Our new work exhibits state-of-the-art performances on the [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) dataset:
[DocScanner: Robust Document Image Rectification with Progressive Learning](https://drive.google.com/file/d/1mmCUj90rHyuO1SmpLt361youh-07Y0sD/view?usp=share_link) with [Repo](https://github.com/fh2019ustc/DocScanner).** 

🔥 **Good news! A comprehensive list of [Awesome Document Image Rectification](https://github.com/fh2019ustc/Awesome-Document-Image-Rectification) methods is available.** 

# DocTr

<p>
    <a href='https://arxiv.org/pdf/2110.12942v2.pdf' target="_blank"><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://huggingface.co/spaces/HaoFeng2019/DocTr' target="_blank"><img src='https://img.shields.io/badge/Online-Demo-green'></a>
</p>

![1](https://user-images.githubusercontent.com/50725551/144743905-2b81e3ab-f2f7-4eee-aa87-f37b740f6998.png)
![2](https://user-images.githubusercontent.com/50725551/144743916-2c0762d0-727f-4d9c-afb2-3161dbaea47a.png)
![3](https://user-images.githubusercontent.com/50725551/144743919-1ff821f1-f2b1-441b-a442-f29e05d08326.png)

> [DocTr: Document Image Transformer for Geometric Unwarping and Illumination Correction](https://arxiv.org/pdf/2110.12942v2.pdf)  
> ACM MM 2021 Oral

Any questions or discussions are welcomed!


## 🚀 Demo [(Link)](https://huggingface.co/spaces/HaoFeng2019/DocTr)
1. Upload the distorted document image to be rectified in the left box.
2. Click the "Submit" button.
3. The rectified image will be displayed in the right box.
4. Our demo environment is based on a CPU infrastructure, and due to image transmission over the network, some display latency may be experienced.

![image](https://user-images.githubusercontent.com/50725551/232953325-2a6782ab-ac49-4f7b-83b7-eae850ccd5dd.png)


## Training
DocTr consists of two main components: a geometric unwarping transformer (GeoTr) and an illumination correction transformer (IllTr).
- For geometric unwarping, we train the GeoTr network using the [Doc3D](https://github.com/fh2019ustc/doc3D-dataset) and [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) dataset.
- For illumination correction, we train the IllTr network based on the [DocProj](https://github.com/xiaoyu258/DocProj) dataset.

## Inference 
1. Download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1eZRxnRVpf5iy3VJakJNTKWw5Zk9g-F_0?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1Cq9bfyAJ9MWwxj0CarqmKw?pwd=jmy1), and put them to `$ROOT/model_pretrained/`.
2. Put the distorted images in `$ROOT/distorted/`.
3. Geometric unwarping. The rectified images are saved in `$ROOT/geo_rec/` by default.
    ```
    python inference.py
    ```
4. Geometric unwarping and illumination rectification. The rectified images are saved in `$ROOT/ill_rec/` by default.
    ```
    python inference.py --ill_rec True
    ```

## Evaluation
- ***Important.*** In the [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html), the '64_1.png' and '64_2.png' distorted images are rotated by 180 degrees, which do not match the GT documents. It is ingored by most of existing works. Before the evaluation, please make a check.
- Note that the performances in our MM paper are computed with the two ***mistaken*** samples in [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html). For reproducing the following quantitative performance on the ***corrected*** [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html), please use the geometric rectified images available from [Google Drive](https://drive.google.com/drive/folders/1kJ34Nk18RVPwYK8mdfcQvU_67whD9tMe?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1Cq9bfyAJ9MWwxj0CarqmKw?pwd=jmy1). For the ***corrected*** performance of [other methods](https://github.com/fh2019ustc/Awesome-Document-Image-Rectification), please refer to our new work [DocScanner](https://drive.google.com/file/d/1mmCUj90rHyuO1SmpLt361youh-07Y0sD/view?usp=share_link).
- ***Image Metrics:***  We use the same evaluation code for MS-SSIM and LD as [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) dataset based on Matlab 2019a. Please compare the scores according to your Matlab version. We provide our Matlab interface file at ```$ROOT/ssim_ld_eval.m```.
- ***OCR Metrics:*** The index of 30 document (60 images) of [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) used for our OCR evaluation is ```$ROOT/ocr_img.txt``` (*Setting 1*). Please refer to [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet) for the index of 25 document (50 images) of [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) used for their OCR evaluation (*Setting 2*). We provide the OCR evaluation code at ```$ROOT/OCR_eval.py```. The version of pytesseract is 0.3.8, and the version of [Tesseract](https://digi.bib.uni-mannheim.de/tesseract/) in Windows is recent 5.0.1.20220118. 
Note that in different operating systems, the calculated performance has slight differences.

|      Method      |    MS-SSIM   |      LD     |     ED (*Setting 1*)    |       CER      |      ED (*Setting 2*)   |      CER     | 
|:----------------:|:------------:|:--------------:| :-------:|:--------------:|:-------:|:--------------:|
|      GeoTr       |     0.5105   |     7.76    |    464.83 |     0.1746     |    724.84 |     0.1832     | 


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{feng2021doctr,
  title={DocTr: Document Image Transformer for Geometric Unwarping and Illumination Correction},
  author={Feng, Hao and Wang, Yuechen and Zhou, Wengang and Deng, Jiajun and Li, Houqiang},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={273--281},
  year={2021}
}
```

```
@article{feng2021docscanner,
  title={DocScanner: Robust Document Image Rectification with Progressive Learning},
  author={Feng, Hao and Zhou, Wengang and Deng, Jiajun and Tian, Qi and Li, Houqiang},
  journal={arXiv preprint arXiv:2110.14968},
  year={2021}
}
```

```
@article{feng2023doctrp,
  title={Deep Unrestricted Document Image Rectification},
  author={Feng, Hao and Liu, Shaokai and Deng, Jiajun and Zhou, Wengang and Li, Houqiang},
  journal={IEEE Transactions on Multimedia},
  year={2023}
}
```


## Acknowledgement
The codes are largely based on [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html), [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet), and [DocProj](https://github.com/xiaoyu258/DocProj). Thanks for their wonderful works.


## Contact
For commercial usage, please contact Professor Wengang Zhou ([zhwg@ustc.edu.cn](zhwg@ustc.edu.cn)) and Hao Feng ([haof@mail.ustc.edu.cn](haof@mail.ustc.edu.cn)).
