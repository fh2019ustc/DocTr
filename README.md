**Good news! Our new work exhibits state-of-the-art performances on [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html) benchmark dataset: 
[DocScanner: Robust Document Image Rectification with Progressive Learning](https://arxiv.org/pdf/2110.14968.pdf)**

# DocTr

![1](https://user-images.githubusercontent.com/50725551/144743905-2b81e3ab-f2f7-4eee-aa87-f37b740f6998.png)
![2](https://user-images.githubusercontent.com/50725551/144743916-2c0762d0-727f-4d9c-afb2-3161dbaea47a.png)
![3](https://user-images.githubusercontent.com/50725551/144743919-1ff821f1-f2b1-441b-a442-f29e05d08326.png)


> [DocTr: Document Image Transformer for Geometric Unwarping and Illumination Correction](https://arxiv.org/pdf/2110.12942.pdf)  
> ACM MM 2021 Oral

Any questions or discussions are welcomed!


## Training
- For geometric unwarping, we train the GeoTr network using the [Doc3d](https://github.com/fh2019ustc/doc3D-dataset) dataset.
- For illumination correction, we train the IllTr network based on the [DRIC](https://github.com/xiaoyu258/DocProj) dataset.

## Inference 
1. Download the pretrained models [here](https://drive.google.com/drive/folders/1eZRxnRVpf5iy3VJakJNTKWw5Zk9g-F_0?usp=sharing) and put them to `$ROOT/model_pretrained/`.
2. Geometric unwarping:
    ```
    python inference.py
    ```
3. Geometric unwarping and illumination rectification:
    ```
    python inference.py --ill_rec True
    ```

## Evaluation
- We use the same evaluation code as [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html) benchmark dataset based on Matlab 2019a. 
- Please compare the scores according to your Matlab version.
- Use the images available [here](https://drive.google.com/drive/folders/1kJ34Nk18RVPwYK8mdfcQvU_67whD9tMe?usp=sharing) for reproducing the quantitative performance reported in the paper and further comparison.


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
