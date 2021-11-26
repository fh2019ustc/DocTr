**Good news! Our new work exhibits state-of-the-art performances on DocUNet benchmark dataset: [DocScanner](https://arxiv.org/pdf/2110.14968.pdf)**

# DocTr

![image](https://user-images.githubusercontent.com/50725551/136645513-da99ddb1-4fa4-49a8-8891-6c546b7f782c.png)

> [DocTr: Document Image Transformer for Geometric Unwarping and Illumination Correction](https://arxiv.org/pdf/2110.12942.pdf)  
> ACM MM 2021 Oral


## Training
For geometric unwarping, we train the GeoTr using the [Doc3d](https://github.com/fh2019ustc/doc3D-dataset) dataset.
For illumination correction, we train the IllTr for illumination correction the [DRIC](https://github.com/xiaoyu258/DocProj) dataset.


## Demo 
1. Download the pretrained models [here](https://drive.google.com/drive/folders/1eZRxnRVpf5iy3VJakJNTKWw5Zk9g-F_0?usp=sharing) and put them to `$ROOT/model_pretrained/`.
2. Test:
    ```
    python inference.py
    ```
    
### Citation

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
