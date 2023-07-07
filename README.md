
# TRUST: Towards Racially Unbiased Skin Tone Estimation via Scene Disambiguation (ECCV2022)
<p align="center"> 
<img src="teaser_final_v5_font_change.JPG">
</p>

This is the official Pytorch implementation of TRUST. 

* We identify, analyze and quantify the problem of biased facial albedo estimation.
* We propose [FAIR Challenge](https://trust.is.tue.mpg.de), a new synthetic benchmark including novel evaluation protocol that measures albedo estimation in terms of skin tone and diversity.
* We propose TRUST, a new network that estimates facial albedo with **more accuracy and less bias** in skin tone, hence the reconstructed 3D head avatar can be faithful and inclusive from a single image. 

Please refer to the [arXiv paper](https://arxiv.org/abs/2205.03962) for more details.
## Getting Started
Clone the repo:
  ```bash
  git clone https://github.com/HavenFeng/TRUST/
  cd TRUST
  ```

### Requirements
* Python 3.8 (numpy, skimage, scipy, opencv)  
* PyTorch >= 1.7 (pytorch3d compatible)  
  You can run 
  ```bash
  pip install -r requirements.txt
  ```
  If you encountered errors when installing PyTorch3D, please follow the [official installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) to re-install the library.

### Usage
1. **Prepare data & models**

    Please check our [project website](https://trust.is.tue.mpg.de) to download the FAIR benchmark dataset and our released pretrained models.    
    After downloading the pretrained models, put them in ./data

2. **Run test**  
    a. FAIR benchmark
    ```bash
    python test.py --test_folder '/path/to/trust_models' --test_split val
    ```   
    change the test_split flag to run on test set or validation set.


## Evaluation
TRUST (ours) achieves 57% lower error of the total score (35% lower on Average ITA error, 77% lower on Bias error),  on the [FAIR Challenge](https://trust.is.tue.mpg.de/challenge.html) compared to the previous state-of-the-art method.  

For more details of the evaluation, please check our [arXiv paper](https://arxiv.org/abs/2205.03962). 


## Citation
If you find our work useful to your research, please consider citing:
```
@inproceedings{Feng:TRUST:ECCV2022,
  title = {Towards Racially Unbiased Skin Tone Estimation via Scene Disambiguation}, 
  author = {Feng, Haiwen and Bolkart, Timo and Tesch, Joachim and Black, Michael J. and Abrevaya, Victoria}, 
  booktitle = {European Conference on Computer Vision}, 
  year = {2022}
}
```

## Notes
Training code will also be released in the future. 

## License
This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE](https://github.com/HavenFeng/TRUST/blob/main/LICENSE) file.
By downloading and using the code and model you agree to the terms in the [LICENSE](https://github.com/HavenFeng/TRUST/blob/main/LICENSE). 

## Acknowledgements
For functions or scripts that are based on external sources, we acknowledge the origin individually in each file.  
Here are some great resources we benefit:  
- [DECA](https://github.com/YadiraF/DECA) for the general framework of 3D face reconstruction
- [FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch) and [TF_FLAME](https://github.com/TimoBolkart/TF_FLAME) for the FLAME model  
- [Pytorch3D](https://pytorch3d.org/), [neural_renderer](https://github.com/daniilidis-group/neural_renderer), [SoftRas](https://github.com/ShichenLiu/SoftRas) for rendering  
- [kornia](https://github.com/kornia/kornia) for image/rotation processing  
- [face-alignment](https://github.com/1adrianb/face-alignment) for cropping   
- [FAN](https://github.com/1adrianb/2D-and-3D-face-alignment) for landmark detection
- [face_segmentation](https://github.com/YuvalNirkin/face_segmentation) for skin mask

We would also like to thank other recent public 3D face reconstruction works that allow us to easily perform quantitative and qualitative comparisons :)  
[DECA](https://github.com/soubhiksanyal/RingNet), 
[Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction/blob/master/renderer/rasterize_triangles.py), 
[GANFit](https://github.com/barisgecer/GANFit),
[INORig](https://github.com/zqbai-jeremy/INORig),
[MGCNet](https://github.com/jiaxiangshang/MGCNet)

This work was partly supported by the German Federal Ministry of Education and Research (BMBF): Tuebingen AI Center, FKZ: 01IS18039B
