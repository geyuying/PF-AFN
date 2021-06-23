# Parser-Free Virtual Try-on via Distilling Appearance Flows, CVPR 2021
Official code for CVPR 2021 paper 'Parser-Free Virtual Try-on via Distilling Appearance Flows'


**The training code has been released.**

![image](https://github.com/geyuying/PF-AFN/blob/main/show/compare_both.jpg?raw=true)

[[Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ge_Parser-Free_Virtual_Try-On_via_Distilling_Appearance_Flows_CVPR_2021_paper.pdf)       [[Supplementary Material]](https://github.com/geyuying/PF-AFN/blob/main/PFAFN_supp.pdf)

[[Checkpoints for Test]](https://drive.google.com/file/d/1_a0AiN8Y_d_9TNDhHIcRlERz3zptyYWV/view?usp=sharing)

[[Training_Data]](https://drive.google.com/file/d/1Uc0DTTkSfCPXDhd4CMx2TQlzlC6bDolK/view?usp=sharing)
[[Test_Data]](https://drive.google.com/file/d/1Y7uV0gomwWyxCvvH8TIbY7D9cTAUy6om/view?usp=sharing)

[[VGG_Model]](https://drive.google.com/file/d/1Mw24L52FfOT9xXm3I1GL8btn7vttsHd9/view?usp=sharing)

## Our Environment
anaconda3

pytorch 1.1.0

torchvision 0.3.0

cuda 9.0

cupy 6.0.0

opencv-python 4.5.1

8 GTX1080 GPU for training; 1 GTX1080 GPU for test

python 3.6

## Installation
conda create -n tryon python=3.6

source activate tryon     or     conda activate tryon

conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=9.0 -c pytorch

conda install cupy     or     pip install cupy==6.0.0

pip install opencv-python

git clone https://github.com/geyuying/PF-AFN.git

cd PF-AFN

## Training on VITON dataset 
1. cd PF-AFN_train
2. Download the VITON training set from [VITON_train](https://drive.google.com/file/d/1Uc0DTTkSfCPXDhd4CMx2TQlzlC6bDolK/view?usp=sharing) and put the folder "VITON_traindata" under the folder "dataset".
3. Dowload the VGG_19 model from [VGG_Model](https://drive.google.com/file/d/1Mw24L52FfOT9xXm3I1GL8btn7vttsHd9/view?usp=sharing) and put "vgg19-dcbb9e9d.pth" under the folder "models".
4. First train the parser-based network PBAFN. Run **scripts/train_PBAFN_stage1.sh**. After the parser-based warping module is trained, run **scripts/train_PBAFN_e2e.sh**.
5. After training the parser-based network PBAFN, train the parser-free network PFAFN. Run **scripts/train_PFAFN_stage1.sh**. After the parser-free warping module is trained, run **scripts/train_PFAFN_e2e.sh**.
6. Following the above insructions with the provided training code, the [[trained PF-AFN]](https://drive.google.com/file/d/1Pz2kA65N4Ih9w6NFYBDmdtVdB-nrrdc3/view?usp=sharing) achieves FID 9.92 on VITON test set with the test_pairs.txt (You can find it in https://github.com/minar09/cp-vton-plus/blob/master/data/test_pairs.txt).

## Run the demo
1. cd PF-AFN_test
2. First, you need to download the checkpoints from [checkpoints](https://drive.google.com/file/d/1_a0AiN8Y_d_9TNDhHIcRlERz3zptyYWV/view?usp=sharing) and put the folder "PFAFN" under the folder "checkpoints". The folder "checkpoints/PFAFN" shold contain "warp_model_final.pth" and "gen_model_final.pth". 
3. The "dataset" folder contains the demo images for test, where the "test_img" folder contains the person images, the "test_clothes" folder contains the clothes images, and the "test_edge" folder contains edges extracted from the clothes images with the built-in function in python (We saved the extracted edges from the clothes images for convenience). 'demo.txt' records the test pairs. 
4. During test, a person image, a clothes image and its extracted edge are fed into the network to generate the try-on image. **No human parsing results or human pose estimation results are needed for test.**
5. To test with the saved model, run **test.sh** and the results will be saved in the folder "results".
6. **To reproduce our results from the saved model, your test environment should be the same as our test environment, especifically for the version of cupy.** 

![image](https://github.com/geyuying/PF-AFN/blob/main/show/compare.jpg?raw=true)
## Dataset
1. [VITON](https://github.com/xthan/VITON) contains a training set of 14,221 image pairs and a test set of 2,032 image pairs, each of which has a front-view woman photo and a top clothing image with the resolution 256 x 192. Our saved model is trained on the VITON training set and tested on the VITON test set.
2. To train from scratch on VITON training set, you can download [VITON_train](https://drive.google.com/file/d/1Uc0DTTkSfCPXDhd4CMx2TQlzlC6bDolK/view?usp=sharing).
3. To test our saved model on the complete VITON test set, you can download [VITON_test](https://drive.google.com/file/d/1Y7uV0gomwWyxCvvH8TIbY7D9cTAUy6om/view?usp=sharing).

## License
The use of this code is RESTRICTED to non-commercial research and educational purposes.

## Acknowledgement
Our code is based on the implementation of "Clothflow: A flow-based model for clothed person generation" (See the citation below), including the implementation of the feature pyramid networks (FPN) and the ResUnetGenerator, and the adaptation of the cascaded structure to predict the appearance flows. If you use our code, please also cite their work as below.


## Citation
If our code is helpful to your work, please cite:
```
@article{ge2021parser,
  title={Parser-Free Virtual Try-on via Distilling Appearance Flows},
  author={Ge, Yuying and Song, Yibing and Zhang, Ruimao and Ge, Chongjian and Liu, Wei and Luo, Ping},
  journal={arXiv preprint arXiv:2103.04559},
  year={2021}
}
```
```
@inproceedings{han2019clothflow,
  title={Clothflow: A flow-based model for clothed person generation},
  author={Han, Xintong and Hu, Xiaojun and Huang, Weilin and Scott, Matthew R},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10471--10480},
  year={2019}
}
```
