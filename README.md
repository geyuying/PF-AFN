# Parser-Free Virtual Try-on via Distilling Appearance Flows, CVPR 2021
Official code for CVPR 2021 paper 'Parser-Free Virtual Try-on via Distilling Appearance Flows'

![image](https://github.com/geyuying/PF-AFN/blob/main/show/compare.jpg?raw=true)

[[Checkpoints]](https://drive.google.com/file/d/1_a0AiN8Y_d_9TNDhHIcRlERz3zptyYWV/view?usp=sharing)

## Our Test Envirenment
anaconda3

pytorch 1.1.0

torchvision 0.3.0

cuda 9.0

cupy 6.0.0

opencv-python 4.5.1

1 GTX1080 GPU

python 3.6

## Installation
conda create -n tryon python=3.6

source activate tryon     or     conda activate tryon

conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=9.0 -c pytorch

conda install cupy     or     pip install cupy==6.0.0

pip install opencv-python

git clone https://github.com/geyuying/PF-AFN.git

cd PF-AFN

## Run the demo
1. First, you need to download the checkpoints from [google drive](https://drive.google.com/file/d/1_a0AiN8Y_d_9TNDhHIcRlERz3zptyYWV/view?usp=sharing) and put the folder "PFAFN" under the folder "checkpoints". The folder "checkpoints/PFAFN" shold contain "warp_model_final.pth" and "gen_model_final.pth". 
2. The "dataset" folder contains the demo images for test, where the "test_img" folder contains the person images, the "test_clothes" folder contains the clothes images, and the "test_edge" folder contains edges extracted from the clothes images with the built-in function in python (We saved the extracted edges from the clothes images for convenience). 'demo.txt' records the test pairs. During test, a person image, a clothes image and its extracted edege are fed into the network to generate the try-on image.
3. To test with the models, run **test.sh** and the results will be saved in the folder "results".
4. **To reproduce our results from the saved model, your test environment should be the same as our test environment, especifically for the version of cupy.** 

## Dataset
1. [VITON](https://github.com/xthan/VITON) contains a training set of 14,221 image pairs and a test set of 2,032 image pairs, each of which has a front-view woman photo and a top clothing image with the resolution 256 x 192. Our model is trained on the VITON training set and tested on the VITON test set.
2. To test our model on the complete VITON test set, you can download [VITON_test](https://drive.google.com/file/d/1Y7uV0gomwWyxCvvH8TIbY7D9cTAUy6om/view?usp=sharing).
