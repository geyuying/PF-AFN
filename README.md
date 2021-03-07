# Parser-Free Virtual Try-on via Distilling Appearance Flows, CVPR 2021
Official code for CVPR 2021 paper 'Parser-Free Virtual Try-on via Distilling Appearance Flows'

![image](https://github.com/geyuying/PF-AFN/blob/main/show/compare.jpg?raw=true)

[[Checkpoints]](https://drive.google.com/file/d/1_a0AiN8Y_d_9TNDhHIcRlERz3zptyYWV/view?usp=sharing)
## Installation
conda create -n 1993 python=3.6

source activate 1993     or     conda activate 1993

conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=9.0 -c pytorchhttps://drive.google.com/file/d/1_a0AiN8Y_d_9TNDhHIcRlERz3zptyYWV/view?usp=sharing

conda install cupy     or     pip install cupy==6.0.0

pip install opencv-python

## Our Test Envirenment
anaconda3

pytorch 1.1.0

torchvision 0.3.0

cuda 9.0

cupy 6.0.0

opencv-python 4.5.1

1 GTX1080 GPU

python 3.6

**In oder to test the model with our checkpoints, please make sure that your environment is the same as our test envirenment**.

## Test With the Models
1. First, you need to download the checkpoints from [google drive](https://drive.google.com/file/d/1_a0AiN8Y_d_9TNDhHIcRlERz3zptyYWV/view?usp=sharing) and put the models under the folder "checkpoints/PFAFN". 
2. The "dataset" folder contains the images for test, and 'demo.txt' records the test pairs.
3. To test with the models, run **test.sh** and the results will be saved in the folder "results".
4. To reproduce our results from the saved model, your test environment should be the same as our test environment. 
