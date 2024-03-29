## DRGCNN

Deep learning model designed for grading diabetic retinopathy

## Requirements
python = 3.8.18  
torchvision = 0.16.0  
torch = 2.1.0  
timm = 0.9.12  
tensorboard = 2.14.0  
tqdm = 4.66.1
## Organisation of files

**eye_pre_process**： Preprocessing of retinal fundus images.   
**Encoder**： Encoder training module.  
**modules**: Contains model structure, loss function and learning rate reduction strategy.  
**utils**：Contains some common functions and evaluation indicators.  
**BFFN**：Binocular Features Fusion Network training module.  
**CAM**： Category attention module.  
## Training process
### Training Encoder
**1.** Construct dataset 
```
├── EyePACS dataset
    ├── train
        ├── class1
            ├── image1
            ├── ...
        ├── class2
            ├── image2
            ├── ...
        ├── ...
        ├── class5
    ├── valid
    ├── test
```
**2.** Configure the configs training file in the Encoder folder.  
**3.** Add Category Attention Module before backbone's avgpool layer.  
**4.** Run the **main.py** function in the Encoder folder to start training.
### Training Binocular Features Fusion Network (BFFN)
**1.** Construct a paired fundus image dataset (pair the left and right fundus images to generate a pkl file in the following format)
```
pkl_dict={
'train':[(path/1_left.png,path/1_right.png,left_label),(path/1_right.png,path/1_left.png,right_label),...],    
'valid':[...],
'test': [...]}
```
**2.** Fill in the generated pkl file address into the parameter configuration part of **main.py**
```
parser.add_argument('--data-index', type=str, default=r"path/pkl_dict.pkl", help='paired image data index path')
```
**3.** Run the **main.py** function in the BFFN folder to start training.
