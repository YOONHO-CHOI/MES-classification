# MES-classification
Recently, endoscopic remission has been defined as Mayo endoscopic sub-score (MES) 0. 

Therefore, the patients with MES 1 are needed to step up for achieving MES 0 of the endoscopic score. 

In discriminating MES 0 and 1, inter-observer variation is very severe among the endoscopists. 

This study aimed to narrow the gap in distinguishing between MES 0 and MES1 by deep learning model. 

## Materials
### 1) Validation & Internal test data
#### Baseline characteristics
<p align="Center"><img src ="https://user-images.githubusercontent.com/54790722/192462032-1bd001a4-e1ed-4eea-9f5a-b0b229f59153.png"width="80%" height="80%"/></p>

### 2) External test data (Hyperkvasir)


## Model
### 1) Preprocessing
<p align="Center"><img src ="https://user-images.githubusercontent.com/54790722/192457379-7e6ea5b6-8d28-4766-9d27-2ccbeae2d1a1.png" width="80%" height="80%"/></p>

### 2) CNN based image classifier
<p align="Center"><img src ="https://user-images.githubusercontent.com/54790722/192457285-dbf3dd9f-d1f2-4d05-89bc-cf8a5a21a204.png" width="90%" height="90%"/></p>


## Results
### 1) 12-fold cross validation
<p align="Center"><img src ="https://user-images.githubusercontent.com/54790722/192452645-b18d4d7b-79df-44cf-9a2d-5ed5ee16af2f.svg"><img src ="https://user-images.githubusercontent.com/54790722/192452613-feb187c9-9c01-44f1-a052-bf2e528a2334.svg">
<p align="Center"><img src ="https://user-images.githubusercontent.com/54790722/192457510-6b2bea4c-a5ef-4534-9a2f-a2c642e46ac8.jpg"></p>



### 2) Internal test & Performance comparison with 7 novices
<p align="Center"><img src ="https://user-images.githubusercontent.com/54790722/192452788-1169b27c-02b2-459d-b850-fc5a3bdc4e86.svg"/></p>
<p align="Center"><img src ="https://user-images.githubusercontent.com/54790722/192452811-eeed3450-caaa-4d39-99cb-c97031b777fd.svg"><img src ="https://user-images.githubusercontent.com/54790722/192452831-b1c97df6-3d55-4cb7-97fd-29807d082c1f.svg">



### 3) External test with Hyperkvasir
<p align="Center"><img src ="https://user-images.githubusercontent.com/54790722/192452480-08e5565b-f968-48a7-9dee-ed44a39d2684.svg">
<p align="Center"><img src="https://user-images.githubusercontent.com/54790722/192453022-3430b73d-f2c6-446c-8cfd-240eef703a6e.svg"/></p>
<p align="Center"><img src ="https://user-images.githubusercontent.com/54790722/192452458-1e9f2b1a-3b28-4040-b274-0ec41478aff2.svg" width="50%" height="50%"><img src ="https://user-images.githubusercontent.com/54790722/192452442-8ee13cae-3773-45ac-8258-969996aaa8e5.svg" width="50%" height="50%">

