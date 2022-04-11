## A Robust Light-Weight Fused-Feature Encoder-Decoder Model for Monocular Facial Depth Estimation from Single Images Trained on Synthetic Data

## Downloads
- [[Downloads]](https://drive.google.com/file/d/11sf9XwmfXT5IFPli-LMUWpVafh_yiKDm/view?usp=sharing, https://drive.google.com/file/d/1tBNQ0xHu4aWh2XPJcJhENC0tf6-Q9HrT/view?usp=sharing) Trained ckpt files for Synthetic Human Facial Depth Depth and Onnex converted file

## Requirements
Tested on 
```
python==3.7.7
torch==1.6.0
h5py==3.6.0
scipy==1.7.3
opencv-python==4.5.5
mmcv==1.4.3
timm=0.5.4
albumentations=1.1.0
tensorboardX==2.4.1
gdown==4.2.1
```
You can install above package with 
```
$ pip install -r requirements.txt
```

## Inference and Evaluate

#### Dataset
###### Synthetic Facial Depth Test Data
We prepared the dataset for training and testing
contact me on the following email: f.khan4@nuigalway.ie for the complete dataset, I will provide the download link

### Evaluation
  
- Evaluate with model (NYU Depth V2)
  
  Result images will be saved in ./args.result_dir/args.exp_name (default: ./results/test)
   - To evaluate, To save pngs, To save visualized depth maps and Inference with image directory
     ```
     $ python ./code/test.py --dataset nyudepthv2 --data_path ./datasets/ --ckpt_dir ./logs/nyudepthv2/0311_test/epoch_09_model.ckpt --save_eval_pngs  --max_depth 10.0 --max_depth_eval 10.0 --do_evaluate --save_visualize
     ```
  #### Results<br/>
![results_paper_1](https://user-images.githubusercontent.com/49758542/162745266-d40040d5-9453-488f-8ece-5c2b55e7f187.png)

![results_paper_2](https://user-images.githubusercontent.com/49758542/162745410-bb74edb3-0a20-4a4d-be4f-6f05132c04fb.png)

![results_paper_3](https://user-images.githubusercontent.com/49758542/162745426-bf8b499b-8a2f-4875-8525-2a0d5ac5d7e1.png)

## Train

for Facial Depth Data
  ```
  $ python ./code/train.py --dataset nyudepthv2 --data_path ./datasets/ --max_depth 10.0 --max_depth_eval 10.0  
  ```
## Onnex conversion 

  ```
  $ python ./code/convert_onnx.py  
  ```

## Webcam live 

  ```
  $ python ./code/webcam_depth_estimation.py  
  ```
  
## On video  

  ```
  $ python ./code/video_depth_estimation.py  
  ```

## Citation

```
  $   
  ```
## References

[1] From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation. [[code]](https://github.com/cleinc/bts)

[2] Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth. [[code]](https://github.com/vinvino02/GLPDepth)

