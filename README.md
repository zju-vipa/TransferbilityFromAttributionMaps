# Deep Model Transferbility from Attribution Maps

- [*"Paper: Deep Model Transferbility from Attribution Maps"*](https:), NeurIPS 2019.

  J. Song, Y. Chen, X. Wang, C. Shen, M. Song

## Getting Started

These instructions below will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Install the following:

```
- Python >= 3.6
- Tensorflow >= 1.10.0
- Matlab R2019a
```

Then, install python packages:

```
pip install -r requirements.txt
```

In order to generate attribution maps from Deep Models, you also need to download [DeepExplain](https://github.com/marcoancona/DeepExplain)  and copy it to your project directory.

```
cp -r DeepExplain-master/deepexplain $DIR/lib/
```

### Probe datasets

Those datasets involved in this project are:

- [Taskonomy Tiny](https://github.com/StanfordVL/taskonomy/tree/master/data#downloading-the-dataset)
- [MS COCO Val 2014](http://images.cocodataset.org/zips/test2014.zip)
- [Indoor Scene](http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar)

Make sure to download them and move to **$DIR/dataset**.

Those datasets need to be arranged in the following format:

```
|- dataset
|   |---taskonomy
|   |   |---collierville_rgb
|   |   |   |---point_0_view_0_domain_rgb.png
|   |   |   |---...
|   |   |---corozal_rgb
|   |   |---darden_rgb
|   |   |---markleeville_rgb
|   |   |---wiconisco_rgb
|   |---coco
|   |   |---COCO_val2014_000000000042.jpg
|   |   |---...
|   |---indoor
|   |   |---Images
|   |   |   |---airport_inside
|   |   |   |   |---airport_inside_0001.jpg
|   |   |   |---bowling
|   |   |   |---...
```

You can also check **$DIR/explain_result/name_of_dataset/imlist.txt** to find out how those images are arranged.

### Pretrained Models

Download pretrained models:

```
sh tools/download_model.sh
```

## Running the tests

Generate Attribution Maps and save corresponding Attributions to explain results directory:

```
cd tools
python deep_attribution.py --dataset taskonomy --explain_result_root explain_result 
python deep_attribution.py --dataset coco --explain_result_root explain_result
python deep_attribution.py --dataset indoor --explain_result_root explain_result
```

Calculate affinity matrix of those tasks according to the Attributions:

```
python ahp.py --dataset taskonomy
python ahp.py --dataset coco
python ahp.py --dataset indoor
```

Plot P@K, R@K Curve:

```  
python plot.py --affinity-matrix $SOURCENPY
```

Plot Task Similarity Tree:

```
matlab task_similarity_tree.m --affinity-matrix $SOURCENPY
```

Visualize some attribution maps of Input data:

```
python viz.py 
```

## Citation

If you find this code useful, please cite the following:

```
@inproceedings{,
	title={},
  author={},
  booktitle={},
  year={},
  organization={}
}
```

## Contact

If you have any question, please feel free to contact (Jie Song, sjie@zju.edu.cn) 

