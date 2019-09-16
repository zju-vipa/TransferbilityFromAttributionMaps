# Deep Model Transferbility from Attribution Maps

- [*"Paper: Deep Model Transferbility from Attribution Maps"*](https:), NeurIPS 2019.

  Jie Song, Yixin Chen, Xinchao Wang, Chengchao Shen, Mingli Song

# Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

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

In order to generate attribution maps from Deep Models, you also need to download [*"DeepExplain"*](https://github.com/marcoancona/DeepExplain)  and copy it to your project directory.

```
cp -r $DOWNLOADS/DeepExplain-master/deepexplain $DIR/
```

### Probe datasets

Those datasets involved in this project are:

- [*"Taskonomy Tiny"*](https://github.com/StanfordVL/taskonomy/tree/master/data#downloading-the-dataset)
- [*"MS COCO"*](http://images.cocodataset.org/zips/test2014.zip)
- [*"Indoor Scene"*](http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar)

## Running the tests

Generate Attribution Maps and save affinity matrix to results directory:

```
python deep_attribution.py --imlist imlist.txt --save-dir $RESULTS 
```

Plot P@K, R@K Curve:

```  
python plot.py --affinity-matrix $SOURCENPY
```

Plot Task Similarity Tree:

```
matlab save_dendrogram.m --affinity-matrix $SOURCENPY
```

Visualize some attribution maps of Input data:

```
python viz.py 
```

## Citation

If you find this code useful, please cite the following:

```


```





