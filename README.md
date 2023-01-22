# Machine-Learning-For-3D-Geometry
This repository contains the my work on the assignments for the Machine Learning for 3D Geometry (ML43D) course at Technical University of Munich (TUM). 
The goal of this course is to explore state-of-the-art algorithms for both supervised and unsupervised machine learning on 3D data, for both analysis and synthesis of 3D shapes and scenes.

## Overview

- Datasets
    - <b>ShapeNet (2015)</b> [[Link]](https://www.shapenet.org/)
<br>3Million+ models and 4K+ categories. A dataset that is large in scale, well organized and richly annotated.<br>

    - <b>ShapeNetCore</b> [[Link]](http://shapenet.cs.stanford.edu/shrec16/):<br> 51300 models for 55 categories.

- Networks
    - <b>Volumetric and Multi-View CNNs for Object Classification on 3D Data (2016)</b> [[Paper]](https://arxiv.org/pdf/1604.03265.pdf) [[Code]](https://github.com/charlesq34/3dcnn.torch)

    - <b>PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation (2017)</b> [[Paper]](http://stanford.edu/~rqi/pointnet/) [[Code]](https://github.com/charlesq34/pointnet)

    - <b>PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space (2017)</b> [[Paper]](https://arxiv.org/pdf/1706.02413.pdf) [[Code]](https://github.com/charlesq34/pointnet2)

    - <b>3D-EPN: Shape Completion using 3D-Encoder-Predictor CNNs and Shape Synthesis</b> [[Paper]](https://arxiv.org/abs/1612.00101.pdf) [[Code]](https://github.com/angeladai/cnncomplete)

    - <b>DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation</b> [[Paper]](https://arxiv.org/abs/1901.05103.pdf) [[Code]](https://github.com/facebookresearch/DeepSDF)


## Prerequisites
- python 3.7
- Pipenv or poetry


## Installing
These instructions will get you a copy of the project up and running on your local machine for development purposes. 

1. Clone the repository by running `git clone https://github.com/MahmouddAhmed/Machine-Learning-For-3D-Geometry.git`
2. Navigate to the project directory by running `cd Machine-Learning-For-3D-Geometry`
3. Install the dependencies by running `pipenv install` or `poetry install`
4. Activate the virtual environment by running `pipenv shell`or run the notebooks using `poetry run jupyter notebook`

## Exercises
The exercises cover the following topics:
1. Basic geometric tasks including hand-crafted shape generation, conversion between representations and simple shape alignment.
2. Machine Learning on 3D shapes with a focus on shape classification and segmentation using simple 3D-CNNs, PointNet and PointNet Segmentation.
3. 3D shape reconstruction using 3D-EPN and DeepSDF.

Each exercise includes a detailed description of the task, the implemented code and the results.

<!-- ## Exercises

### Exercise 1 

In this exercise, we will go over a few basic geometric tasks including hand-crafted shape generation and conversion between representations as well as simple shape alignment. It is fundamental to understand how these work before we dive into various machine learning tasks in the next two exercises. Being able to solve these tasks will also make working on your projects a lot easier later on.

#### Exercise 1.1
Generating and visualization continious Signed distance fields for Sphere, Torus and Atom
#### Exercise 1.2
Converting Signed distance fields to occupancy grids
#### Exercise 1.3
Converting Signed distance fields to Triangle Meshes using the marching cubes algorithm
#### Exercise 1.5
Converting Triangle Meshes to point clouds by sampling points based on the baracentric cordinates
#### Exercise 1.5
Rigid shape allignment using procrustes allignment algorithm


### Exercise 2
Machine Learning on 3D shapes by taking a look at shape classification and segmentation.

#### Exercise 2.1
- Implementing a simple toy dataset and dataloader with artificial data
- Implementing a simple 3D-CNN that classifies if a given SDF is Sphere or Torus
- Training the simple NN using the toy daataset
#### Exercise 2.2
- Implementing the 3DCNN using pytorch
- Training the model on Voxelized ShapeNet data
#### Exercise 2.3
- Implementing the PointNet using pytorch
- Training the model on ShapeNetPointCloud dataset
#### Exercise 2.4
- Implementing the PointNet Segmentation branch using pytorch
- Training the model on ShapeNetParts dataset


### Exercise 3 
We will take a look at two major approaches for 3D shape reconstruction in this last exercise.
#### Exercise 3.1 : Shape Reconstruction from 3D SDF grids with 3D-EPN
- Implementing a  dataset and dataloader for ShapeNet_SDF dataset
- Implementing a  dataset and dataloader for ShapeNet_DF dataset
- Implementing of the 3D-EPN architecture using pytorch
- Training the 3D-EPN for 3d-reconstruction
#### Exercise 3.2 : Shape Reconstruction from 3D SDF grids with 3D-EPN
- Implementing of the DeepSDF Auto-encoder architecture using pytorch
- Training the 3D-EPN for 3d-reconstruction
- Visualize the interpolation between latent codes of two objects -->



## Acknowledgments

I would like to thank the Machine Learning for 3D-Geometry course at the Technical University of Munich and my supervisors for providing the opportunity to work on these assignments.
<br><br>


## References
[1] Park, Jeong Joon, et al. "Deepsdf: Learning continuous signed distance functions for shape representation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019

[2] Mescheder, Lars, et al. "Occupancy networks: Learning 3d reconstruction in function space." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

[3] Lorensen, William E., and Harvey E. Cline. "Marching cubes: A high resolution 3D surface construction algorithm." ACM siggraph computer graphics 21.4 (1987): 163-169.

[4] Schönemann, Peter H. "A generalized solution of the orthogonal procrustes problem." Psychometrika 31.1 (1966): 1-10.

[5] Qi, C. et al. “Volumetric and Multi-view CNNs for Object Classification on 3D Data.” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016): 5648-5656.

[6] Qi, C. et al. “PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017): 77-85.

[7] Dai, Angela, Charles Ruizhongtai Qi, and Matthias Nießner. "Shape completion using 3d-encoder-predictor cnns and shape synthesis." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.

