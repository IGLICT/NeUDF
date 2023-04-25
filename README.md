# NeUDF
We present an open surface reconstruction method using UDF, called NeUDF, for reconstructing objects with high-fidelity open surfaces from 2D image inputs.

![](./static/teaser.jpg)

## [Project page](http://geometrylearning.com/neudf/) |  [Paper](http://geometrylearning.com/neudf/paper.pdf)
This is the official repo for the implementation of **NeUDF: Leaning Neural Unsigned Distance Fields with Volume Rendering** (CVPR 2023).

## Usage

### Setup environment

Installing the requirements using:
```shell
pip install -r requirements.txt
```

To compile [MeshUDF](https://github.com/cvlab-epfl/MeshUDF) to extract open mesh from the learned UDF field, please run:
```shell
cd custom_mc
python setup.py build_ext --inplace
cd ..
```

To use [PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab) and a customized Screened poisson to extract open mesh from the learned UDF field, please run:
```shell
pip install pymeshlab
```
To build PyMeshLab from source, please refer to [PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab).

### Running

- **Train without mask**

```shell
python exp_runner.py --mode train --conf ./confs/womask_open.conf --case <case_name>
```

- **Train with mask**

```shell
python exp_runner.py --mode train --conf ./confs/wmask_open.conf --case <case_name>
```

- **Extract surface using MeshUDF** 

```shell
python exp_runner.py --mode validate_mesh_udf --conf <config_file> --case <case_name> --is_continue
```

The corresponding mesh can be found in `exp/<case_name>/<exp_name>/meshes/mu<iter_steps>.ply`.

- **Extract surface using Screened Poisson** 

```shell
python exp_runner.py --mode validate_mesh_spsr --conf <config_file> --case <case_name> --is_continue
```

The corresponding mesh can be found in `exp/<case_name>/<exp_name>/meshes/spsr<iter_steps>.ply`.

- **Extract surface using MarchingCubes** 

```shell
python exp_runner.py --mode validate_mesh --conf <config_file> --case <case_name> --is_continue
```

The corresponding mesh can be found in `exp/<case_name>/<exp_name>/meshes/<iter_steps>.ply`.

### Evaluation using pretrained models

- **Without mask setting** 
```shell
python exp_runner.py --mode validate_mesh_udf --conf ./confs/womask_open.conf --case my_rack_test --is_continue --resolution 512
```

- **With mask setting** 
```shell
python exp_runner.py --mode validate_mesh_udf --conf ./confs/wmask_open.conf --case my_whale_test --is_continue --resolution 512
```

### Datasets and results

You can download the full datasets and results [here](https://drive.google.com/drive/folders/1g2x5v6QWUdjQkNoszL2d68I2Gp0VRj5E?usp=sharing) and put them in ./public_data/ and ./exp/, respectively.

The data is organized as follows:

```
public_data
|-- <case_name>
    |-- cameras_xxx.npz
    |-- image
        |-- 000.png
        |-- 001.png
        ...
    |-- mask
        |-- 000.png
        |-- 001.png
        ...
exp
|-- <case_name>
    |-- <conf_name>
        |-- checkpoints
            |-- ckpt_400000.pth
```

### Train NeUDF with custom data

Please refer to the  [Data Conversion](https://github.com/Totoro97/NeuS/tree/main/preprocess_custom_data) in NeuS.


## Citation

Cite as below if you find this repository is helpful to your project:

```
@inproceedings{Liu23NeUDF,
    author = {Liu, Yu-Tao and Wang, Li and Yang, Jie and Chen, Weikai and Meng, Xiaoxu and Yang, Bo and Gao, Lin},
    title = {NeUDF: Leaning Neural Unsigned Distance Fields with Volume Rendering},
    booktitle={Computer Vision and Pattern Recognition (CVPR)},
    year = {2023},
}
```

## Acknowledgement

Some code snippets are borrowed from [NeuS](https://github.com/Totoro97/NeuS), [NDF](https://github.com/jchibane/ndf) and [MeshUDF](https://github.com/cvlab-epfl/MeshUDF). Thanks for these great projects.
