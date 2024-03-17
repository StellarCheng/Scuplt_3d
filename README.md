# Sculpt3D: Multi-View Consistent Text-to-3D Generation with Sparse 3D Prior
### [Project Page](https://stellarcheng.github.io/Sculpt3D/) | [Arxiv Paper](https://arxiv.org/abs/2403.09140)



<p style='text-align: justify;'> 
Recent works on text-to-3d generation show that using only 2D diffusion supervision for 3D generation tends to produce results with inconsistent appearances (e.g., faces on the back view) and inaccurate shapes (e.g., animals with extra legs). Existing methods mainly address this issue by retraining diffusion models with images rendered from 3D data to ensure multi-view consistency while struggling to balance 2D generation quality with 3D consistency. In this paper, we present a new framework Sculpt3D that equips the current pipeline with explicit injection of 3D priors from retrieved reference objects without re-training the 2D diffusion model. Specifically, we demonstrate that high-quality and diverse 3D geometry can be guaranteed by keypoints supervision through a sparse ray sampling approach. Moreover, to ensure accurate appearances of different views, we further modulate the output of the 2D diffusion model to the correct patterns of the template views without altering the generated object's style. These two decoupled designs effectively harness 3D information from reference objects to generate 3D objects while preserving the generation quality of the 2D diffusion model. Extensive experiments show our method can largely improve the multi-view consistency while retaining fidelity and diversity.
</p>

## Updates
- 17/3/2024: Code Released.




## Installation

- Install pytorch and torch vision
```sh
# torch1.12.1+cu113
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# or torch2.0.0+cu118
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

- Install dependencies:
```sh
pip install -r requirements.txt
```

Our codes are based on the implementations of [ThreeStudio](https://github.com/threestudio-project/threestudio).
If you have any problem with the installation, you may search the issues in their repos first.
Also feel free to open a new issue here.

## Quickstart


### Run Sculpt3D
```
python run.py 
```

## Credits

Sculpt3D is built on the following open-source projects:
- **[ThreeStudio](https://github.com/threestudio-project/threestudio)** Main Framework
- **[OpenShape](https://github.com/Colin97/OpenShape_code)** Shape Retrieval

Credits from ThreeStudio
- **[Lightning](https://github.com/Lightning-AI/lightning)** Framework for creating highly organized PyTorch code.
- **[OmegaConf](https://github.com/omry/omegaconf)** Flexible Python configuration system.
- **[NerfAcc](https://github.com/KAIR-BAIR/nerfacc)** Plug-and-play NeRF acceleration.

## Citation
```
@article{chen2024sculpt3d,
  title={Sculpt3D: Multi-View Consistent Text-to-3D Generation with Sparse 3D Prior},
  author={Chen, Cheng and Yang, Xiaofeng and Yang, Fan and Feng, Chengzeng and Fu, Zhoujie and Foo, Chuan-Sheng and Lin, Guosheng and Liu, Fayao},
  journal={arXiv preprint arXiv:2403.09140},
  year={2024}
}
```