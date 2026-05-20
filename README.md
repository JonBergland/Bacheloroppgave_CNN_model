### Bachelorthesis gr82 CNN and ViT
This is the repository for the models used to compare to the base model in https://github.com/Vetletb/Bacheloroppgave-gr82-feature-extraction.


#### To run the project
Run this command to install the neccesary libraries
```sh
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

#### Entry Points
- `main_vit.py` - Main entry point for running experiments
- `main_resnet.py` - ResNet model experiments
- `resnet_trainer.py` - ResNet training utilities
- `vision_transformer_trainer.py` - Vision Transformer training utilities

#### Model References
The models in this project are based on the following papers:

1. Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. "Deep Residual Learning for Image Recognition." 2015. arXiv: 1512.03385 [cs.CV]. https://arxiv.org/abs/1512.03385

2. Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit and Neil Houlsby. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." 2021. arXiv: 2010.11929 [cs.CV]. https://arxiv.org/abs/2010.11929
