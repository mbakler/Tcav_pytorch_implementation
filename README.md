# Unofficial implementation of "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)"


This is a unofficial pytorch implementation of Kim et.al 2018 "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)" (https://arxiv.org/abs/1711.11279) on the DSprites dataset (https://github.com/deepmind/dsprites-dataset). Currently the workflow uses a pretrained model on the Dsprites datasets, where the model predicts the shape class, with concepts being also the shape latent value.

The workflow goes as follows:

1. Loads the data and pretrained model (dataset located in dsprites-dataset folder)
2. Calculates the CAV scores for concepts
3. Calculates the Tcav scores for layers
4. Plots the tcav scores by layer
5. Estimates and plots the similarity of concept activations for layers

Dependencies
------------

Dependencies are outlined in the requirements.txt file and can be installed as follows


    $ pip install -r requirements.txt

Running the workflow
------------

To run the model, the following needs to be run 


    $ python main.py

To specify a different model, the model architecture in model.py needs to be changed, layer names need to be changed in main.py and the pretrained parameters file (model_dict.pth) need to be updated. 
To specify a different dataset and concepts, changes to Dataloader.py need to be made with respective changes in main.py regarding data and concept data loading. 
