import torch
from Model import Dsprite_network
from Model_wrapper import Tcav_wrapper
from Tcav_scorer import tcav_scorer
from Dataloader import DSprite_dataset
from visualisations import make_plots
from Similarity_analysis import Similarity_analysis_plot


def main():
    """
    The main entry point of the Tcav analysis
    Currently a pretrained model with example layer names and concept names are used
    The pretrained model predicts the shape labels for images with a 100% accuracy on test set
    The concept data is gotten from Dsprite and is also the shape latent feature, each concept is a different shape
    Firstly the Dsprite data is loaded, then concepts are created and Dsprite concept data is loaded
    Then for each concept the Cav vectors and Tcav scores are calculated for each layer and regarding each final class label
    Finally a dictionary with tcav scores is printed out and plots are concstructed
    
    """
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    # Get data
    dataset = DSprite_dataset()
    data_size = 100
    latent_samples, x = dataset.get_random_images(data_size)
    label_index = 1
    labels = latent_samples[:, label_index]

    #Make into tensors
    x = torch.tensor(x).float()
    labels = torch.tensor(labels, dtype = torch.int8).reshape(data_size, )

    # Define model, load pretrained model and add wrapper
    model = Dsprite_network(3)
    model.load_state_dict(torch.load("state_dict.pt"))
    layer_names = ["layer1","layer2", "layer3", "layer4", "layer5", "layer6"]
    model = Tcav_wrapper(model, layer_names)
    tcav_score_calculator = tcav_scorer(model)
    tcav_scores = {}
    concepts = ["Concept 1", "Concept 2", "Concept 3"]
    concept_cav_vectors = {}
    for idx, concept in enumerate(concepts):

        # Define concept and non_concept data
        concept_data, non_concept_data = dataset.get_concept_data((1, idx), 50)
        
        # Make into tensors
        concept_data = torch.tensor(concept_data).float().unsqueeze(1)
        non_concept_data = torch.tensor(non_concept_data).float().unsqueeze(1)
        
        tcav_scores[concept] = {}
        cav_vectors = tcav_score_calculator.get_cav_vectors(layer_names, concept_data, non_concept_data)
        concept_cav_vectors[concept] = cav_vectors

        for layer in layer_names: # Loop through layers for tcav scores
            cav_vector = cav_vectors[layer]
            gradients = tcav_score_calculator.get_conceptual_sensitivity(x, labels, cav_vector, layer)
            tcav_scores_layer = tcav_score_calculator.get_tcav_scores(gradients)
            tcav_scores[concept][layer] = tcav_scores_layer # Update the tcav scores current layer of current concept
    
    print(tcav_scores)
    make_plots(tcav_scores)
    Similarity_analysis_plot(concept_cav_vectors, layer_names)



if __name__ == "__main__":
    main()