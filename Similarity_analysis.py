# Imports
from itertools import combinations
import numpy as np
import seaborn as sns; sns.set_theme()

def Similarity_analysis_plot(concept_cav_vectors, layer_names):
    """
    Main plotting function that creates a similarity heatmap from cav vectors accross layers for concepts
    Args:
        concept_cav_vectors (dict): Input dictionary of cav vectors, where
              key: concept_name, value: dict {layer_name : cav_vector}
        layer_names (list): List of layer names
    """
    # parse and obtain necessary values
    cav_vectors = list(concept_cav_vectors.values())
    concepts = list(concept_cav_vectors.keys())
    number_of_layers = len(cav_vectors[0])
    number_of_concepts =  len(concepts)
    combinations_list = [x for x in range(number_of_concepts)]
    all_combinations_list = list(combinations(combinations_list, 2))

    similarity_list = []
    concept_combinations = []
    for layer_index in range(number_of_layers): # loop through layers
        layer_similarity_list = []
        layer_name = layer_names[layer_index]

        for combination in all_combinations_list: # Loop through all combinations of concepts
            first_concept = concepts[combination[0]]
            second_concept = concepts[combination[1]]
            if len(concept_combinations) != len(all_combinations_list): # If concept label is not in the list, append
                concept_combinations.append(f"{first_concept} and {second_concept}")
            
            # get similarity scores
            concept_1_cav = concept_cav_vectors[first_concept][layer_name].flatten()
            concept_2_cav = concept_cav_vectors[second_concept][layer_name].flatten()
            similarity = np.dot(concept_1_cav,concept_2_cav)
            layer_similarity_list.append(similarity)
        similarity_list.append(layer_similarity_list)
    
    ax = sns.heatmap(similarity_list, xticklabels = concept_combinations, yticklabels=layer_names)
    ax.set_title(f'Similarity scores for cav vectors accross layers')
    fig = ax.get_figure()
    fig.savefig("result_plots\\Similarity_scores.png",bbox_inches='tight', pad_inches=0.5) 

if __name__ == "__main__":
    pass



