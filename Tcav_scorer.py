# Imports
from CAV import CAV_vector
import numpy as np
import torch

class tcav_scorer(object):
    """
    Class to generate the TCAV score for a given layer for a concept for a class
    Calculates the TCAVq_C_L_K
    """
    def __init__(self, model_wrapper) -> None:
        """
        Init for the TCAV_scorer
        Args:
            model_wrapper (Tcav_wrapper): Model wrapped with the Tcav_wrapper object
        """
        self.model = model_wrapper
    
    def get_cav_vectors(self, layers,  concept_data, non_concept_data):

        """
        Get the cav vectors for the current concept
        Args:
            layers (list): List of layer names
            concept_data (torch.tensor, torch.nn.utils.dataset): Data for the concept
            non_concept_data (torch.tensor, torch.nn.utils.dataset): Data for the counterexamples of concept
        Returns:
            cav_vectors (dict) : Dictionary (key layer name, value CAV vector) of the unit normal CAV vectors
                values are shape 1 x flattened_activations
        """
        concept_activations, non_concept_activations = self._get_activations(concept_data, non_concept_data)
        cav_vectors = {}
        for layer in layers:
            concept_activations_layer = concept_activations[layer]
            non_concept_activations_layer = non_concept_activations[layer]
            cav_vector = self._calculate_cav(concept_activations_layer, non_concept_activations_layer)
            cav_vectors[layer] = cav_vector
        return cav_vectors

    def _calculate_cav(self, concept_activations, non_concept_activations):
        """
        calculate the cav vector for the current concept and layer
        Args:
            concept_activations (dict): Dictionary of activations for concept data, key is layer name (str), value is tensor of activations
            non_concept_activations (dict): Dictionary of activations for non_concept data, key is layer name (str), value is tensor of activations
        Returns:
            cav_vector (np.ndarray) : The unit normal CAV vector, shape 1 x flattened_activations
        """
        cav = CAV_vector()
        cav.train_linear_classifier(concept_activations, non_concept_activations)
        cav_vector = cav.get_cav()
        return cav_vector

    def _get_activations(self, concept_data, non_concept_data):
        """
        Get the activations for concept and non concept data
        Args:
            concept_data (torch.tensor, torch.nn.utils.dataset): Data for the concept
            non_concept_data (torch.tensor, torch.nn.utils.dataset): Data for the counterexamples of concept
        Returns:
            concept_activations (dict): Dictionary of activations for concept data, key is layer name (str), value is tensor of activations
            non_concept_activations (dict): Dictionary of activations for non_concept data, key is layer name (str), value is tensor of activations
        """
        _ = self.model(concept_data)
        concept_activations = self.model.activations.copy()
        _ = self.model(non_concept_data)
        non_concept_activations = self.model.activations
        return concept_activations, non_concept_activations

    def get_conceptual_sensitivity(self, x, labels, cav_vector, layer_name):
        """
        Calculates the S_C_K_L gradients for a given layer for a concept for each class in the dataset
        Args:
            x (torch.tensor): Dataset for the model
            labels (torch.tensor): Labels for the model
            cav_vector (numpy.ndarray) : The unit normal CAV vector for a concept for current layer, shape 1 x flattened_activations
            layer_name (str): Layer name for which to calculate the conceptual sensitivity
        Returns: 
            gradients (dict): Dictionary of conceptual gradient signs for each class labels, key is class label, value is list of booleans
                True if conceptual sensitivity positive, False if negative
        """
        gradients = {}
        for datapoint, class_label in zip(x, labels):
            output = self.model(datapoint.reshape(1,1,64,64))
            class_label = class_label.item()
            
            if class_label not in gradients.keys(): # Update grad dict if new entry for class
                gradients[class_label] = []

            grad = self.model.generate_gradients(class_label, layer_name) # Get gradients
            directional_derivative = np.dot(grad.flatten(), cav_vector.flatten()) > 0
            gradients[class_label].append(directional_derivative)

        return gradients

    def get_tcav_scores(self, gradients):
        """
        Calculates the Tcav scores given conceptual sensitivities for classes
        Args:
            gradients (dict): Dictionary of conceptual gradient signs for each class labels, key is class label, value is list of booleans
                True if conceptual sensitivity positive, False if negative
        Returns: 
            tcav_scores (dict): Dictionary of tcav scores, key is class label, value is tcav score
        """
        tcav_scores = {}
        for label, derivatives in gradients.items():
            tcav_score = np.sum(derivatives)/len(derivatives)
            tcav_scores[label] = tcav_score
        return tcav_scores