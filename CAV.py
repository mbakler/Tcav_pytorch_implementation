# Imports
import torch 
from sklearn.linear_model import LogisticRegression
import numpy as np



class CAV_vector(object):
    """
    Class to generate CAV vectors for a concept for a layer
    """

    def __init__(self):
        """
        Init for CAV
        """

    def train_linear_classifier(self, concept_activations, non_concept_activations):
        """
        Train the linear logistic regression classifier to separate concept and non concept activations
        CAV vector is the normal to the hyperplane separating the 2 classes for the classifier
        This is equivalent to doing -1 x slope_coefficients
        Args:
            concept_activations (torch.tensor): Tensor of activations for the network when the concept data was forward passed
            non_concept_activations (torch.tensor): Tensor of activations for the network when the non concept data was forward passed
        """
        #  Get layer activations and labels
        batch_size = concept_activations.shape[0]
        zeros = torch.zeros((batch_size, 1))
        ones = torch.ones((batch_size, 1))
        labels = torch.cat((zeros, ones)).ravel()
        layer_concept_activation = concept_activations.reshape(batch_size, -1) # Flatten activations
        layer_non_concept_activation = non_concept_activations.reshape(batch_size, -1) # Flatten activations
        activations = torch.cat((layer_concept_activation.detach(), layer_non_concept_activation.detach()))

        # Define the model, currently using LogisticRegression, train the model and get the normal
        model = LogisticRegression(max_iter=1000)
        model.fit(activations, labels)
        self.cav = -np.array(model.coef_)
    
    def get_cav(self):
        """
        Calculate and return the unit normal cav
        Returns:
            unit_normal_cav (np.ndarray): Cav corresponding to the latest concept and class layer
        """
        unit_normal_cav = self.cav/np.linalg.norm(self.cav)
        return unit_normal_cav


