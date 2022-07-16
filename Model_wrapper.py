# Imports
import torch




class Tcav_wrapper(object):
    """
    A model wrapper for models to be used to get activations and calculate gradients 
    
    """
    def __init__(self, model, layers) -> None:
        """
        Init for the wrapper
        Registers hooks for model layers (model._modules) that are in the layers list
        Args:
            model (Pytorch.nn.Module): a model to be used for the wrapper
            layers (list): list of layer names that the activations and gradients will be calculated for
        """
        self.model = model
        self.activations = {} 

        def get_activation(name): 
            """
            Function to register forward hooks for model layers to get activations for layers during forward pass 
            Args:
                name (str): layer name to get the activations for
            """

            def hook(model, input, output):
                """
                Function to define a forward hook for modules

                """
                self.activations[name] = output

            return hook 

        for name, module in model._modules.items(): # Register hooks
            if name in layers:
                module.register_forward_hook(get_activation(name))
    
    def save_gradient(self, grad):

            """
            Function to save the gradients during backward pass
            """
            
            self.gradients = grad 

    def generate_gradients(self, class_index, layer_name):
            """
            Function to register backward hooks and get gradients for layer activations
            Args:
                class_index (int): index of the class to be analysed 
                layer_name (str): name of the layer that  the gradients will be calculated for
            Returns:
                gradients (np.ndarray): Gradients for a given class
            """
            activation = self.activations[layer_name]
            activation.register_hook(self.save_gradient)
            logit = self.output[:, class_index]
            logit.backward(torch.ones_like(logit), retain_graph=True)
            
            # Gradients of parameters for a given layer activations for a given class
            gradients = self.gradients.cpu().detach().numpy()
            return gradients  


    def eval(self):
            """
            Model eval function for the wrapper
            """
            self.model.eval() 

    def train(self):
            """
            Model train function for the wrapper
            """
            self.model.train()   

    def to(self, device):
            """
            Model to device function for the wrapper
            """
            self.model = self.model.to(device)
            return self   

    def __call__(self, x):
            """
            Model call function for the wrapper
            """
            self.output = self.model(x)
            return self.output    