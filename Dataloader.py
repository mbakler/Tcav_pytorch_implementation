# Imports
from torch.utils.data import Dataset
import numpy as np


# Proper dataset for Dspirtes
class DSprite_dataset(Dataset):
    """
    Custom dataset object for the Dsprite dataset holding all of the images with sampling functions
    """
    
    def __init__(self):
        """
        Init function
        imgs: (737280 x 64 x 64, uint8) Images in black and white.
        latents_values: (737280 x 6, float64) Values of the latent factors.
        latents_classes: (737280 x 6, int64) Integer index of the latent factor values. Useful as classification targets.
        metadata: some additional information, including the possible latent values.
        Latents_sizes: (1x6, int64) max size for each latent factor
        """
        # Load dataset
        dataset_zip = np.load("dsprites-dataset\\dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", allow_pickle=True, encoding = "latin1", fix_imports=False)
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.metadata = dataset_zip['metadata'][()]
        self.latents_sizes = self.metadata['latents_sizes']
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))


    def _latent_to_index(self, latents):
        """
        Calculates the image indexes corresponding to current latent values
        Args:
            latents (np.ndarray): array of latent values 
        Return:
            indexes (np.ndarray): Indexes corresponding to images for the input latent values
        """
        indexes = np.dot(latents, self.latents_bases).astype(int)
        return indexes
        
    def _sample_latent(self, size=1):
        """
        Sample a set of random latent factor values
        Args:
            size (int): How many datapoints to sample 
        Return:
            samples (np.ndarray): Sampled datapoints with latent values, shape sizex6
        """
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes): # Assign random latent factor values
            samples[:, lat_i] = np.random.randint(lat_size, size=size)

        return samples
        

    def _sample_latent_concept(self, concept_tuple = (0, 0), size=1):
        """
        Sample a set of datapoints corresponding to a concept and non_concept
        For concept samples, concept latent factor will have a fixed value
        For non_concept samples, concept latent factor will have a random value
        Args:
            concept_tuple (tuple): The concept information where:
                 first element is the latent idx correspondong to the concept latent factor
                 Second element is the latent value for the latent concept factor
            size (int): How many datapoints to sample 
        Return:
            tuple (concept_samples, non_concept_samples): 
                concept_samples (np.ndarray): Sampled latent datapoints with concept latent factor having concept value, shape sizex6
                concept_samples (np.ndarray): Sampled latent datapoints with concept latent factor not having concept value, shape sizex6
        """
        concept_idx, concept_value = concept_tuple
        concept_samples = np.zeros((size, self.latents_sizes.size))
        non_concept_samples = np.zeros((size, self.latents_sizes.size))
        non_concept_values = [x for x in range(self.latents_sizes[concept_idx]) if x != concept_value] # Get allowed values for concept factor

        for lat_i, lat_size in enumerate(self.latents_sizes):
          if concept_idx == lat_i: # fix factor value if concept factor for concept data, random for non_concept
            concept_samples[:, lat_i] = concept_value
            non_concept_samples[:, lat_i] = np.random.choice(non_concept_values, size=size)
          else:
            concept_samples[:, lat_i] = np.random.randint(lat_size, size=size)
            non_concept_samples[:, lat_i] = np.random.randint(lat_size, size=size)

        return concept_samples, non_concept_samples

    def get_concept_data(self, concept_tuple, size):
        """
        Sample concept data from the Dsprite images
        Args:
            concept_tuple (tuple): The concept information where:
                 first element is the latent idx correspondong to the concept latent factor
                 Second element is the latent value for the latent concept factor
            size (int): How many datapoints to sample 
        Return:
            tuple (concept_samples, non_concept_samples): 
                concept_samples (np.ndarray): Sampled images with concept latent factor having concept value, shape sizex6
                concept_samples (np.ndarray): Sampled images with concept latent factor not having concept value, shape sizex6
        """
        concept_samples, non_concept_samples = self._sample_latent_concept(concept_tuple, size)
        indices_sampled_concept = self._latent_to_index(concept_samples)
        indices_sampled_non_concept = self._latent_to_index(non_concept_samples)

        imgs_concept = self.imgs[indices_sampled_concept]
        imgs_non_concept = self.imgs[indices_sampled_non_concept]
        return imgs_concept, imgs_non_concept


    def get_random_images(self, size):
        """
        Sample random data from the Dsprite images
        Args:
            size (int): How many datapoints to sample 
        Return:
            tuple (latent_samples, img_samples): 
                latent_samples (np.ndarray): Sampled latent values with random latent factor combinations
                img_samples (np.ndarray): Sampled images from the latent samples
        """
        latent_samples = self._sample_latent(size)
        indices_sampled = self._latent_to_index(latent_samples)
        img_samples = self.imgs[indices_sampled]
        return latent_samples, img_samples
      
    def __getitem__(self, idx):
        """
        Getter method, where given index a tuple of masked and unmasked images will be returned
        Args:
            idx (int): Index 
        Return:
           image (np.ndarray): image corresponding to the index
        """
        return self.imgs[idx]

    def __len__(self):
        """
        Returns length of dataset
        Return:
            length (int): Length of dataset
        """
        return len(self.imgs)



if __name__ == "__main__":
    pass