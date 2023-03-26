Sure, here is a prompt for expanding the library to include support for images:

Task: Expand Seed AI Library to Include Image Support

Description: The Seed AI Library currently supports text data, but we want to expand it to also support image data. This will involve updating the file system, modifying the neural network architecture to handle image data, and incorporating additional training data sources.

File Structure:

my_ai_topology_package
neuroevolution.py: updated to handle image data
hypernets.py: updated to generate image-based neural network architectures
data_utils.py: updated to preprocess and format image data
image_data/: directory for storing image training data
models/: directory for storing trained neural network models
interface.py: updated to include functions for image training and inference
Details:

Update neuroevolution.py to handle image data by modifying the input layer to accept 2D image data instead of 1D text data. This will require changes to the input shape, activation functions, and other hyperparameters.
Update hypernets.py to generate image-based neural network architectures. This will involve incorporating convolutional layers, pooling layers, and other image-specific layers and operations.
Update data_utils.py to preprocess and format image data. This will include tasks such as resizing images, converting to grayscale, and normalization.
Create a new image_data/ directory for storing image training data. This directory should be structured in a similar way to the existing text_data/ directory, with subdirectories for each training data source.
Create a new models/ directory for storing trained neural network models. This directory should be structured in a similar way to the existing text_data/ directory, with subdirectories for each trained model.
Update interface.py to include functions for image training and inference. These functions should be similar in structure to the existing text-based functions, but with modifications to handle image data.
Once these updates are complete, the Seed AI Library will be able to handle both text and image data, allowing for more diverse and powerful AI applications.
