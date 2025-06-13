# cons454_project

CONS 454 Final Project: Using SciKit-Learn to Explore Machine Learning Solutions for Camera Trap Data Classification

Ben McPhee 2025


First task: Implement some baseline classification models on flattened image representations.
Setup: I hand-selected 100 training images and 50 validation images, with either animals, people, or nothing in them, and made these the 3 classes for images to be put into.
Results: Dummy model accuracy - 78%, logistic regression model accuracy - 100% training, 62% validation.

Second task: Use the pre-trained vgg16 model on its own to test out its classification abilities.
Setup: I selected 10 images for the model to identify with its 1000 built-in classes.
Results: I have given the models most likely classification as well as any others of interest.
hippopotamus -> predicted hippo with 74% probability
empty shot of watering hole -> valley 32%, coral reef 6%
same empty shot at night -> hay 14%, volcano 4%
porcupine at night -> badger 24%, porcupine 21%
someone's legs -> maillot 16%
a golf cart -> golf cart 20%
elephant legs with others in the background -> indian elephant 42%, african elephant 30%
two impala slightly out of frame -> impala 40%
giraffe legs -> hyena 16%, zebra 15%
blurry hyena at night -> ram 8.5%, hyena 6%

Third task: Applying transfer learning to densenet (another image model) so I can use it as the base for a new logistic regression model.
Setup: The same training and validation sets from part 1, and then a grid search over LR's hyperparameters for optimization.
Results: Logistic regression model (with C=10 and max_iter=3000) accuracy - 100% training, 86% validation.