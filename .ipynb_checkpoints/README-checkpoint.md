# CONS 454 Final Project: Using SciKit-Learn to Explore Machine Learning Solutions for Camera Trap Classification

## Benjamin McPhee 2025
<br><br>
### **Part 1: Experimentation**

**Baseline**  
*Task*: Implement some baseline classification models on flattened image representations.  
*Setup*: I hand-selected 100 training images and 50 validation images, with either animals, people, or nothing in them, and made these the 3 classes for images to be put into.  
*Results*: Dummy model (picking animal every time) accuracy - 78%, Logistic regression model accuracy - 100% training, 62% validation.

**Image Classification**  
*Task*: Use the pre-trained vgg16 model on its own to test baseline classification abilities.  
*Setup*: I selected 10 images for the model to identify with its 1000 built-in classes.  
*Results*: I have given the models most likely classification as well as any others of interest.  
| Image                                             | Prediction                               |
|---------------------------------------------------|------------------------------------------|
| hippopotamus                                      | hippo (74%)                              |
| empty shot of watering hole                       | valley (32%), coral reef (6%)            |
| same empty shot at night                          | hay (14%), volcano (4%)                  |
| porcupine at night                                | badger (24%), porcupine (21%)            |
| someone's legs                                    | maillot (16%)                            |
| a golf cart                                       | golf cart (20%)                          |
| elephant legs with others in the background       | indian elephant (42%), african elephant (30%) |
| two impala slightly out of frame                  | impala (40%)                             |
| giraffe legs                                      | hyena (16%), zebra (15%)                 |
| blurry hyena at night                             | ram (8.5%), hyena (6%)                   |

**Transfer Learning**  
*Task*: Applying transfer learning to densenet (another image classifier, more efficient for this purpose than vgg16) so I can use it as the base for an improved logistic regression model.  
*Setup*: The same training and validation sets from part 1, and then a grid search over LR's hyperparameters for optimization.  
*Results*: Logistic regression model (with C=10 and max_iter=3000) accuracy - 100% training, 86% validation.  

**Object Detection**  
*Task*: Object detection with the pretrained YOLO model.  
*Setup*: I used the same 10 images from part 2 for the model to pick objects out of.  
*Results*:  The golf cart example is notable because there are only 2 people in the cart, YOLO also picked up their golf bags.
| Image                                             | Prediction                                      |
|---------------------------------------------------|-------------------------------------------------|
| hippopotamus                                      | dog (83%)                                       |
| empty shot of watering hole                       | no detections                                   |
| same empty shot at night                          | no detections                                   |
| porcupine at night                                | dog (29%)                                       |
| someone's legs                                    | person (95%), person (26%)                      |
| a golf cart                                       | truck (51%), 4 people (37–78%)                  |
| elephant legs with others in the background       | 3 elephants (74–85%)                            |
| two impala slightly out of frame                  | cow (39%), horse (33%)                          |
| giraffe legs                                      | giraffe (87%)                                   |
| blurry hyena at night                             | no detections                                   |

**Primary Takeaways**  
Image classification achieved moderate success, in order to really work I would need the model to be better trained on savanna-relevant data.
Transfer learning took this step, I like the initial results and will focus on this model for a higher degree of training and fine-tuning, including more diverse classes.
Object detection could be useful for filtering out empty and human-caused pictures, but struggled on the animals. I would like to explore using it for the former, but need to be careful about it dropping useful photos like the final hyena one.

**Extra Testing**  
*Task*: Examine edge cases where YOLO object detection could fall through if I use it to filter out empty photos.  
*Setup* I found a new set of 10 images to test with, mainly night shots that are partials or poorly focused.  
*Results*: YOLO again struggles with blurry night images, but is able to find things in most other troublesome shots. The stray "cow" prediction is notable because it found it under the legs of the elephant, and as far as I can tell there is no calf hiding there that triggered it.
| Image                                             | Prediction                                      |
|---------------------------------------------------|-------------------------------------------------|
| close-up antelope neck (night)                    | no detections                                   |
| hyena looking away (night)                        | cat (89%)                                       |
| close-up impala legs (day)                        | 3 horses (42-64%)                               |
| civet at the edge of the light (night)            | cat (35%)                                       |
| extremely close-up antelope face (night)          | person (32%)                                    |
| blurry lioness (night)                            | no detections                                   |
| elephant at the edge of the light (night)         | elephant (58%), cow (30%)                       |
| blurry leopard (night)                            | no detections                                   |
| male lion with kill, half in frame (night)        | elephant (49%)                                  |
| thick grass (day)                                 | no detections                                   |

