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


### **Part 2: The Final Model**

**Take 1**  
Based on my research so far, I decided to pursue the transfer learning approach where I train a model to seperate images into my classifications. I manually sorted one folder of the camera trap photos I had been given for this project (1000 images) into my three categories from before: people, animals, and nothing. I chose these so that an ideal model could filter out the two most common kinds of noise in large sets of photos. With the same optimized hyperparameters from part 1, I trained a new model on this much larger dataset, which took about 15 minutes. Upon testing, I found it to have a shocking 55% accuracy on my validation set, which was ~150 images I hand selected from my entire pool of images I captured or was given. I made a confusion matrix to explore what was going wrong, and found that the majority of "people" images were being labelled as animals. While a more skilled model may be trained to differentiate between them, since mine is basically classifying photos as having animals, having one specific animal, or having nothing, I decided to change the classification to simply "something" or "nothing". In the data I was training with, there were relatively few shots of people in the way, so I figured they could be filtered out afterwards manually.  
After another 15-20 minutes of waiting, I tested again and got an improved validation score of 78%. Not ideal, but pretty decent. Examining this model's confusion matrix as well as some classification metrics showed me that it was pretty good at identifying animals (88% recall on "something") but half the time labelled empty scenes as "something" (48% precision on "nothing"). Due to time constraints I decided to go with this and make a proper program. I exported the model and built a proper python program that can be run with "python final.py --input_folder data/images --output_folder data/output" inside my project folder. First, I ran it on the same ten images I had used for object identification earlier. This gave... less than ideal results. The only positive identification was of an elephant, and every single other photo was labelled as "nothing". This was quite contrary to my validation work from earlier.  
  With overfitting in mind, I selected about 60 new images from the same camera trap as the training data to try again. Of these, only 5 were positively identified, and while they were correct, dozens of other animals were labelled as nothing. I'm not sure where the model went wrong, perhaps in export from my environment to a standalone script, but unfortunately I do not have the time to further analyze and debug it. However, I still enjoyed exploring this and would like to give it more time outside of this project. I would try breaking classification down by species to get the model away from problems of "this or that", and I would try to find more complicated models to use transfer learning on.

**Take 2**  
While writing my final report, I was inspired to give YOLO a try at acting as the proper model. I made a simple program that ran it over a set of input images similarly to the transfer learning model, but this time I divided them into my three original classes based on whether YOLO detected people, nothing, or anything else. This was actually quite successful, as it generally classified well, getting 44/56 animals. 7 were "no detections" as I worried in part 1, and 5 were "people" apparently, but this is still better than the other model. If I were to use YOLO with transfer learning, I think it would be the best yet, as I could individually classify animals the easiest provided I teach it that monkeys and impala are not horses, and that hyenas are not cats or sheep. This is unfortunately more complex than I have time for, but again something to explore more in the future and a way to end this on a positive note!