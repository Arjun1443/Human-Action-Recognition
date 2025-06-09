# Human-Action-Recognition
## 1. Definition and Objective:

*Human Action Recognition (HAR) refers to the process of automatically identifying and labeling human activities from data.

*The primary goal is to understand human behavior through captured input and assign an appropriate action label (e.g., walking, jumping, waving).

## 2. Significance and Applications:
* HAR has gained substantial attention in computer vision and artificial intelligence due to its broad applicability:

* Surveillance and Security: Detecting abnormal behavior, intrusion detection.

* Healthcare Monitoring: Fall detection in elderly patients, posture correction.

* Human-Computer Interaction (HCI): Gesture recognition for VR/AR or gaming interfaces.

* Sports Analysis: Technique improvement and real-time activity tracking.

* Autonomous Driving: Understanding pedestrian movements.

* Smart Homes and IoT: Automated control based on user activity.

## 3. Data Modalities for HAR:
* HAR can be performed using multiple sensor data types, each offering distinct advantages:
* RGB (Red-Green-Blue) Video:
* Captures rich visual details like appearance and background.
* Suitable for traditional video surveillance and behavior understanding.
# Skeleton Data:
* Captures joint positions of the human body (usually extracted from RGB or depth data).
* Useful for abstracting motion patterns and reducing noise from the background.
# Depth Maps:
* Provide 3D structural information of the scene.
* Helpful in poor lighting conditions and for distinguishing overlapping objects.

# Infrared (IR) Images:

* Can detect temperature-based signals and operate in dark environments.

* Often used in night surveillance or medical scenarios.

# Point Cloud Data:

* Represents 3D spatial coordinates captured via LiDAR or depth sensors.

* Useful for fine-grained action analysis in 3D environments.

# Event Streams:

* High-speed asynchronous data from event cameras.

* Suitable for detecting fast motions with minimal latency and low power.

# Audio Signals:

* Capture sound associated with actions (e.g., clapping, footsteps).

* Useful in conjunction with visual modalities for contextual understanding.

# Acceleration (Inertial Sensors):

* Collect motion data via wearable devices (smartphones, smartwatches).

* Effective in mobile applications and low-cost setups.

# Radar Signals:

* Detect motion through electromagnetic wave reflection.

* Work in non-line-of-sight and low-light conditions.

* WiFi Signal (CSI - Channel State Information):

* Leverages WiFi signal fluctuations caused by human movements.

* Ideal for privacy-sensitive or occluded environments.

## 4. Advantages of Multi-Modality:

* Combining multiple modalities enhances recognition robustness and accuracy.

* Each modality contributes complementary information:

* For example, RGB + Skeleton improves spatial and temporal understanding.

* Audio + Acceleration enables recognition even without visual cues.

Generalization across different users, environments, and camera angles.
# What is Human Action Recognition (HAR)?
* Human activity recognition, or HAR for short, is a broad field of study concerned with identifying the specific movement or action of a person based on sensor data.
Movements are often typical activities performed indoors, such as walking, talking, standing,etc.
* Human activity recognition, or HAR for short, is a broad field of study concerned with identifying the specific movement or action of a person based on sensor data.
Movements are often typical activities performed indoors, such as walking, talking, standing,etc.
# Why it is important ?
*Human activity recognition plays a significant role in human-to-human interaction and interpersonal relations.
*Because it provides information about the identity of a person, their personality, and psychological state, it is difficult to extract.
*The human ability to recognize another person’s activities is one of the main subjects of study of the scientific areas of computer vision and machine learning. As a result of this research, many applications, including video surveillance systems, human-computer interaction, and robotics for human behavior characterization, require a multiple activity recognition system.
# Below are some practical applications of HAR:
![punching](https://github.com/user-attachments/assets/fbea80d3-45c4-4292-9f48-41186e334e49)

* Here we can see that the AI is able to identify what the man in the video is doing. This might raise the question of importance of identification of the action. Let's look at another example below:
  Here we can see that the model is able to identify the troublesome student who is running in the classroom highlighted in red. Whereas the other kids who are walking normally are colored in green.
![har_run](https://github.com/user-attachments/assets/c14092b8-8567-4faf-ab68-dd19b211ce9b)

This is a just small example of the endless applications that can help us automate monotonous and dangerous jobs.
# What is a CNN?
* A convolutional neural network (CNN) is a type of artificial neural network used in image recognition and processing that is specifically designed to process pixel data.

* CNNs are powerful image processing, artificial intelligence (AI) that use deep learning to perform both generative and descriptive tasks, often using machine vison that includes image and video recognition, along with recommender systems and natural language processing (NLP).
# VGG16
* The ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) is a yearly event designed to highlight and test computer vision models. During the 2014 ImageNet challenge, Karen Simonyan and Andrew Zisserman from the Visual Geometry Group at the Department of Engineering Science, University of Oxford, presented their model in a paper titled “VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION,” which achieved 1st and 2nd place in both object detection and classification. The original paper can be accessed via the link below:

1409.1556.pdf (arxiv.org)

* A convolutional neural network, commonly referred to as a ConvNet, is a type of artificial neural network. This network consists of an input layer, an output layer, and several hidden layers. VGG16 is a specific type of CNN (Convolutional Neural Network) recognized as one of the leading models in computer vision today.

* VGG16 serves as an object detection and classification algorithm capable of categorizing 1000 images across 1000 distinct categories with an accuracy of 92.7%. It is widely regarded as one of the most effective algorithms for image classification and is user-friendly when applied with transfer learning.

# Transfer Learning
* Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task. It is a popular approach in deep learning where pre-trained models are used as the starting point on computer vision and natural language processing tasks given the vast compute and time resources required to develop neural network models on these problems and from the huge jumps in skill that they provide on related problems.

# Sample training images data
def show_img_train():
    img_num = np.random.randint(0,12599)
    img = cv.imread('data/train/' + train_action.filename[img_num])
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title(train_action.label[img_num])
 show_img_train()

# Action Distribution
![image](https://github.com/user-attachments/assets/641ec10d-9c22-45aa-902c-d80089ef3478)

# Model Summary
* The loss function that we are trying to minimize is Categorical Cross Entropy. This metric is used in multiclass classification. This is used alongside softmax activation function.

Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data. This algorithm is straight forward to implement and computationally efficient.

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
cnn_model.summary()

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 vgg16 (Functional)          (None, 512)               14714688  
                                                                 
 flatten (Flatten)           (None, 512)               0         
                                                                 
 dense (Dense)               (None, 512)               262656    
                                                                 
 dense_1 (Dense)             (None, 15)                7695      
                                                                 
=================================================================
Total params: 14,985,039
Trainable params: 270,351
Non-trainable params: 14,714,688
_________________________________________________________________
# Accuracy
from sklearn.metrics import accuracy_score, log_loss
print('Log Loss:',log_loss(np.round(y_preds),y_test))
print('Accuracy:',accuracy_score(np.round(y_preds),y_test))

* Log Loss: 12.417512465789333
* Accuracy: 0.6317460317460317
# Next Steps and Recommendations

* In order to improve the accuracy, we can unfreeze few more layers and retrain the model. This will help us further improve the model.

* We can tune the parameters using KerasTuner.

* The model reached a good accuracy score after the 20 epochs but it has been trained for 60 epochs which leads us to believe that the model is overfit. This can be avoided with early stopping.

* The nodes in the deep layers were fully connected. Further introducing some dropout for regularization can also be done to avoid over-fitting.
