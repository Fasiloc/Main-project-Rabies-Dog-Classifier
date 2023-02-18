# Rabies Dog Classifier

 ## About
Rabies is a deadly viral disease that affects the nervous system of mammals, including dogs. It
is transmitted through the saliva of an infected animal, usually through a bite. The disease can
be prevented through vaccination, but once symptoms develop, it is nearly always fatal. Early
signs of rabies in dogs include behaviour changes, muscle weakness, and fever, but as the
disease progresses, symptoms can include seizures, paralysis, and hydrophobia (fear of water).
Rabies viruses can spread to humans from other animals. Most of the cases Are spread through
dogs since humans interact with dogs more frequently. During pandemic lockdown period,
number of stray dogs increased in the street and caused to increase rabies cases also. This
caused wide fear towards stray dogs. There are also increased cases of attack toward dogs.
Through this project we are creating a program to identify rabies affected dogs from a video
feed by analysing different symptoms present on dogs. We are aiming to create system to help
government authorities and animal control services to find rabies infected dogs and give proper
treatment thus decrease rabies spreading further.

 ## Collection of data
 
 Classification of rabies dog project aim to distinguish dogs with rabies infection from
a video of a dog or live feed like CCTV feed. The program we create will build a model from
data we given to it. We needed to build a model that recognises and classify videos given to it
as normal dog and rabid dog (Dogs with rabies infection), it should also able to detect different
characteristics or behaviour that exhibits on both normal dog and rabid dog. We have to collect
video clippings of rabid dogs and normal dogs from the different sources in the internet. We
collected different types of videos like documentaries, news, camera footages and shorts from
different platforms like YouTube, Facebook, Vimeo, Dailymotion etc.

The collected videos are then cut to differentiate different characters by watching them
thoroughly and stored them on corresponding folders. Then those videos further edited to make
small clippings of lengths between 5 seconds to 11 seconds. We made grayscale videos of each
video to increase data samples and to balance data on both normal and rabies dog samples.
After data collection and manipulation is done, all the characteristic samples are checked to
ensure that it contains minimum amount of videos required to get a respectable accuracy in the
program. Characteristics which contains less videos are eliminated with a plan of adding them
in future when data are ready and available. Finally we limited number of videos to 60 to obtain
balance in data. The characteristics that lacked videos are added with grayscale videos of the
characteristics to reach the 60-video mark.

The final dataset contains 2 folders – Normal Dog and Rabies Dog, in normal Dog
folder, there are 5 characters of normal dog namely Barking, Digging, Playing, Running and
tail wagging and in Rabies Dog folder, there are 6 characters namely Bone in throat syndrome,
Dropped jaw, Hyper Salivation, Incoordination, Paralysis and Sudden Aggression.

 ## Program
 
 The collected videos (Dataset) stored in specific location and the location path stored into
 *datase_path* variable. Then convert the dataset into Data Frame in order to sort the videos
 into two sections as tag and video name using a for loop and iterated through all the videos
 in the folder and stored into a list and created the Data Frame. it is the stored into a csv
 file. The stored csv file is opened using *read_csv* function and stored to *train_df* and
 *test_df* variables.
 
 Next step is to create functions one *crop_center_square* for crop each frame from the video
 into a specific size, and *Load_video* function for loading each video to the model. Inside
 this function we call the crop_center_square function as part of resizing the frame from the
 video. Both of these functions are used as utilities for opening videos in OpenCV.
 
 Feature Extraction is used to extract features from each frame. i.e we have 10 features
in total, using the *build_feature_extractor* function it will give a feature for each
frame. In this function, we use Transfer Learning technique in order to predict a specific
feature for the frame. **Inception V3** is a type of Convolutional Neural Network. It consists
of many convolution and max pooling layers. Finally, it includes fully connected neural
networks. It is a pre-trained image recognition model that has been shown to attain
greater than 78.1% accuracy on the ImageNet dataset. The Inception V3 model is
maintained by Keras API in TensorFlow. We defined the function and stored it in a variable
named feature_extractor. The algorithm can understand data in the form of numerical values.
Here, our class names are in the form of strings, so that we need to convert these strings
into integers. For that we are using **StringLookup Layer** from Keras. All the labels has
been converted into numbers. i.e. each number corresponding to each label.

Now, we are setting default values for hyperparameters like image size, batch size, epochs,
max sequence length and number of features. After that, we define a function named
*prepare_all_videos*. It includes all the functions we defined earlier and the default
hyperparameters that were given. This function just performs all the functions that is
feeding videos into network,cropping frames, feature extraction, etc. After running the
*prepare_all_videos* function, we just print the results such as Frame Features in train set,
Train Labels and Test Labels in train set, etc. 

Now, we are entering into the RNN (Recurrent Neural Network) part. i.e, we are providing
the data into the Recurrent Layers like GRU. 
**Recurrent Neural Network (RNN)** – A Recurrent Neural Network (RNN) is a type of artificial
neural network which uses sequential data or time series data. An RNN Classifier with
LSTM algorithm used for feature extraction can increase the performance of the model.
**Gated Recurrent Unit (GRU)** – The Gated Recurrent Unit (GRU) is a type of Recurrent Neural
Network (RNN) that, in certain cases, has advantages over long short term memory (LSTM).
GRU uses less memory and is faster than LSTM, however, LSTM is more accurate when using
datasets with longer sequences. The *frame_feature_input* and *mask_input* are used to input
the data to the network. Then we add the layers like GRU, Dropout, Dense and use activation
functions Relu and Softmax.
Finally we call the model and compile the model. After compiling the model, the model is
saved in the current folder (the folder were the program file is saved). Then model fitting
is performed with a specific number of epochs. After model fitting the Test Accuracy of the
model is calculated. Next is the *prepare_single_video* function which is used to convert each
video that is loaded into the function into the format that the algorithm can understand.

The *Video_play* function is responsible for the detection of the subject (Here the subject
is Dog). For that we have written some OpenCV code that enables the model to fix 
rectangular frame around the subject. In order to focus on a specific subject we need to
import a special xml file DogCascade.xml that enables the detection of dogs. This file helps
the model to focus only on the dogs and avoid other objects. (This xml file is stored in the
same folder that the program file is stored.). This function is called at the end of the 
program after prediction. *Sequence_prediction* is the function that enables the model to make
predictions. i.e,. based on the given training data, the model will determine to which feature
(or category) the video is more dominant. Videos are randomly selected from the testing folder
using Random function during sequence prediction(These videos are stored in a variable called
*test_video*.

we are defining some variables that includes Rabies and Normal (two lists containing the
features denoting the characters. i.e, Rabies contain features of Rabies and Normal contains
features of Normal). Other variables are Rab, Nor, and No_Detect for denoting presence of 
each category. Inside the sequence prediction, we have written some code that help us to 
reach the final results. 

When the line for i in np.argsort(probabilities)[::-1] executes, the control moves into next
condition i.e. if count < len(Rabies) (This is to ensure that out of 10 features only 5 features
need to be printed. The count variable is already defined inside the sequence prediction). 
If it is False, the control just go out of the function. If it is True, then it jumps to next
condition that is, if class_vocab[i] == ‘No_Detection’. If this condition is True, it just
increase the value of No_Detect by 1.(No_Detect denotes the absence of dog in the video. It will
only works if there is No Dog found in the video). 

After all these conditions, the function justd prints the feature names and there probabilities
one by one (upto 5 features). During each executes, the condition below is executed. i.e. if
class_vocab[i] in Rabies. This condition is to check whether the printed feature belongs to
Rabies class or Normal class. If it belongs to Rabies, then value of Rab increase by 1 else
value of Nor increase by 1. After Sequence prediction, 5 features along with there probabilities
will be printed. These features will be either dominant to Rabies or Normal. Among the 5
features, if 3 or more features denotes Rabies, then the final result can be assumed as Rabies
else it is Normal. If No Detection have high priority among the features then the result can be
assumed as No Dog Detected. The Rab, Nor and No_Detect variables denotes the same. i.e. if Rab
is greater, the result will be Rabies. If Nor is greater, result will be Normal. IfNo_Detect is
greater, the result will be No Dog Detected.
