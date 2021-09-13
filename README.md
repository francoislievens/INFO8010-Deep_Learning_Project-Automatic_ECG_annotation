# INFO8010-Deep_Learning_Project




This code is organized as following: 

###The FINAL_MODEL directory:

In this folder, we set a copy of our model with our definitive model weights.
So, there are in this folder no code who refer to the training and testing process,
but additional libraries who permit to easily use the final algorithm in order to 
read, segmente and annote raw ECG's signal of any format.

The final demo of our algorithm can be run by using the DEMO.ipynb Notebook of this folder.


### The BeatSegmenter folder:

This folder contain the implementation of the beat segmenter model and 
all libraries who are use for the training process, the learning set handling and 
testing process.

### The BeatAnnoter folder:

By the same way, this folder contain all classes who refer to the training of our beat classifier.


### General Structure of BeatSegmenter and BeatAnnoter:

* Each model's folder contain a Model sub-folder who contain model's weights and training performances tracking. 
* Each model have a Training.py file who contain the training procedure
* Each model's folder contain associated notebook to show the learning evolution and some code to test the model
* Dataset handlers who are use are MIT_Dataset.py for the MIT-BIH dataset and BittiumTrainSetBuilder.py to manage the training process
on the large quantity of recording form the CHU. Theses records where previously segmented by samples who contain 30 min one leads records who
  are serialized on an external hard drive.
  This step was done using our implementation BittiumPreparator.py in the DataPreparator folder.




