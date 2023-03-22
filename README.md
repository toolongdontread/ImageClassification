# ImageClassification-CarModel
This is a CNN model and aims to specify 20 classes of cars. Of course it can specify the other things. But you have to use the ```train.py``` to train your own data. Otherwise, just simply use the ```trained_model.h5``` for classification. 

## Preparation
1. Install python to your PC (Remember to add it to PATH)
2. Prepare your own dataset. If you don't, there is a deafault dataset for you

## Default dataset
There are 20 classes of cars in the folder 'DATA'. Please enlarge it if necessary

## ```train.py```
This is the python file for model training

There is also a validation set inside
### How to run
VS Code: Just simply click the 'Run' button on the VS Code

Terminal: Type ```python train.py```

## ```test.py```
This is the python file for testing the model accuracy
### How to run
Type ```python test.py 'dataset_path model_path output_path'```

## ```result.txt```
This is a text file which stores the test loss and test accuracy

## ```trained_model.h5```
This is a pre-trained model

The average accuracy now is 79%.

## Disclaimer
The model is still under development. We cannot ensure that all programs are error-free. YOU HAVE TO TAKE YOUR OWN RISK while using our programs.
