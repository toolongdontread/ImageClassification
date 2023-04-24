# ImageClassification
This is a CNN model aims to specify 20 classes of cars. Of course it can specify other things. But you have to use the ```train2.py``` to train your own data. 

## Preparation
1. Install python to your PC (Remember to add it to PATH)
2. Prepare your own dataset. If you don't, there is a default dataset for you

## Default dataset
Train dataset: ```./csbh_data/train```

Test dataset: ```./DATA/test```

There are 20 classes of cars in the above folders. You can enlarge it if necessary.

## ```train2.py```
This is the python file for model training.

### How to run
VS Code: Just simply click the 'Run' button on the VS Code

Terminal: Type ```python train.py```

## ```test.py```
This is the python file for testing the model accuracy.
### How to run
Type ```python test.py dataset_path model_path output_path```

## ```result.txt```
This is a text file which stores the test loss and test accuracy.

## Where will be the model stored?
After running the ```train2.py```, a folder named ```model``` will be generated. You can find the trained model inside.

## Disclaimer
The model is still under development. We cannot ensure that all programs are error-free. YOU HAVE TO TAKE YOUR OWN RISK while using our programs!
