# colab-wheat-coccinellid-Detection


## How to start with Colab?

i) create a google account. Install git from https://git-scm.com/downloads/win and clone the https://github.com/SauravUpadhyaya/wheat-coccinellid-detection-colab.git repository. To clone the repository, open terminal, goto your home and create a new folder folder. 

Use git clone https://github.com/SauravUpadhyaya/wheat-coccinellid-detection-colab.git to use this tutorial as a reference.

ii) Access the Google colab page (https://colab.research.google.com/)

    You should be able to see the below screen. Click on the "New notebook button". 

![alt text](<Screenshot 2025-07-16 122128.png>)

iii) As soon as you create a new notebook, you will see notebook where you can start coding. 


## What could be the next step after accessing Colab for the first time?

i) Use "colab_wheat_coccinellid_detection_project.ipynb" as your reference. 

ii) At this time, I assume you have "Wheat_Coccinellid.zip" data somewhere in your local machine. If not, download the dataset from  https://www.sciencedirect.com/science/article/pii/S2352340924005523 and scroll down until you see the "Data Availability" section. Click on the Wheat_Coccinellid (Original data) to start downloading. Great job!!!

iii) In your new notebook within the top tool bar section, you see "+ Code" and "+ Text" fields. Click on the te  "+ Code" to start writing your first code. It's always recommended to use GPU provided by the Colab. For this, click Runtime > Click Change runtime type > Choose Runtime Type as Python 3 > Choose Hardware accelerator as T4 GPU. Make sure that GPU is only available for at most 3 hours based on it's usage and time restriction. 

iv) As you have to upload your Wheat_Coccinellid.zip data, use the following code to upload that data.


from google.colab import files

uploaded = files.upload()


v) Since, its a zipped data, you have to unzip it. Use "!unzip Wheat_Coccinellid.zip" to start unzipping it. As soon as you execute that code, you will see train folder in your left side bar within your notebook.

vi) Make sure you are using colab_wheat_coccinellid_detection_project.ipynb for your reference. Now, declare some variables to give path to those sub-folders. 

Install the below libraries and other libraries based on your need:

%pip install pandas
%pip install matplotlib
%pip install scikit-learn
%pip install ultralytics

Also, don't forget to import the standard libraries such as math, os and so on for their usage. 

vii) Use the script or write on your own aiming to see the contents of those folders.

viii) Try to use "matplotlib" library to plot your images. If there is  text data try to use pandas dataframe to see those in tabular form.

ix) Write a scrip to draw the bounding boxes on images just to check if the labels provided are good enough for our project.

x) If everthing is good till this stage and you are comfortable to move forward, use train_test_split from sklearn.model_selection to split your data into training dataset. validation dataset and testing dataset. It is always highly recommended to use more data for training so that model learns over time.

xi) In (x) step, I talked about model. In this project, we will be using pre-trained YOLOv8m model for our dataset. Use "ultralytics" to import "YOLO". Use " YOLO('yolov8m.pt')" to load a pre-trained model YOLOv8m and declare the necessary parameters before training.  

Refer to https://docs.ultralytics.com/modes/train/#introduction to know more about training.

xii) During training time, try to check the results of each epcoch just to see whether there is overfitting/underfitting or not. In each epoch, metrics such as classification loss (cls_loss), bounding box loss (box_loss), dfl_loss, precision, recall,  mean average precision at IOU 0.5 (mAP50) and mAP50-95 were monitored to check the model’s learning progress and detect overfitting/underfitting scenarios throughout the training. 


Use https://docs.aws.amazon.com/machine-learning/latest/dg/model-fit-underfitting-vs-overfitting.html and https://www.ultralytics.com/glossary/overfitting resources to know more about Underfitting vs. Overfitting,

xiii) Evaluate your model result based on metrics such as classification loss (cls_loss), bounding box loss (box_loss), dfl_loss, precision, recall,  mean average precision at IOU 0.5 (mAP50) and mAP50-95.

Use the https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall resource to know more about the classification metrics.

xiv) Use the best.pt to test on your validation and test datasets.

xv) Use "matplotlib" library to plot your results using validation and test datasets. 

By this time, you have the result of the evaluation metrics. In this research, coccinellid detection was successfully performed on wheat plant images using the YOLOv8m model. You were also able to see the performance of model in the detection of coccinellid in maze plants.



## Bonus point: 

If you have GPU in your local machine, follow the below tutorial for better performance and higher efficiency.

https://github.com/SauravUpadhyaya/wheat-coccinellid-detection-project

Note: Use git clone https://github.com/SauravUpadhyaya/wheat-coccinellid-detection-project.git to use this tutorial as a reference.

https://www.youtube.com/watch?v=Rn2S1SFObn4

Thank you!



