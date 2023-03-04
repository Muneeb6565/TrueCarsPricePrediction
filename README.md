# Used Car Price Prediction
The quest to build an automobile started in 1672 with the invention of the first steam-powered vehicle. However, the first practical modern automobile was developed by Carl Benz in 1886, and later the Ford T model became the first mass-produced automobile in 1908. It was around this time that the car reselling industry also started emerging. Until recently, the reselling business operated only through offline channels. However, now with advancements in Computer Science and Information Technology the business moved online and customers could check the price of their old vehicles with the help of artificial intelligence online.

We are a team of 6 members working as data analyst interns at Technocolabs softwares. We have built a machine learning model to predict the prices of used vehicles for Truecar.Inc an US based automotive resale and pricing website.

Python was used for EDA, feature engineering, and model building, and Flask and AWS were used for deployment.

## The Dataset
1. Price - The Price of the car.	
2. Year	- The year in which the car was registered. The data ranges from 1997 - 2018 i.e 22 years.
3. Milage - Total miles the car has been driven.	
4. City - The city in which car is registered. Total number of such cities are 2446
5. State - The state in which car is registered. Total number of such states are 51.
6. Vin - A unique number which identifies the vehicle. 	
7. Make - The company which manufactured the car. Total number of such comanies are 58.
8. Model - Model name of the car. Total number of such models are 3107.

## Demo
You can access the predictive model we have built using the following link :
http://ec2-54-173-160-237.compute-1.amazonaws.com:8080/

## Process Flow

![flowchart12](https://user-images.githubusercontent.com/112056538/222874372-8fdc6a63-32a7-4f0a-8a95-5d2d2944c8d0.png)


### 1. Importing Libraries and Data
The first step was to import the required libraries for Data importing, data manipulation, mathematical calculations, visualisation and to import functions related to algorithm. The libraries - Pandas, Numpy, Scipy, Matplotlib, seaborn and sklearn.

### 2. Exploratory Data Analysis (EDA)
The dataset was explored and some analysis was done performed on it to reveal insights which are difficuilt to find in the raw data. So in EDA the first step was to check for *Shape* of the data then checking for *Null values* and *Duplicate values*. With the use of *df.describe* function the count, mean, standard deviation and the Inter quartile range was found. Then *Univariate analysis* was performed with help of visualisations to check for Skewness, Kurtosis and outliers also the Count of diffrent features were obtained. Similarly *Bivariate analysis* was performed to find the relation between two or more than two features.

### 3. Feature Engineering
Encoding of the data was done using *One hot encoding* and *Target encoding* as the number of labels were very high under some features and encoding it directly would have given memory error so for encoding purpose the top 10 labels under each feature were considered. As the values in the data had a high range so it was necessary to scale down the numbers in the data in range of 0-1 which simplifies the calculation for the model we were going to build. So, we used the *MinMaxScaler* function from the Sklearn library to transform the data. 

### 4. Feature Selection
#### 4.1 Principle Component Analysis 
The orignal dataset was encoded and hence had 39 columns representing different encoded features were present. So, here we applied 'Principal Component Analysis' to create clustures of data points which were closely related to each other and after applying PCA the number of features reduced from 39 to 8 and also total variance explained by the principle components was calculated. This can be a useful metric for evaluating the effectiveness of the dimensionality reduction process in retaining the original information in the dataset.

![PCA Explained varience score](https://user-images.githubusercontent.com/112056538/222874452-07ec1ee4-a0e6-43fe-9bbb-3d6218ca8e10.png)

#### 4.2 Multicolinearity
Then The 'Multicolinearity' between the independent variables was checked the *variance inflation factor* (VIF) is calculated for each feature in the 'x_train' dataset using the 'variance_inflation_factor()' function from the 'statsmodels' library. The VIF is a measure of the degree of multicollinearity between a feature and the other features in the dataset. A high VIF value indicates that the feature is highly correlated with other features, which can cause problems in some regression models. Finally, a pandas DataFrame is created with the VIF values and the column names are set to the feature names of the original dataframe (df). A heatmap was created using the seaborn library to visualise the results of Multicolinearity test.
#### 4.3 Normality of Residual
A simple linear regression model is trained and the resulting predictions are stored in the 'y_pred' variable. The residual values (i.e., the differences between the actual and predicted target variable values) are calculated and a 'kernel density estimate (KDE) plot' of the residual values is generated using seaborn library. Additionally, a 'probability plot' (i.e., Q-Q plot) of the residual values is generated. The probplot() function plots the residuals against a theoretical distribution (in this case, the normal distribution) and visually indicates whether the residuals are normally distributed or not. These plots can be used to assess the distribution of the residuals and whether the assumptions of the linear regression model are being met.
#### 4.4 Homoscadasticity
Homoscedasticity, also known as homogeneity of variance, is a statistical property that describes the condition in which the variance of the residuals is constant across the range of predicted values. It is an important assumption of linear regression modeling that should be assessed and satisfied for the model to be valid and reliable.A scatter plot is created where the x-axis represents the predicted values, while the y-axis represents the residuals. The scatter plot can be used to assess whether the linear regression model is making systematic errors (i.e., under- or over-predicting the target variable for a particular range of values), or whether the errors are random and evenly distributed across the range of predicted values. The scatter plot shows a distinct pattern which indicates that linear regression model is not capturing some important features of the data and needs to be improved.
#### 4.5 Autocorelation of Errors
The residuals from the linear regression model trained earlier is plotted using the line plot. The line plot can be useful for visualizing the distribution of the residuals and identifying any patterns or trends in the data that may indicate violations of the assumptions of linear regression modeling. A pattern is observed in the residuals plot which suggests that there is autocorrelation in the error terms. Autocorrelation occurs when the error terms at one point in time are correlated with the error terms at another point in time, either positively or negatively.

### 5. Model Building
##### 5.1 Linear Regression with r2_score as 41 %.
##### 5.2 Ridge Regression with r2_score as 42 %.
##### 5.3 Lasso Regression with r2_score as 39 %.
##### 5.4 Random Forest
Random forest is an ensemble learning method that combines multiple decision trees to create a more accurate and robust classifier. Each tree in the forest is constructed using a random subset of the training data and a random subset of the features. During the training process, each tree is trained to predict the class label of the training instances using a randomly selected subset of the features. The final class prediction of the random forest classifier is based on the majority vote of the individual trees. 
###### 5.4.1 Base Model
The model uses 40 decision trees (n_estimator = 40) and applies the 'score()' method to computes the accuracy of the random forest classifier on the test data.
###### 5.4.1 Cross Validation
+ Using KFold method split the dataset into 3 equal sized folds. 
+ Then the *'kf'* object created in the previous code block is used to split a list of numbers [1, 2, 3, 4, 5, 6, 7, 8, 9] into 3 different train and test sets. The output will show the indices of each training and testing set for the 3 different splits of the data. 
+ Next we defined a function called *'get_score'* that takes in a machine learning model, training data and testing data. Then using model.score() method we got the accuracy score of the model on the testing data. 
+ As we are dealing with an imbalanced dataset where one class has significantly more samples than the other so we use 'StratifiedKFold' as it preserves the proportion of samples from each class in each fold. An empty list 'scores_rf' is created. This list will be used to store the scores of a machine learning model on different train-test splits of a dataset, using cross-validation. 
+ Then load the 'load_digits()' dataset from the sklearn.datasets module, which contains images of digits 0-9. At the end of the loop, scores_rf will contain the accuracy scores of the random forest model on the 3 different train-test splits of the digits dataset. 
+ cross_val_score() is used to calculate the scores for a RandomForestClassifier model with 40 decision trees on the entire digits dataset. The cv=3 parameter specifies that the dataset should be split into 3 equal-sized folds for cross-validation. The 'Cross_val_score()' is used to calculate the scores for a RandomForestClassifier model with 40 decision trees on the entire digits dataset. The cv=3 parameter specifies that the dataset should be split into 3 equal-sized folds for cross-validation. The function returns an array of 3 scores, which are the accuracy scores of the model on each of the 3 folds of the digits dataset. 
+ The next step calculates the average accuracy score of a RandomForestClassifier model on the digits dataset using 10-fold cross-validation. The np.average() function from the numpy module is then used to calculate the average of these 10 scores which is 94.155 %.

![RF accuracy](https://user-images.githubusercontent.com/112056538/222873767-8dfab99d-9912-467a-bb46-37f8f8db6396.png)


## Deployment

### 1. Pipeline Building

A pipeline in machine learning refers to the process of chaining together different preprocessing steps and machine learning models into a single unit, which can be used to automate the entire process of data preparation, training, and prediction. This makes it easier to deploy machine learning models in production.

+ Imported the necessary modules and classes
+ Created a list of tuples where each tuple represents a step in the pipeline (The steps - EDA, FE, Model building)
+ Created an instance of the *'Pipeline'* class, passing the list of tuples created in the previous step.
+ Fitting the pipeline to the training data.
+ Used the fitted pipeline to make predictions on new data.
Using a pipeline to deploy machine learning models has several benefits, including increased efficiency, improved reproducibility, and easier maintenance.

### Deployment using AWS EC2 instance and Flask application

+ Logged in to the AWS Management Console and create an EC2 instance.
+ Chose an appropriate instance type and configuration.
+ Allocated an Elastic IP address to the instance.
+ Installed the necessary dependencies on the EC2 instance.
+ Copied the machine learning model file to the EC2 instance.
+ Created a Flask application to serve the machine learning model.
+ Run the Flask application on the EC2 instance.


![deployed model 01](https://user-images.githubusercontent.com/112056538/222890318-23d4fed9-4653-4e8e-a559-08ec4d8a1cb9.png)


![deployed model 2](https://user-images.githubusercontent.com/112056538/222890324-1e8bf7b1-1aa6-4879-936b-c30aa0ca6cc0.png)




## Acknowledgement

I would like to express my sincere gratitude to everyone who supported us throughout this project, "Used Car Price Prediction". First and foremost, we are deeply thankful to our project supervisor Mr. Yasin Shah and our mentor Mr. Gowtham Elangovan for providing us with the guidance, support, and resources necessary for the successful completion of this project. Their expertise and knowledge were invaluable, and their constructive feedback and suggestions helped us to improve the quality of our work. Finally, we would like to acknowledge the open-source community for providing us with access to the tools and libraries that we used in this project. Without their contributions, this project would not have been possible.

Thank you all for your support and encouragement.

## Authors

+ Muneeb Ur Rehman (Team leader) | https://www.linkedin.com/in/muneeburrehman065/ 
+ Logeshvaran A | https://www.linkedin.com/in/logeshvaran-a-0a4566242
+ Nashiha Sulthana | https://www.linkedin.com/in/nashiha-sulthana-6a4370205
+ Akhil Anarth | https://www.linkedin.com/in/akhil-anarth08




