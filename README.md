**Introduction**  
This assignment contains two parts, data prediction and data classification.  
  
(1) Data prediction procedure  
Step 1: involves fitting a simple linear regression model to explore the relationship between the dependent variable watch_time and the independent variable confusion. The model is evaluated based on the R-squared value, and the results are visualized with a scatter plot of actual and predicted values.  
Step 2: extends the analysis to a multiple linear regression model where watch_time is predicted based on multiple independent variables: participation, confusion, and key_points. Similar to Objective 1, the model's performance is evaluated, and the slope and intercept are printed.  
  
(2) Data classification procedure  
Step 1: implements logistic regression to classify whether students are certified based on features like forum posts, grade, and assignment. It evaluates performance using recall, precision, F1 score, and accuracy, and visualizes the confusion matrix.  
Step 2: utilizes a decision tree classifier for the same task and similarly outputs evaluation metrics and a confusion matrix.  
Step 3: applies Naive Bayes classification, providing another comparison of performance metrics. Feature correlation and class distribution checks are also included for exploratory data analysis.  
  
**Key documents**  
(1) _1_regression_in_prediction.py_ is about data prediction.  
(2) _2_classification.py_ is about data classification.  
(3) _Results of assn_1 and assn_2.docx_ tells the outputs and analysis for the two python files.  
  
**Self-reflection**  
For this assignment, I would rate myself 4.5/5. I deducted 0.5 points because, despite my best efforts, I still struggled to grasp the mathematical principles underlying each machine learning model. This left me feeling somewhat confused and unsure about choosing the best model for different datasets. However, I am pleased that I wrote all the code independently, utilizing the documentation from the scikit-learn library, online video tutorials, and hints provided in the assignment introduction. One particular aspect that remains perplexing is the classification assignment: regardless of how small or large the C (regularization parameter) value is, every model consistently scored 100/100. This seems implausible to me. In addressing this concern, I consulted ChatGPT, which provided the following insights:  
  
_(1) Is there an issue with the dataset?  
The dataset might contain features that lead to overfitting, or the relationship between the labels (the certified column) and the features (forum.posts, grade, assignment) might be too simple, making it easy for the model to separate the data._    
_(2) Is there collinearity between the features?  
Your independent variables (forum.posts, grade, assignment) may be highly correlated, which could make the model's decision too straightforward. You can check the correlation between these features using pd.DataFrame.corr()._
_(3) Class imbalance problem  
If the certified classes are heavily imbalanced (e.g., most samples are either "yes" or "no"), the model might simply predict the majority class, leading to seemingly "perfect" performance. You can examine the data distribution by using data['certified'].value_counts().  
(4)  Randomness and overfitting issues  
Using a relatively large value of C=100.0 could cause overfitting, as this represents the inverse of regularization strength. You can try reducing the value of C to increase regularization and prevent the model from overfitting the training set._  
  
Despite these explanations, I am still left confused. This experience has highlighted the importance of understanding the theoretical foundations of machine learning models. I realize that while coding is an essential skill, a deeper comprehension of the underlying mathematical concepts is crucial for effectively selecting and implementing models. Moving forward, I plan to dedicate more time to studying these principles and seeking clarification on complex topics.  
