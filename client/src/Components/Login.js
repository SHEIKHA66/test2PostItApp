  <td>

    
   <p> import pandas as pd</p>
<p>from sklearn.preprocessing import LabelEncoder</p>
<p>from sklearn.preprocessing import StandardScaler</p>
<p>from sklearn.model_selection import train_test_split, GridSearchCV</p>
<p>from sklearn.neighbors import KNeighborsClassifier</p>
<p>from sklearn.metrics import accuracy_score, classification_report, confusion_matrix</p>
<p>from sklearn.linear_model import LogisticRegression</p>
<p>from sklearn.datasets import fetch_california_housing</p>
<p>from sklearn.svm import SVR</p>
<p>from sklearn.metrics import mean_squared_error, r2_score</p>
<p>import numpy as np</p>
<p>from sklearn.cluster import KMeans</p>
<p>import matplotlib.pyplot as plt</p>
<p>from sklearn import datasets</p>
<p>import sklearn.model_selection as skms</p>
<p>from sklearn import neighbors, metrics</p>
<p>import seaborn as sns</p>
<p>from sklearn.model_selection import train_test_split</p>
<p>from sklearn.linear_model import LinearRegression</p>




<p>df=pd.read_csv(اسم الداتا.csv') </p>
<p>df.head()</p>

<p>label_encoder = LabelEncoder()</p>
<p>df.variety = label_encoder.fit_transform(df.اسم اخر كولوم) </p>

<p>x=df[['اسماء الكولوم كلهن عدا اخر واحد" ]]</p> 
<p>y=df['اسم اخر كولوم']df</p>



<p>X_train, X_test, y_train, y_test = train_test_split(x, y,  test_size=0.3, random_state=42)</p>



<p>model=KNeighborsClassifier(n_neighbors=7,metric='euclidean',weights='uniform' )</p>
<p>model.fit(X_train, y_train)</p>

<p>pred=model.predict(X_test)</p>

<p>model=DecisionTreeClassifier(criterion='gini', splitter='random')</p>
<p>model.fit(X_train, y_train)</p>

<p>pred=model.predict(X_test)</p>


         
<p>#Accuracy</p>
<p>score = accuracy_score(y_test, pred)*100</p>
<p>score</p>

<p># Confusion Matrix and Classification Report</p>
<p>conf_matrix = confusion_matrix(y_test, pred)</p>
<p>class_report = classification_report(y_test, pred)</p>
<p>print("Confusion Matrix:")</p>
<p>print(conf_matrix)</p>
<p>print("\nClassification Report:")</p>
<p>print(class_report)</p>


       
<p># Define the parameter grid for GridSearchCV</p>

 <p>param_grid = </p>
         <p>'n_neighbors': [3, 5, 7, 9, 11], </p>
 <p>'metric': ['euclidean', 'manhattan', 'minkowski'],</p>
 <p>'weights': ['uniform', 'distance']</p> 
 
<p># Initialize the KNN classifier</p>
<p>model = KNeighborsClassifier()</p>


<p># Perform GridSearchCV with cross-validation</p>
<p>grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1) </p>
<p>grid_search.fit(X_train, y_train)</p>

<p># Display the best parameters and score from the grid search</p>
<p>print("Best Parameters:", grid_search.best_params_)</p>
<p>print("Best Cross-Validation Accuracy:", grid_search.best_score_)</p>


<p># Define the parameter grid for GridSearchCV</p>
<p>param_grid =  </p>
<p>'penalty': ['l1', 'l2', 'elasticnet', None]  </p>

<p># Initialize the KNN classifier</p>
<p>model = LogisticRegression()</p>



<p># Define the parameter grid for GridSearchCV</p>
<p>param_grid =  </p>
 <p>'criterion': ['gini', 'entropy', 'log_loss'],</p>
<p>'splitter' : ['best', 'random']  </p>
<p></p>
<p># Initialize the DT classifier</p>
<p>model = DecisionTreeClassifier()</p>


  <p>#svr::::::::::::::::::::::::::::::::::::::::::::::::::::::::::</p>
      
<p># Load the external dataset (California Housing dataset)</p>
<p>data = fetch_california_housing()</p>
<p>X = data.data </p>
<p>y = data.target </p>

<p># Preprocess the data (standardize features)</p>
<p>scaler = StandardScaler() </p>
<p>X_scaled = scaler.fit_transform(X)</p>

<p># Split the data into training and testing sets</p>
<p>X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)</p>

<p># Define the parameter grid for GridSearchCV</p>
<p>param_grid = </p>
 <p>'kernel': ['linear', 'rbf', 'poly'], # Kernel types</p>
 <p>'C': [0.1, 1, 10, 100], # Regularization parameter</p>
 <p>'epsilon': [0.01, 0.1, 0],</p>
<p> 'gamma': ['scale', 'auto'], # Kernel coefficient for RBF and poly</p>
<p> 'degree': [2, 3, 4] # Degree for polynomial kernel (only for 'poly')</p>
<p></p>

<p># Initialize the SVR model</p>
<p>svr = SVR()</p>

<p># Perform GridSearchCV with cross-validation</p>
<p>grid_search = GridSearchCV(svr, param_grid, cv=5,scoring='neg_mean_squared_error',verbose=1)</p>
<p>grid_search.fit(X_train, y_train)</p>

<p># Evaluate the model with the best parameters on the test set</p>

<p>best_svr = grid_search.best_estimator_</p>
<p>y_pred = best_svr.predict(X_test)</p>
             
<p># Print evaluation metrics</p>
<p>mse = mean_squared_error(y_test, y_pred)</p>
<p>r2 = r2_score(y_test, y_pred)</p>
<p>print("\nTest Set Mean Squared Error (MSE):", mse)</p>
<p>print("Test Set R-squared (R2):", r2)</p>
      
<p>#knn:::::::::::::::::::::::::::::::::::::::::::::::::::::::</p>
   <p>   # Load the Iris dataset</p>
<p>iris = datasets.load_iris()</p>
<p># Create a DataFrame for the dataset</p>
<p>iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)</p>
<p>iris_df['target'] = iris.target</p>
<p># Simple train/test split of the dataset</p>
<p>(iris_train_ftrs, iris_test_ftrs, iris_train_tgt, iris_test_tgt) = skms.train_test_split(iris.data, iris.target, test_size=.25)</p>
<p># Print the shapes of the training and testing sets</p>
<p>print("Train features shape:", iris_train_ftrs.shape)</p>
<p>print("Test features shape:", iris_test_ftrs.shape)</p>
<p>knn = neighbors.KNeighborsClassifier(n_neighbors=3)</p>
<p>fit = knn.fit(iris_train_ftrs, iris_train_tgt)</p>
<p>preds = fit.predict(iris_test_ftrs)</p>
<p>print("3NN accuracy:", metrics.accuracy_score(iris_test_tgt, preds))</p>



<p>#Linear Regression::::::::::::::::::::::::::::::::::::::::::::::::</p>
    


<p># Step 2: Load Data</p>
<p># Sample data: Hours studied vs Exam scores</p>
<p>data = </p>
    <p>'Hours_Studied': [2, 3, 4, 5],</p>
    <p>'Exam_Score': [60, 70, 80, 85]</p>
<p></p>
<p>df = pd.DataFrame(data)</p>

<p># Step 3: Extracting Features and Target Variable</p>
<p>X = df[['Hours_Studied']]</p>
<p>y = df['Exam_Score']</p>

<p># Creating a Linear Regression model</p>
<p>model = LinearRegression()</p>

<p># Fitting the model on the training data</p>
<p>model.fit(X, y)</p>



  <p># Step 5: Understanding and Displaying Coefficients</p>
<p>print("Intercept:", model.intercept_)</p>
<p>print("Coefficient:", model.coef_)</p>

<p># Step 6: Making Predictions on Existing Data</p>
<p>y_pred = model.predict(X)</p>

<p># Step 7: Evaluating Model Performance</p>
<p>sse = ((y - y_pred) ** 2).sum()</p>
<p>print('Sum of Squared Error:', sse)</p>

<p># Step 8: Making Predictions for New Data</p>
<p>new_data = pd.DataFrame(كرلي</p>
   <p> 'Hours_Studied': [2.5, 3.5, 4.5, 5.5]كرلي)</p>

<p>predicted_score = model.predict(new_data)</p>
<p>print('Predicted Exam Score:', predicted_score)</p>

<p># Step 9: Visualizing Results</p>
<p>plt.figure(figsize=(10, 6))</p>

<p># Plotting the training data points</p>
<p>plt.scatter(X, y, color='blue', label='Training Data')</p>

<p># Plotting the regression line</p>
<p>plt.plot(X, y_pred, color='red', label='Regression Line')</p>

<p>plt.title("Linear Regression: Hours Studied vs Exam Score")</p>
<p>plt.xlabel("Hours Studied")</p>
<p>plt.ylabel("Exam Score")</p>
<p>plt.legend()</p>
<p>plt.show()</p>


  <p>#:::::::::::::::::::::::::::::::::</p>
<p>  #create heatmap    </p>
<p>import matplotlib.pyplot as plt;</p> 
<p>import seaborn as sns;</p>
<p>sns.heatmap(conf_matrix, annot=True); plt.show()</p>

  <p>#Evl.</p>


<p>from sklearn.datasets import load_iris</p>
<p>from sklearn.model_selection import train_test_split</p>
<p>from sklearn.ensemble import RandomForestClassifier</p>
<p>from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix</p>

<p># Load dataset</p>
<p>data = load_iris()</p>
<p>X, y = data.data, data.target</p>

<p># Split the dataset into training and testing sets (70%-30%)</p>
<p>X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)</p>

<p># Train the model</p>
<p>model = RandomForestClassifier()</p>
<p>model.fit(X_train, y_train)</p>

<p># Make predictions and evaluate</p>
<p>y_pred = model.predict(X_test)</p>

<p># Accuracy</p>
<p>score = accuracy_score(y_test, y_pred) * 100</p>
<p>print('Accuracy:', score)</p>

<p># Confusion Matrix and Classification Report</p>
<p>conf_matrix = confusion_matrix(y_test, y_pred)</p>
<p>class_report = classification_report(y_test, y_pred)</p>

<p>print("Confusion Matrix:")</p>
<p>print(conf_matrix)</p>
<p>print("\nClassification Report:")</p>
<p>print(class_report)</p>



<p>from sklearn.datasets import load_iris</p>
<p>from sklearn.model_selection import LeaveOneOut</p>
<p>from sklearn.ensemble import RandomForestClassifier</p>
<p>from sklearn.metrics import accuracy_score</p>

<p># Load dataset</p>
<p>data = load_iris()</p>
<p>X, y = data.data, data.target</p>

<p># Initialize LOOCV</p>
<p>loo = LeaveOneOut()</p>
<p>model = RandomForestClassifier()</p>
<p>accuracies = []</p>
<p>i = 1</p>

<p># Perform LOOCV</p>
<p>for train_index, test_index in loo.split(X):</p>
   <p> print("Iteration :", i)</p>
    <p>i = i + 1</p>

    <p># Split the data into training and testing sets</p>
    <p>X_train, X_test = X[train_index], X[test_index]</p>
    <p>y_train, y_test = y[train_index], y[test_index]</p>

    <p># Train the model</p>
    <p>model.fit(X_train, y_train)</p>

    <p># Test the model</p>
   <p> y_pred = model.predict(X_test)</p>

    <p># Calculate accuracy for the current iteration</p>
    <p>accuracy = accuracy_score(y_test, y_pred)</p>
   <p> accuracies.append(accuracy)</p>

<p># Calculate and print the average accuracy</p>
<p>average_accuracy = sum(accuracies) / len(accuracies)</p>
<p>print(f"Average Accuracy with LOOCV: average_accuracy:.2f")</p>


<p>from sklearn.model_selection import KFold, cross_val_score</p>
<p>from sklearn.ensemble import RandomForestClassifier</p>
<p>from sklearn.datasets import load_iris</p>

<p># Load dataset</p>
<p>data = load_iris()</p>
<p>X, y = data.data, data.target</p>

<p># Set up K-Fold Cross-Validation with 10 folds</p>
<p>kf = KFold(n_splits=10, shuffle=True, random_state=42)</p>
<p>model = RandomForestClassifier()</p>

<p># Perform cross-validation</p>
<p>scores = cross_val_score(model, X, y, cv=kf)</p>

<p># Print results</p>
<p>print('Cross-Validation Scores:', scores * 100)</p>
<p>print('Average Accuracy:', scores.mean() * 100)</p>


<p>import numpy as np</p>
<p>from sklearn.utils import resample</p>
<p>from sklearn.ensemble import RandomForestClassifier</p>
<p>from sklearn.metrics import accuracy_score</p>
<p>from sklearn.datasets import load_iris</p>
<p>from sklearn.model_selection import train_test_split</p>

<p># Load dataset</p>
<p>data = load_iris()</p>
<p>X, y = data.data, data.target</p>

<p># Generate bootstrap samples</p>
<p>n_iterations = 10</p>
<p>bootstrapped_accuracies = []</p>

<p>for i in range(n_iterations):</p>
   <p> # Create a bootstrap sample</p>
    <p>X_resampled, y_resampled = resample(X, y, random_state=i)</p>
    <p>X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3)</p>
    
    <p># Train and evaluate model</p>
    <p>model = RandomForestClassifier()</p>
    <p>model.fit(X_train, y_train)</p>
    <p>y_pred = model.predict(X_test)</p>
    
    <p># Calculate accuracy</p>
    <p>accuracy = accuracy_score(y_test, y_pred)</p>
    <p>bootstrapped_accuracies.append(accuracy)</p>

<p># Calculate and print the average accuracy</p>
<p>print(f"Average Accuracy with Bootstrap Sampling: np.mean(bootstrapped_accuracies):.2f")</p>


  
    </td>
