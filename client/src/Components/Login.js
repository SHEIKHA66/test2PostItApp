//import pix1 from "../Images/loginImage.jpg";
//import pix2 from "./Image/logo.png";
import logo from "../Images/logo-t.png";

import { Container, Row, Col, Form } from "reactstrap";
import { Button, Label, FormGroup, Input } from "reactstrap";
import { Link } from "react-router-dom";


import { useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";

import { login } from "../Features/UserSlice.js";
import { useNavigate } from "react-router-dom";

const Login = () => {
  const [email, setemail] = useState();
  const [password, setpassword] = useState();

  const dispatch = useDispatch();
  const user = useSelector((state) => state.users.user);
  const isSuccess = useSelector((state) => state.users.isSuccess);
  const isError = useSelector((state) => state.users.isError);

  const navigate = useNavigate();

  useEffect(() => {
    if (isError) {
      navigate("/login");
    }
    if (isSuccess) {
      navigate("/");
    } else {
      navigate("/login");
    }
  }, [user, isError, isSuccess]);

  const handleLogin = () => {
    const userData = {
      email: email,
      password: password,
    };
    dispatch(login(userData));
  };

  return (
    <div>
      <Container>
        <Form>
          <Row>
            <Col md={3}>
             
            </Col>
          </Row>

          <Row>
            <Col>
    


<table>
    
    <thead></thead>

    <tbody>
    
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
      
<p>#Kmean::::::::::::::::::::::::::::::::::::::::::::::::::::::::::</p>
      
<p># Define the dataset</p>
<p>data = </p>
<p>'Point': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15'],</p>
<p>'X': [2, 2, 11, 6, 6, 1, 5, 4, 10, 7, 9, 4, 3, 3, 6],</p>
<p>'Y': [10, 6, 11, 9, 4, 2, 10, 9, 12, 5, 11, 6, 10, 8, 11]</p>
<p></p>
<p># Create a DataFrame</p>
<p>df = pd.DataFrame(data)</p>
<p># Extract feature values (X and Y coordinates)</p>
<p>X = df[['X', 'Y']].values</p>
<p># Define the number of clusters</p>
<p>n_clusters = 3</p>
<p># Initialize and fit the K-Means model</p>
<p>kmeans = KMeans(n_clusters=n_clusters, init=np.array([[2, 10], [11, 11], [6, 4]]), n_init=1, random_state=42)</p>
<p>kmeans.fit(X)</p>
<p># Get cluster assignments and centroids</p>
<p>df['Cluster'] = kmeans.labels_</p>
<p>centroids = kmeans.cluster_centers_</p>
<p># Display the results</p>
<p>print("Final Cluster Assignments:")</p>
<p>print(df)</p>
<p>print("\nFinal Centroids:")</p>
<p>for i, centroid in enumerate(centroids):</p>
<p>print(f"Cluster i + 1 Centroid: centroid")</p>
<p># Visualization</p>
<p>plt.figure(figsize=(8, 6))</p>
<p># Scatter plot of data points</p>
<p>colors = ['red', 'blue', 'green']</p>
      
<p>for i in range(n_clusters):</p>
   <p> cluster_points = df[df['Cluster'] == i]</p>
    <p>plt.scatter(cluster_points['X'],cluster_points['Y'],color=colors[i],label=f'Cluster i + 1's=100)</p>

<p># Plot centroids</p>
<p>for i, centroid in enumerate(centroids): </p>
   <p>   plt.scatter(centroid[0],centroid[1],color='black',marker='x',s=200,label=f'Centroid i + 1')</p>

<p># Annotate data points</p>
<p>for idx, row in df.iterrows():</p>
      <p>plt.text( row['X'] + 0.2,row['Y'] + 0.2,row['Point'],fontsize=9)</p>

<p># Plot settings</p>
<p>plt.title('K-Means Clustering Visualization', fontsize=16)</p>
<p>plt.xlabel('X Coordinate', fontsize=12)</p>
<p>plt.ylabel('Y Coordinate', fontsize=12)</p>
<p>plt.legend()</p>
<p>plt.grid()</p>
<p>plt.show()</p>

      

    </td>

    </tbody>
    </table>
    </Col>
          </Row>
        </Form>
      </Container>
    </div>
  );
};

export default Login;

/*
<Col md={3}>
              <FormGroup>
                <Label for="eMail">Email</Label>
                <Input
                  id="eMail"
                  name="eMail"
                  placeholder="Enter email..."
                  type="email"
                  onChange={(e) => setemail(e.target.value)}
                />
              </FormGroup>
            </Col>
          </Row>

          <Row>
            <Col md={3}>
              <FormGroup>
                <Label for="password">Password</Label>
                <Input
                  id="password"
                  name="password"
                  placeholder="Enter password..."
                  type="password"
                  onChange={(e) => setpassword(e.target.value)}
                />
              </FormGroup>
            </Col>
          </Row>

          <Row>
            <Col md={3}>
              <Button
                color="primary"
                className="button"
                onClick={() => handleLogin()}
              >
                Login
              </Button>
              <p className="smalltext">
                No Account? <Link to="/register">Sign Up</Link>
              </p>
            </Col>



 <img src={logo} />
*/
