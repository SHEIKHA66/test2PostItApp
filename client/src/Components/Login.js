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
    


from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import necessary libraries
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score



import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score

import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt




import matplotlib.pyplot as plt


import seaborn as sns



sns.heatmap(conf_matrxi,annot=True)


plt.show()







model=KNeighborsClassifier(n_neighbors=7,metric='euclidean',weights='uniform' )
model.fit(X_train, y_train)

pred=model.predict(X_test)

#Accuracy
score = accuracy_score(y_test, pred)*100
score

conf_matrix = confusion_matrix(y_test, pred)
class_report = classification_report(y_test, pred)
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)





# Define the parameter grid for GridSearchCV
param_grid = {
 'n_neighbors': [3, 5, 7, 9, 11],
 'metric': ['euclidean', 'manhattan', 'minkowski'],
 'weights': ['uniform', 'distance']
}

model = KNeighborsClassifier()

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)




from sklearn.linear_model import LogisticRegression
# Define the parameter grid for GridSearchCV
param_grid = { 
 'penalty': ['l1', 'l2', 'elasticnet', None] 
}

model = LogisticRegression()

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)




param_grid = { 
 'criterion': ['gini', 'entropy', 'log_loss'],
 'splitter' : ['best', 'random']  
}

model = DecisionTreeClassifier()

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)




svr = SVR()

grid_search = GridSearchCV(svr, param_grid, cv=5, 
                           scoring='neg_mean_squared_error',verbose=1)
grid_search.fit(X_train, y_train)


print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score (Negative MSE):", grid_search.best_score_)


best_svr = grid_search.best_estimator_
y_pred = best_svr.predict(X_test)
             

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nTest Set Mean Squared Error (MSE):", mse)
print("Test Set R-squared (R2):", r2)




#KKKKK


df = pd.DataFrame(data)
X = df[['X', 'Y']].values

n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, init=np.array([[2, 10], [11, 11], [6, 4]]), n_init=1, random_state=42)
kmeans.fit(X)

df['Cluster'] = kmeans.labels_
centroids = kmeans.cluster_centers_
# Display the results
print("Final Cluster Assignments:")
print(df)
print("\nFinal Centroids:")
for i, centroid in enumerate(centroids):
    print(f"Cluster {i + 1} Centroid: {centroid}")


plt.figure(figsize=(8, 6))

colors = ['red', 'blue', 'green']




for i in range(n_clusters):
    cluster_points = df[df['Cluster'] == i]
    plt.scatter(cluster_points['X'], cluster_points['Y'], color=colors[i], label=f'Cluster {i + 1}', s=100)

for i, centroid in enumerate(centroids):
    plt.scatter(centroid[0], centroid[1], color='black', marker='x', s=200, label=f'Centroid {i + 1}')

for idx, row in df.iterrows():
    plt.text(row['X'] + 0.2, row['Y'] + 0.2, row['Point'], fontsize=9)

plt.title('K-Means Clustering Visualization', fontsize=16)
plt.xlabel('X Coordinate', fontsize=12)
plt.ylabel('Y Coordinate', fontsize=12)
plt.legend()
plt.grid()
plt.show()


    
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
