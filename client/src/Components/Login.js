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
<p>from sklearn.preprocessing import LabelEncoder</p>
<p>from sklearn.neighbors import KNeighborsClassifier</p>
<p>from sklearn.metrics import accuracy_score, classification_report, confusion_matrix</p>


<p>from sklearn.datasets import fetch_california_housing</p>
<p>from sklearn.model_selection import train_test_split, GridSearchCV</p>
<p>from sklearn.svm import SVR</p>
<p>from sklearn.preprocessing import StandardScaler</p>
<p>from sklearn.metrics import mean_squared_error, r2_score</p>



<p>import pandas as pd</p>
<p>import numpy as np</p>

<p>from sklearn.linear_model import LogisticRegression</p>
<p>from sklearn.tree import DecisionTreeClassifier,plot_tree</p>
<p>from sklearn.metrics import classification_report,confusion_matrix</p>
<p>from sklearn.model_selection import train_test_split,cross_val_score</p>

<p>import numpy as np</p>
<p>from sklearn.cluster import KMeans</p>
<p>import pandas as pd</p>
<p>import matplotlib.pyplot as plt</p>
  <p>from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix</p>




<p>import matplotlib.pyplot as plt</p>


<p>import seaborn as sns</p>
    
    </td>

  <td>
  
  <p>
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
  
  </p>
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
