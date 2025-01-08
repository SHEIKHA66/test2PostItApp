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
