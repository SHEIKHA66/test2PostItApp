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
    
  

    </tbody>
      <td>
 <p>Imports</p>
 <p>Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force </p>
  <p>npx create-react-app . </p>
 <p>npm install</p>
 <p> npm start</p>
 <p>npm install reactstrap</p>
 <p>npm install  bootstrap</p>
 <p>npm install yup</p>
 <p>npm install react-hook-form</p>
 <p>npm install @hookform/resolvers</p>
 <p>npm install @reduxjs/toolkit react-redux</p>

    
------ Home----------
 <p> import logo from "../Images/logo-t.png";</p>
<p>import Posts from "./Posts";</p>
<p>import SharePost from "./SharePost";</p>
<p>import User from "./User";</p>
<p>import { Container, Row, Col } from "reactstrap"; //import the Reactstrap Components</p>

<p>const Home = () =علامة اصغر من او اكبرمن و علامة القوس مال كرلي براكت  {</p>
 


    </td>
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
