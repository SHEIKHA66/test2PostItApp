import * as yup from "yup";

export const userSchemaValidation = yup.object().shape({
  name: yup.string().required("name is required"),
  email: yup.string().email("Not an email form").required("email is required"),
  password: yup.string().min(4).max(20).required("password is required"),
  confpassword: yup
    .string()
    .oneOf([yup.ref("password"), null], "password is required")
    .required(),
});
