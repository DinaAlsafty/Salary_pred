## import libraries
import streamlit as st 
import numpy as np
import joblib
from utils import process_and_predict_new

## load model
# Model = joblib.load('linear_reg.pkl')
# inputs = joblib.load('inputs.pkl')

def salary_reg():
    #title
    st.title(' Salary Regression')
    st.markdown('<hr>', unsafe_allow_html=True)
    

    #input fields
    Gender = st.selectbox('Gender', options=['m','f'])
    percentage_10 = st.number_input('percentage_10')
    board_10 = st.text_input('board_10')
    graduation_12 = st.number_input('graduation_12')
    percentage_12 = st.number_input('percentage_12')
    board_12 = st.text_input('board_12')
    CollegeTier = st.selectbox('CollegeTier', options=[1,2])
    Degree = st.selectbox('Degree', options=['B.Tech/B.E.','M.Tech./M.E.','MCA'])
    Specialization = st.selectbox('Specialization', options=['electronics engineering',
                                                             'computer engineering',
                                                             'electronics & telecommunications',
                                                             'mechanical engineering',
                                                             'electrical engineering',
                                                             'civil engineering',
                                                             'other'])
    collegeGPA = st.number_input('collegeGPA')
    CollegeCityTier = st.selectbox('CollegeCityTier', options=[1,2])
    CollegeState = st.text_input('CollegeState')
    GraduationYear = st.number_input('GraduationYear')
    English = st.number_input('English')
    Logical = st.number_input('Logical')
    Quant = st.number_input('Quant')
    Domain = st.number_input('Domain')
    conscientiousness = st.number_input('conscientiousness')
    agreeableness = st.number_input('agreeableness')
    extraversion = st.number_input('extraversion')
    nueroticism = st.number_input('nueroticism')
    openess_to_experience = st.number_input('openess_to_experience')
    Age = st.number_input('Age')
    department_score = st.number_input('department_score')
    st.markdown('<hr>', unsafe_allow_html=True)

    if st.button('Predict Salary '):
        # concatenate users data
        new_data = np.array([Gender, percentage_10, board_10, graduation_12,
                          percentage_12, board_12, CollegeTier, Degree, 
                          Specialization, collegeGPA, CollegeCityTier,
                          CollegeState, GraduationYear, English, Logical, Quant, Domain,
                          conscientiousness, agreeableness, extraversion,
                          nueroticism, openess_to_experience, Age, 
                          department_score])
        # call the function from utils.py to apply the pipeline
        y_pred = process_and_predict_new(X_new=new_data)[0]
        y_pred = np.e**y_pred - 1

        # display the results
        st.success(f'Salary prediction is {y_pred}')  ## Reemeber you made log transformation --> You Must get the inverse (final prediction)





if __name__=='__main__':
    # call the fun 
    salary_reg()