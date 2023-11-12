## main
import numpy as np
import pandas as pd

import os


## secondary
from datasist.structdata import detect_outliers
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

## sklearn -- preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn_features.transformers import DataFrameSelector
from sklearn.compose import ColumnTransformer
from category_encoders import BinaryEncoder
import joblib
from sklearn.linear_model import LinearRegression

## using pandas
TRAIN_DATASET_PATH = os.path.join(os.getcwd(), 'Engineering_graduate_salary.csv')
df = pd.read_csv(TRAIN_DATASET_PATH)


def edit_board_12(current_column):
    idx1 = df[df[current_column]=='0'].index
    df.loc[idx1, current_column] = df[current_column].mode()

    idx2 = df[(df[current_column]=='u p board')|(df[current_column]=='bright way college, (up board)')|(df[current_column]=='u p')
            |(df[current_column]=='up bord')|(df[current_column]=='upboard')|(df[current_column]=='up')|(df[current_column]=='up bourd')
            |(df[current_column]=='up baord')].index
    df.loc[idx2, current_column] = 'up board'

    idx3 = df[ (df[current_column]=='board of intermediate education')|(df[current_column]=='intermediate')
            |(df[current_column]=='board of intermeadiate education')|(df[current_column]=='intermidiate')
            |(df[current_column]=='baord of intermediate education')|(df[current_column]=='board of intermediate')].index
    df.loc[idx3, current_column] = 'intermediate board'

    idx4 = df[ (df[current_column]=='hisher seconadry examination(state board)')|(df[current_column]=='kerala state board')
            |(df[current_column]=='kerala state hse board')|(df[current_column]=='board of higher secondary examination, kerala')
            |(df[current_column]=='hse')].index
    df.loc[idx4, current_column] = 'kerala board'

    idx5 = df[ (df[current_column]=='p u board, karnataka')|(df[current_column]=='karnataka pre university board')
            |(df[current_column]=='pu')|(df[current_column]=='pu board ,karnataka')|(df[current_column]=='pu board karnataka')
            |(df[current_column]=='epartment of pre-university education')|(df[current_column]=='puboard')
            |(df[current_column]=='pre university board of karnataka')|(df[current_column]=='department of pre-university education')
            |(df[current_column]=='pre-university board')|(df[current_column]=='pre university board, karnataka')
            |(df[current_column]=='dept of pre-university education')|(df[current_column]=='karnataka pre-university board')
            |(df[current_column]=='dpue')| (df[current_column]=='pre university board')|(df[current_column]=='pre-university')
            |(df[current_column]=='pre-university')|(df[current_column]=='pub')|(df[current_column]=='pue')].index
    df.loc[idx5, current_column] = 'pu board'

    idx6 = df[ (df[current_column]=='state board - west bengal council of higher secondary education : wbchse')
            |(df[current_column]=='west bengal state council of technical education')
            |(df[current_column]=='west bengal council of higher secondary education')].index
    df.loc[idx6, current_column] = 'wbchse'

    idx7 = df[ (df[current_column]=='karnataka education board')|(df[current_column]=='karnataka state')
            |(df[current_column]=='karnataka state board')|(df[current_column]=='karnataka pu board')
            |(df[current_column]=='karnataka pre unversity board')].index
    df.loc[idx7, current_column] = 'karnataka board'

    idx8 = df[ (df[current_column]=='board of intermediate education,ap')|(df[current_column]=='board fo intermediate education, ap')
            |(df[current_column]=='state  board of intermediate education, andhra pradesh')|(df[current_column]=='ap')
            |(df[current_column]=='board of intermediate ap')|(df[current_column]=='board of intermediate education, andhra pradesh')
            |(df[current_column]=='board of intmediate education ap')|(df[current_column]=='andhra board')
            |(df[current_column]=='andhra pradesh state board')|(df[current_column]=='board of intermediate education, ap')
            |(df[current_column]=='ap intermediate board')|(df[current_column]=='ap board for intermediate education') ].index
    df.loc[idx8, current_column] = 'ap board'

    idx9 = df[ (df[current_column]=='directorate of technical education,banglore')
            |(df[current_column]=='department of technical education, bangalore')].index
    df.loc[idx9, current_column] = 'banglore board'
    
    idx10 = df[ (df[current_column]=='maharashtra state board')|(df[current_column]=='maharashtra satate board')
            |(df[current_column]=='maharashtra')|(df[current_column]=='maharashtra state(latur board)') ].index
    df.loc[idx10, current_column] = 'maharashtra board'

    idx11 = df[ (df[current_column]=='state')|(df[current_column]=='stateboard')].index
    df.loc[idx11, current_column] = 'state board'

    idx12 = df[ (df[current_column]=='higher secondary state certificate')
            |(df[current_column]=='certificate for higher secondary education (chse)orissa')
            |(df[current_column]=='board of secondary school of education')].index
    df.loc[idx12, current_column] = 'hsc'

    idx13 = df[ (df[current_column]=='rajasthan board of secondary education')
            |(df[current_column]=='board of secondary education, rajasthan')|(df[current_column]=='rajasthan board ajmer')].index
    df.loc[idx13, current_column] = 'rajasthan board'

    idx14 = df[ (df[current_column]=='mpboard')|(df[current_column]=='mp')|(df[current_column]=='mpbse')
            |(df[current_column]=='madhya pradesh board')|(df[current_column]=='madhya pradesh open school')].index
    df.loc[idx14, current_column] = 'mp board'

    idx15 = df[ (df[current_column]=='bseb')].index
    df.loc[idx15, current_column] = 'pseb'

    idx16 = df[ (df[current_column]=='mp')|(df[current_column]=='mpboard')|(df[current_column]=='state boardmp board ')].index
    df.loc[idx16, current_column] = 'mp board'

    r = df[current_column].value_counts()
    df.loc[df[current_column].isin(r[r<5].index), current_column] = df[current_column].mode()

## edit Specialization column 
def edit_Specialization(current_column):
    
    idx1 = df[ (df[current_column]=='mechanical and automation')|(df[current_column]=='mechanical & production engineering')
              |(df[current_column]=='automobile/automotive engineering')].index
    df.loc[idx1, current_column] = 'mechanical engineering'
        
    idx2 = df[ (df[current_column]=='instrumentation and control engineering')|(df[current_column]=='electronics and communication engineering')
              |(df[current_column]=='applied electronics and instrumentation')|(df[current_column]=='electronics and computer engineering')
              |(df[current_column]=='electronics & instrumentation eng')|(df[current_column]=='electronics')
              |(df[current_column]=='electronics and instrumentation engineering')].index
    df.loc[idx2, current_column] = 'electronics engineering'
    
    idx3 = df[ (df[current_column]=='information & communication technology')|(df[current_column]=='celectronics & telecommunications')].index
    df.loc[idx3, current_column] = 'telecommunication engineering'
        
    idx4 = df[ (df[current_column]=='computer application')|(df[current_column]=='computer networking')
              |(df[current_column]=='information technology')|(df[current_column]=='computer science & engineering')
              |(df[current_column]=='computer science and technology')|(df[current_column]=='computer and communication engineering')
              |(df[current_column]=='information science engineering')|(df[current_column]=='information science')].index
    df.loc[idx4, current_column] = 'computer engineering'

       
    idx5 = df[ (df[current_column]=='control and instrumentation engineering')|(df[current_column]=='electronics and electrical engineering')
              |(df[current_column]=='electrical and power engineering')].index
    df.loc[idx5, current_column] = 'electrical engineering'
    
    r = df[current_column].value_counts()
    df.loc[df[current_column].isin(r[r<13].index), current_column] = 'other'

def edit_board_10(current_column):
    idx1 = df[ (df[current_column]=='karnataka')|(df[current_column]=='karnataka secondary school of examination')
              |(df[current_column]=='karnataka secondary education board')|(df[current_column]=='karnataka education board')
              |(df[current_column]=='karnataka state education examination board')|(df[current_column]=='karnataka state board')
              |(df[current_column]=='karnataka board of higher education')|(df[current_column]=='karnataka secondary education examination board')
              |(df[current_column]=='karnataka education board (keeb)')|(df[current_column]=='karnataka secondary education')
              |(df[current_column]=='karnataka state secondary education board')].index
    df.loc[idx1, current_column] = 'karnataka board'

    idx2 = df[ (df[current_column]=='matric')].index
    df.loc[idx2, current_column] = 'metric'

    idx3 = df[ (df[current_column]=='maharashtra state board')|(df[current_column]=='maharashtra sate board')
              |(df[current_column]=='maharashtra state board,pune')|(df[current_column]=='maharastra board')
              |(df[current_column]=='maharashtra satate board')|(df[current_column]=='ssc maharashtra board')
              |(df[current_column]=='maharashtra state board of secondary and higher secondary education')
              |(df[current_column]=='maharashtra state board for ssc')].index
    df.loc[idx3, current_column] = 'maharashtra board'

    idx4 = df[ (df[current_column]=='state')|(df[current_column]=='stateboard')].index
    df.loc[idx4, current_column] = 'state board'   

    idx5 = df[ (df[current_column]=='up bourd')|(df[current_column]=='upboard')|(df[current_column]=='up')
              |(df[current_column]=='u p board')|(df[current_column]=='bright way college, (up board)')
              |(df[current_column]=='u p')|(df[current_column]=='up bord')|(df[current_column]=='up baord')
              |(df[current_column]=='up board , allahabad')|(df[current_column]=='up(allahabad)')
              |(df[current_column]=='up board,allahabad')].index
    df.loc[idx5, current_column] = 'up board'

    idx6 = df[ (df[current_column]=='mp')|(df[current_column]=='mpboard')|(df[current_column]=='state boardmp board ')
              |(df[current_column]=='mpbse')].index
    df.loc[idx6, current_column] = 'mp board'

    idx7 = df[df[current_column]=='0'].index
    df.loc[idx7, current_column] = df[current_column].mode()

    idx8 = df[ (df[current_column]=='rajasthan board of secondary education')|(df[current_column]=='rajasthan board ajmer')
              |(df[current_column]=='board of secondary education, rajasthan')|(df[current_column]=='secondary board of rajasthan')].index
    df.loc[idx8, current_column] = 'rajasthan board'

    idx9 = df[ (df[current_column]=='state board - west bengal board of secondary education : wbbse')|(df[current_column]=='wbbse')
              |(df[current_column]=='west bengal board of secondary education')].index
    df.loc[idx9, current_column] = 'wbbse board'

    idx10 = df[ (df[current_column]=='board of secondary education - andhra pradesh')
               |(df[current_column]=='board of secondary education,andhara pradesh')
              |(df[current_column]=='board of secondary education, andhra pradesh')
              |(df[current_column]=='board of secondary education,andhra pradesh')
              |(df[current_column]=='board of ssc education andhra pradesh')
              |(df[current_column]=='state board of secondary education, andhra pradesh')].index
    df.loc[idx10, current_column] = 'andhra pradesh board'


    idx11 = df[ (df[current_column]=='uttar pradesh')].index
    df.loc[idx11, current_column] = 'uttar pradesh board'

    idx12 = df[ (df[current_column]=='uttrakhand board')].index
    df.loc[idx12, current_column] = 'uttarakhand board'

    idx13 = df[ (df[current_column]=='secondary school of education')|(df[current_column]=='secondary state certificate')
              |(df[current_column]=='secondary school cerfificate')|(df[current_column]=='board secondary  education')
              |(df[current_column]=='board of secondary education')|(df[current_column]=='central board of secondary education')].index
    df.loc[idx13, current_column] = 'board of secondary school education'

    idx14 = df[ (df[current_column]=='jkbose')|(df[current_column]=='j&k state board of school education')].index
    df.loc[idx14, current_column] = 'jk board'

    idx15 = df[ (df[current_column]=='uttaranchal state board')].index
    df.loc[idx15, current_column] = 'uttranchal board'

    idx16 = df[ (df[current_column]=='kerala state technical education')].index
    df.loc[idx16, current_column] = 'kerala state board'

    idx17 = df[ (df[current_column]=='board of secondary education (bse) orissa')|(df[current_column]=='bse,orissa')].index
    df.loc[idx17, current_column] = 'board of secendary education orissa'

    idx18 = df[ (df[current_column]=='state board of secondary education, ap')].index
    df.loc[idx18, current_column] = 'ap state board'
    
    idx19 = df[ (df[current_column]=='jharkhand secondary education board')].index
    df.loc[idx19, current_column] = 'jharkhand secondary board'

    r = df[current_column].value_counts()
    df.loc[df[current_column].isin(r[r<5].index), current_column] = df[current_column].mode()

#edit conscientiousness & conscientiousness & extraversion & nueroticism column
def get_abs(col):
    
    df[col] = df[col].abs()
# Log Transform for the target
def log_transform(x):
    return np.log1p(x)

#rename columns
df.rename(columns = {'10percentage': 'percentage_10', '10board': 'board_10','12graduation': 'graduation_12'}, inplace = True)
df.rename(columns = {'12board':'board_12', '12percentage': 'percentage_12'}, inplace = True)


_ = edit_board_12('board_12')

## edit Specialization column
df['DOB'] = pd.to_datetime(df['DOB'], infer_datetime_format=True)

   
## apply
_ = edit_Specialization('Specialization')


    
_ = edit_board_10('board_10') 

##edit GraduationYear column
df['GraduationYear'].loc[df['GraduationYear']==0] = df['GraduationYear'].mode()


_ = get_abs('conscientiousness')
_ = get_abs('agreeableness')
_ = get_abs('extraversion')
_ = get_abs('nueroticism')
_ = get_abs('openess_to_experience')

# extract Age feature from DOB feature
df['Age'] = (datetime.now().year-df['DOB'].dt.year)
df.drop(columns=['DOB'],axis=1, inplace=True)

#extract deprtment score 
df['department_score'] = df['ComputerProgramming'] + df['ElectronicsAndSemicon'] + df['ComputerScience'] + df['MechanicalEngg'] + df['ElectricalEngg'] + df['TelecomEngg'] +df['CivilEngg'] 
# drop the original columns
df.drop(columns=['ComputerProgramming', 'ElectronicsAndSemicon', 'ComputerScience', 'MechanicalEngg', 'ElectricalEngg', 'TelecomEngg', 'CivilEngg'],axis=1, inplace=True)

df['department_score'].loc[df['department_score']==-7] = df['department_score'].median()

#drop unnessary row
df = df[df.Degree !='M.Sc. (Tech.)' ]

#drop unnessary columns

df.drop(columns=['ID', 'CollegeID', 'CollegeCityID'],axis=1, inplace=True)
## Get the Numerical cols firstly

df['Domain'].loc[df['Domain']==-1] = df['Domain'].median()

out_cols = ['percentage_10', 'percentage_12', 'collegeGPA', 'English', 'Logical', 'Quant','department_score','agreeableness',
            'department_score','Domain', 'conscientiousness', 'extraversion', 'nueroticism','openess_to_experience']
for col in out_cols:
    each_idx = detect_outliers(df=df, n=0, features=[col])
    each_median = df[col].median()

    df.loc[each_idx, col] = each_median

df['CollegeTier'] = df['CollegeTier'].astype(str)
df['graduation_12'] = df['graduation_12'].astype(str)
df['GraduationYear'] = df['GraduationYear'].astype(str)

## to feature and target
X = df.drop(columns=['Salary'], axis=1)
y = df['Salary']


## split to train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=58)


# Log Transform
y_train_log = log_transform(y_train)
y_valid_log = log_transform(y_valid)

#for numerical cols

num_cols = X_train.select_dtypes(include='number').columns.tolist()
#for category cols
cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()


num_pipeline = Pipeline(steps= [('Impute Numerical', KNNImputer(n_neighbors=5)),
                                 ('Scaling', StandardScaler())])

cat_pipeline = Pipeline(steps= [('Impute Categorical', SimpleImputer(strategy='most_frequent')),
                                 ('BE', BinaryEncoder())])

preprocessing = ColumnTransformer(transformers=[('Numerical', num_pipeline, num_cols),
                                                 ('Categorical', cat_pipeline, cat_cols)],
                                   )
final_pipeline = Pipeline(steps=[('Preprocessing', preprocessing), ('Model', LinearRegression())])
final_pipeline.fit(X_train, y_train_log)


def process_and_predict_new(X_new):
    '''  this function is to apply the pipeline to the user data.taking alist
    Args:
    *****
         (X_new: List)--> The user input as a list.

    Returns:
    *******
          (Xprocessed: 2D numpy array) --> the processed numpy array of user input          
    '''
#     inputs = joblib.load('inputs.pkl')
#     df_new = pd.DataFrame(columns = inputs)
    df_new = pd.DataFrame([X_new])
    df_new.columns = X_train.columns

    # Adjust the datatypes
    df_new['Gender'] = df_new['Gender'].astype(str)
    df_new['percentage_10'] = df_new['percentage_10'].astype(float)
    df_new['board_10'] = df_new['board_10'].astype(str)
    df_new['graduation_12'] = df_new['graduation_12'].astype(str)
    df_new['percentage_12'] = df_new['percentage_12'].astype(float)
    df_new['board_12'] = df_new['board_12'].astype(str)
    df_new['CollegeTier'] = df_new['CollegeTier'].astype(str)
    df_new['Degree'] = df_new['Degree'].astype(str)
    df_new['Specialization'] = df_new['Specialization'].astype(str)
    df_new['collegeGPA'] = df_new['collegeGPA'].astype(float)
    df_new['CollegeCityTier'] = df_new['CollegeCityTier'].astype(float)
    df_new['CollegeState'] = df_new['CollegeState'].astype(str)
    df_new['GraduationYear'] = df_new['GraduationYear'].astype(str)
    df_new['English'] = df_new['English'].astype(float)
    df_new['Logical'] = df_new['Logical'].astype(float)
    df_new['Quant'] = df_new['Quant'].astype(float)
    df_new['Domain'] = df_new['Domain'].astype(float)
    df_new['conscientiousness'] = df_new['conscientiousness'].astype(float)
    df_new['agreeableness'] = df_new['agreeableness'].astype(float)
    df_new['extraversion'] = df_new['extraversion'].astype(float)
    df_new['nueroticism'] = df_new['nueroticism'].astype(float)
    df_new['openess_to_experience'] = df_new['openess_to_experience'].astype(float)
    df_new['Age'] = df_new['Age'].astype(float)
    df_new['department_score'] = df_new['department_score'].astype(float)

#     print(df_new.shape)
#     X_proceessed = final_pipeline.transform(df_new)
    
    out = final_pipeline.predict(df_new)

#     print(X_proceessed.shape)
    return out
    

