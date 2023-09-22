#Importing necessary modules
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Data collection
data = pd.read_csv("Cardiovascular disease prediction dataset.csv")

#Preparing replacement to process feature data
rep_f = {
    "Age (years)": [0],
    
    'Gender':["Male",
              "Female"],
    
    'Marital status': ["Single",
                       "Married",
                       "Divorced"],
    
    'Occupation': ["Professional",
                   "Business",
                   "Government Employee", 
                   "Daily wage Labour", 
                   "Unemployed", 
                   "Student", 
                   "Retired", 
                   "Housewife", 
                   "Salaried Employee"],

    'Working hours per week': ["Upto 40 hours per week",
                               "40- 50 hours per week",
                               "More than 50 hours per week"],
    
    'History of Diabetes (Type 2)': ["NonDiabetic",
                                     "Recently detected",
                                     "Longterm Controlled",
                                     "Longterm Uncontrolled"],
    
    "History of Hypertension": ["Yes",
                                "No"],
    
    "History of Dyslipidemia (altered lipid profile)": ["Yes",
                                                        "No"],
    
    'H/o Alcohol consumption': ["Yes",
                                "No"],
    
    "H/o Smoking": [0],
    
    "Are you engaged in any physical activity? (Walking/Jogging/Running/Swimming/Cycling/Exercise/ Workouts)": [0],
    
    "Diet preference?": [0],
    
    "How often do you eat in restaurants?": ["Less than once a week",
                                            "Once a week",
                                            "More than once a week"],
    
    "Hours of Sleep per day?": ["Less than 6 hours per day",
                              "6 to 8 hours per day",
                              "More than 8 hours per day"],
    
    "Do you suffer from Sleep Apnoea?": ["Yes",
                                         "No"],
    
    "H/o chronic kidney disease": ["Yes",
                                   "No"],
    
    "H/o Cerebrovascular disease (Stroke/ Paralysis/TIA)": ["Yes",
                                                            "No"],
    
    "Family History of Cardiovascular disease": ["Yes",
                                                 "No"],
    
    "Pulse": [0],
    
    "Systole": [0],
    
    "Diastole": [0],
    
    "Body Mass Index (BMI)   Kg/m2": ["Normal (19-24)",  
                                     "Overweight (25-30)",
                                     "Obese ( >30)"],
    
    "Total Cholesterol": ["Normal (Upto 200 mg/dl)",
                         "Increased ( more than 200 mg/dl)"],
    
    "Triglycerides": ["Normal ( Upto 200 mg/dl)",
                     "Increased ( More than 200 mg/dl)"],
    
    "LDL": ["Normal (Upto 130 mg/dl)",
            "Increased ( More than 130 mg/dl)"],    
    
    "HDL": ["Normal (More than 35mg/dl)",
            "Decreased (Less than 35mg/dl)"],
    
    "Fasting Blood sugar": ["Normal (Upto 125mg/dl)",
                            "Increased (More than 125mg/dl)"],
    
    "Post Prandial Blood sugar": ["Normal (Upto 180mg/dl)",
                                  "Increased (More than 180mg/dl)"],
    
    "HbA1c": ["Less than 6.1",
             "6.1 to 7.1",
             "More than 7.1"],
    
    "hsCRP": ["Normal (upto 3mg/L)",
              "Increased (More than 3mg/L)"],
    
    "Do you consume Whole fat dairy or animal products": ["Yes",
                                                          "No"]
}

#Column-wise feature data processing
data["H/o Smoking"] = np.where(data["H/o Smoking"]=="No", 0, np.where(data["If smoker, since how many years?"]=="Less than 5 years", 1, 2))
data = data.drop(["If smoker, since how many years?"], axis = 1)

data["Are you engaged in any physical activity? (Walking/Jogging/Running/Swimming/Cycling/Exercise/ Workouts)"] = np.where(data["Are you engaged in any physical activity? (Walking/Jogging/Running/Swimming/Cycling/Exercise/ Workouts)"]=="No", 0, np.where(data["If yes, how many minutes per day?"]=="Less than 40 minutes per day", 1, np.where(data["If yes, how many minutes per day?"]=="40 minutes per day", 2, 3)))
data = data.drop(["If yes, how many minutes per day?"], axis = 1)

data["Diet preference?"] = np.where(data["Diet preference?"]=="Vegetarian", 0, np.where(data["If Non-Vegetarian, do you consume red meat?"]=="No", 1, 2))
data = data.drop(["If Non-Vegetarian, do you consume red meat?"], axis = 1)

s = []
d = []
for i in data["Blood pressure (mmHg)"]:
    s.append(int(i.split("/")[0]))
    d.append(int(i.split("/")[1]))
    
data["Systole"] = s
data["Diastole"] = d
data = data.drop(["Blood pressure (mmHg)"], axis = 1)

for col in rep_f.keys():
    data[col] = data[col].replace(rep_f[col], list(range(len(rep_f[col]))))
    
    a = data[col].copy()
    
    if not a.dtype == 'O':
        a = (a - a.min())/(a.max()-a.min())
    data[col] = a

#Outcome data processing
rep_o = [np.nan, 
       "Insignificant coronary artery disease", 
       "Significant coronary artery disease"]

data['Findings of Angiography'] = data['Findings of Angiography'].replace(rep_o, [0,1,2])

#Separating Features and Outcomes
X = data[rep_f.keys()]
Y = data["Findings of Angiography"]

#Fitting data into model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X.iloc[:70],Y.iloc[:70])

#Making prediction and calculating accuracy
pred= model.predict(X.iloc[70:])

acc = accuracy_score(Y.iloc[70:], pred)
    
print(acc)