import streamlit as st 
import numpy as np 
import pandas as pd 
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from PIL import Image

# ML Libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# heading
st.markdown("<h1 style='text-align: center; color: yellow;'>DIABETES PREDICTION</h1>", unsafe_allow_html=True)
st.markdown("<p><TT>Designed and Developed by <a style='text-decoration:none;color:red' target='_blank' href='https://github.com/Ritesh778'>J.Ritesh</a>", unsafe_allow_html=True)

# sidebar for navigation
with st.sidebar:
    img = Image.open("ritss.jpeg")
    st.image(img)
    st.text(""" I am Ritesh pursuing BTech in the branch of CSE(artificial
intelligence and data science) at Vignan’s institute of information technology, Visakhapatnam.
I am a motivated individual with a solid work ethic and have the ability to
produce the best result in a pressurizing situation. Eager
to learn new technologies and methodologies. I’m also
enthusiastic about solving new challenges.""")
    st.subheader("Reach out through")
    socials = ["LinkedIn","Github", "GMail"]
    linkedin = "https://www.linkedin.com/in/ritesh-j-2b1331214/"
    github = "https://github.com/Ritesh778"
    mail = "riteshj0507@gmail.com"
    with st.expander("Links to all my Socials"):
        a = st.selectbox("Socials", socials)
        if a =="LinkedIn":
            st.write(linkedin)
        elif a =="Github":
            st.write(github)
        elif a=="GMail":
            st.write(mail)
    
st.write("Diabetes is a chronic disease that occurs when your blood glucose is too high. This application helps to effectively detect if someone has diabetes using Machine Learning. " )



#Get the data
df = pd.read_csv("diabetes.csv")

# replacting 0 with nan
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)


# replacing missing values

# function to find the mean 
def median_target(var):   
    temp = df[df[var].notnull()]
    temp = round(temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].mean().reset_index(), 1)
    return temp


# Glucose
df.loc[(df['Outcome'] == 0 ) & (df['Glucose'].isnull()), 'Glucose'] = 110.6
df.loc[(df['Outcome'] == 1 ) & (df['Glucose'].isnull()), 'Glucose'] = 142.3

# Blood pressure
df.loc[(df['Outcome'] == 0 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 70.9
df.loc[(df['Outcome'] == 1 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 75.3

# Skin thickness
df.loc[(df['Outcome'] == 0 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 27.2
df.loc[(df['Outcome'] == 1 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 33.0

# Insulin
df.loc[(df['Outcome'] == 0 ) & (df['Insulin'].isnull()), 'Insulin'] = 130.3
df.loc[(df['Outcome'] == 1 ) & (df['Insulin'].isnull()), 'Insulin'] = 206.8

# BMI
df.loc[(df['Outcome'] == 0 ) & (df['BMI'].isnull()), 'BMI'] = 30.9
df.loc[(df['Outcome'] == 1 ) & (df['BMI'].isnull()), 'BMI'] = 35.4



# splitting columns
X = df.drop(columns='Outcome')
y = df['Outcome']


#scaling
scaler = StandardScaler()
X =  pd.DataFrame(scaler.fit_transform(X), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])


# Split the dataset into 70% Training set and 30% Testing set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

x_tr = x_train.loc[:,['Insulin','Glucose','BMI','Age','SkinThickness']]

name = st.text_input('What is your name?').capitalize()

#Get the feature input from the user
def get_user_input():

    insulin = st.text_input('Enter your insulin 2-Hour serum in mu U/ml')
    glucose = st.text_input('What is your plasma glucose concentration?')
    BMI = st.text_input('What is your Body Mass Index?')
    age = st.text_input('Enter your age')
    skin_thickness = st.text_input('Enter your skin fold thickness in mm')

    
    user_data = {'Insulin': insulin,
                'Glucose': glucose,
                'BMI': BMI,
                'Age': age,
                'Skin Thickness': skin_thickness,
                 }
    features = pd.DataFrame(user_data, index=[0])
    return features
user_input = get_user_input()


bt = st.button('Get Result')

if bt:
    gb = GradientBoostingClassifier(random_state=1)
    gb.fit(x_tr, y_train)
    prediction = gb.predict(user_input)
    

    if prediction == 1:
        st.write(name,", you either have diabetes or are likely to have it. Please visit the doctor as soon as possible.")
        st.snow()
        
    else:
        st.write('Hurray!', name, 'You are diabetes FREE.')
        st.balloons()