#processing data
import pandas as pd

data = pd.read_csv("G3_heart_disease.csv")  # loading the dataset

data['Heart Disease'] = data['Heart Disease'].map({"Presence": 1, "Absence": 0})

print(data.columns[data.isna().any()])  # checking for NAN values

# seperate the dataset into input and output values
x = data.iloc[:, :-1].values  # choosing input columns
y = data.iloc[:, -1].values  # choosing output columns

# spliting dataset into training and testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=45, test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(x.shape)

#taking inputs from user on webpage
#
import streamlit as st
st.title("HEART DISEASE PREDICTION")

age = st.number_input("Age",step=1)
#
gender = st.selectbox("Sex",options=["Male","Female"])
if gender == "Male":
    gender = 1
else:
    gender = 0
#
chest = st.selectbox("Chest pain type",options=["Heart Attack","Angina","Pericarditis","Myocarditis"])
if chest == "Heart Attack":
    chest = 1
elif chest == "Angina":
    chest = 2
elif chest == "Pericarditis":
    chest = 3
else:
    chest = 4
#
bp = st.number_input("Blood Preasure",value=100)
#
Cholesterol = st.number_input("Cholesterol",value=100)
#
fbs = st.selectbox("FBS over 120kg",options=["YES","NO"])
if fbs == "YES":
    fbs = 1
else:
    fbs = 0
#
ekg = st.selectbox("EKG results",options=[0,1,2])
#
hr = st.number_input("Max HR",value=100)
#
exercise = st.selectbox("Exercise angina",options=["YES","NO"])
if exercise == "YES":
    exercise = 1
else:
    exercise = 0
#
depr = st.number_input("ST depression",step=0.0)
#
slope = st.selectbox("SLope of ST",options=[0,1,2,3])
#
vessels = st.number_input("Number of vessels fluoro",step=1)
#
thallium = st.number_input("Thallium",step=1)
#
new_user_input = [[age, gender, chest, bp, Cholesterol, fbs, ekg, hr, exercise, depr, slope, vessels, thallium]]
#

from streamlit_option_menu import option_menu
with st.sidebar:
    option = option_menu("SELECT ONE OF THE FOLLOWING",options=["SVM","KNN","DECISION TREE","XGBOOST","MODEL GRAPHS"])
    submit = st.button("submit")
#
if submit:
# SVM
    if option == "SVM":
        from sklearn.svm import SVC

        svc = SVC(probability=True)
        svc.fit(x_train, y_train)

        svc_test_prediction = svc.predict(x_test)
        svc_new_user_input_prediction = svc.predict(new_user_input)  # predicting result for new user input
        svc_probabilistic = svc.predict_proba(new_user_input)  # getting probabilities for each class
        #

        # Extracting predicted probabilities for the positive class (class 1)
        positive_probs = svc_probabilistic[0]  # Assuming there's only one prediction for new_user_input

        # Generating labels for the pie chart
        # labels = [f'Class {i + 1}' for i in range(len(positive_probs))]

        # Plotting the pie chart
        import matplotlib.pyplot as plt

        st.title("DIAGONISIS")
        if svc_new_user_input_prediction == 0:
            st.success("NO HEART DISEASE")
        else:
            st.error("POSSIBILITY OF HEART DISEASE")

        fig, ax = plt.subplots()
        ax.pie(svc_probabilistic[0], autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribution of Predicted Probabilities')
        ax.axis('equal')

        # Displaying the pie chart in Streamlit
        st.pyplot(fig)

        st.title("MODEL PARAMETERS")

        # getting accuracy ,precision of the model
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        st.success(f"ACCURACY OF THE MODEL : {accuracy_score(y_test, svc_test_prediction)}")
        st.error(f"PRECISION OF THE MODEL : {precision_score(y_test, svc_test_prediction)}")
        st.warning(f"RECALL SCORE OF THE MODEL IS : {recall_score(y_test, svc_test_prediction)}")
        st.info(f"F1 SCORE OF THE MODEL IS : {f1_score(y_test, svc_test_prediction)}")
# KNN
    elif option == "KNN":
        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier()
        knn.fit(x_train, y_train)

        knn_test_prediction = knn.predict(x_test)
        knn_new_user_input_prediction = knn.predict(new_user_input)  # predicting result for new user input
        knn_probabilistic = knn.predict_proba(new_user_input)  # getting probabilities for each class

        positive_probs = knn_probabilistic[0]  # Assuming there's only one prediction for new_user_input

        import matplotlib.pyplot as plt

        st.title("DIAGONISIS")
        if knn_new_user_input_prediction == 0:
            st.success("NO HEART DISEASE")
        else:
            st.error("POSSIBILITY OF HEART DISEASE")

        fig, ax = plt.subplots()
        ax.pie(knn_probabilistic[0], autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribution of Predicted Probabilities')
        ax.axis('equal')

        # Displaying the pie chart in Streamlit
        st.pyplot(fig)

        st.title("MODEL PARAMETERS")

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        st.success(f"ACCURACY OF THE MODEL : {accuracy_score(y_test, knn_test_prediction)}")
        st.error(f"PRECISION OF THE MODEL : {precision_score(y_test, knn_test_prediction)}")
        st.warning(f"RECALL SCORE OF THE MODEL IS : {recall_score(y_test, knn_test_prediction)}")
        st.info(f"F1 SCORE OF THE MODEL IS : {f1_score(y_test, knn_test_prediction)}")
# DECISION TREE
    elif option == "DECISION TREE":
        from sklearn.tree import DecisionTreeClassifier

        dt = DecisionTreeClassifier()
        dt.fit(x_train, y_train)

        dt_test_prediction = dt.predict(x_test)
        dt_new_user_input_prediction = dt.predict(new_user_input)  # predicting result for new user input
        dt_probabilistic = dt.predict_proba(new_user_input)

        positive_probs = dt_probabilistic[0]  # Assuming there's only one prediction for new_user_input


        import matplotlib.pyplot as plt

        st.title("DIAGONISIS")

        if dt_new_user_input_prediction == 0:
            st.success("NO HEART DISEASE")
        else:
            st.error("POSSIBILITY OF HEART DISEASE")

        fig, ax = plt.subplots()
        ax.pie(dt_probabilistic[0], autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribution of Predicted Probabilities')
        ax.axis('equal')

        # Displaying the pie chart in Streamlit
        st.pyplot(fig)

        st.title("MODEL PARAMETERS")

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        st.success(f"ACCURACY OF THE MODEL : {accuracy_score(y_test, dt_test_prediction)}")
        st.error(f"PRECISION OF THE MODEL : {precision_score(y_test, dt_test_prediction)}")
        st.warning(f"RECALL SCORE OF THE MODEL IS : {recall_score(y_test, dt_test_prediction)}")
        st.info(f"F1 SCORE OF THE MODEL IS : {f1_score(y_test, dt_test_prediction)}")
# XGBOOST
    elif option == "XGBOOST":
        from xgboost import XGBClassifier
        xgb = XGBClassifier()
        xgb.fit(x_train, y_train)

        xgb_test_prediction = xgb.predict(x_test)
        xgb_new_user_input_prediction = xgb.predict(new_user_input)  # predicting result for new user input
        xgb_probabilistic = xgb.predict_proba(new_user_input)


        positive_probs = xgb_probabilistic[0]  # Assuming there's only one prediction for new_user_input

        import matplotlib.pyplot as plt

        st.title("DIAGONISIS")

        if xgb_new_user_input_prediction == 0:
            st.success("NO HEART DISEASE")
        else:
            st.error("POSSIBILITY OF HEART DISEASE")

        fig, ax = plt.subplots()
        ax.pie(xgb_probabilistic[0], autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribution of Predicted Probabilities')
        ax.axis('equal')

        # Displaying the pie chart in Streamlit
        st.pyplot(fig)

        st.title("MODEL PARAMETERS")

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        st.success(f"ACCURACY OF THE MODEL : {accuracy_score(y_test, xgb_test_prediction)}")
        st.error(f"PRECISION OF THE MODEL : {precision_score(y_test,xgb_test_prediction)}")
        st.warning(f"RECALL SCORE OF THE MODEL IS : {recall_score(y_test, xgb_test_prediction)}")
        st.info(f"F1 SCORE OF THE MODEL IS : {f1_score(y_test, xgb_test_prediction)}")

    elif option == "MODEL GRAPHS":
        import matplotlib.pyplot as plt

        models = ["SVM", "KNN", "Decision Tree", "XGBoost"]
        accuracies = [57.4, 66.6, 66.6, 81.4]
        precisions = [61.5, 68.1, 65.3, 86.3]

        # Create a bar plot for accuracies
        plt.figure(figsize=(8, 6))
        plt.bar(models, accuracies,color='red')
        plt.xlabel('Models')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Comparison')
        st.pyplot(plt)

        # Create a bar plot for precisions
        plt.figure(figsize=(8, 6))
        plt.bar(models, precisions)
        plt.xlabel('Models')
        plt.ylabel('Precision (%)')
        plt.title('Precision Comparison')
        st.pyplot(plt)

        st.subheader("By Visual Comparison XGBOOST is th best")
else:
    st.error("CLICK SUBMIT BUTTON")