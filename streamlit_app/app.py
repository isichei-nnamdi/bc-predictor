import streamlit as st
import pickle
#import joblib
import pandas as pd
from PIL import Image
#import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# from sklearn.svm import SVC
# from sklearn import decomposition
# from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import normalize
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import GridSearchCV | \C:\Users\isichei\Documents\Git\bc-predictor\streamlit_app\breast-cancer.csv
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.model_selection import train_test_split, KFold, cross_val_score

#Data importation 
data = pd.read_csv("https://github.com/isichei-nnamdi/bc-predictor/blob/main/streamlit_app/breast-cancer.csv", on_bad_lines='skip')
test_data = pd.read_csv('https://github.com/isichei-nnamdi/bc-predictor/blob/main/streamlit_app/y_test.csv', on_bad_lines='skip')

favicon = Image.open('https://github.com/isichei-nnamdi/bc-predictor/blob/main/streamlit_app/bc_fav.png')
st.set_page_config(page_title="BC-classifier", page_icon= favicon)


def main():
    st.header("Breast Cancer Classifier")
    

    page_options = ["Describe", "Predict"]
    selection = option_menu( menu_title=None,
                            options=page_options,
                            icons=["bar-chart-steps", "graph-up-arrow"],
                            orientation='horizontal',
                            styles={
                                        "container": {"padding": "0!important", "background-color": "#ED2E38"},
                                        "icon": {"color": "black", "font-size": "25px",  },
                                        "nav-link": {
                                            "font-size": "20px",
                                            "text-align": "center",
                                            "margin": "5px",
                                            "--hover-color": "#eee",
                                            "color": "white"
                                        },
                                        "nav-link-selected": {"background-color": "white", "color": "#ED2E38"},
                                    },
        )    
    
    if selection == 'Describe':
        option = st.selectbox(
        'Choose a descriptive option below:',
        ('Descriptive Statistics', 'Cancer Type Plot', 'Multicolinearity Estimator', 'Linearity Checker'))

        if option == 'Descriptive Statistics':
            st.markdown(""" <p align="justify"> Users can perform some descriptive analysis on the characteristics of the dataset they have linked to or uploaded in this section of the app. To run any relevant analysis, select any option from the list above.</p><br><br>
            <ul>
                <li><p align="justify"><b style= "color: #ED2E38">Cancer Type Plot:</b> This permits the user to look into how many instances of the target variable have been recorded (type of cancer). In addition to showing the total number of cases of each form of cancer found in the dataset, a bar plot will also be shown.</p>
                <li><p align="justify"><b style= 'color: #ED2E38'>Multicolinearity Estimator:</b> The function of the app that enables the user to explore the highly correlated features is the multicolinearity estimator. Since multicolinearity of feature variables has a detrimental effect on the model, the user must be aware of which variables are linked in order to make an informed decision before moving further with the model training.</p>
                <li><p align="justify"><b style= 'color: #ED2E38'>Linearity Checker:</b> Using this function of the app, the user can examine the features that are linearly related. The user selects the desired features from a drop-down menu of all the accessible features, and a scatter matrix is then created for visualization.</p>
            </ul>
            """,unsafe_allow_html=True)

        elif option == 'Cancer Type Plot':
            st.markdown(f"""<p align="justify">The dataset connected with this app was examined, and two distinct diagnoses of breast cancer were recorded as Benign (B) and Malignant (M). See the table and bar plot below for visualization of the frequency of each cancer type.</p>""", unsafe_allow_html=True)
            st.table(data['diagnosis'].value_counts())
            df = pd.DataFrame(data.groupby('diagnosis').count()['radius_mean']).reset_index()
            fig = px.bar(data_frame=df.rename(columns={'diagnosis': 'Diagnosis', 'radius_mean': 'Frequency'}),
                        x='Diagnosis',
                        y='Frequency',
                        color= 'Diagnosis',
                        title = 'Plot of Cancer Type')
            st.plotly_chart(fig, use_container_width=True)
        
        elif option == 'Multicolinearity Estimator':
            options = st.multiselect(
                    "Select features you'll like to test for multicolinearity from the list below:",
                    data.columns.drop(['id', 'diagnosis']), help="Select one or more features to check for multicolinearity")

            df = pd.DataFrame(data[options])
            
            st.table(df.head(5))
            df_2 = df.corr()
            fig = px.imshow(df_2,  aspect='auto')
            fig.layout.height = 500
            fig.layout.width = 2000
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            plt.clf()
            options = st.multiselect(
                    "Select features you'll like to test for linearity from the list below:",
                    data.columns.drop('id'), default='diagnosis', help="Select one or more features to check for linearity")

            df = pd.DataFrame(data[options])
            
            st.table(df.head(5))

            fig = px.scatter_matrix(data_frame=df, dimensions=df.columns.drop('diagnosis'), color='diagnosis')
            
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown("""This section of the app gives the user the ability to run a predictive analysis on new dataset, predicting the types of breast cancer.
            """,unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            dataframe = pd.read_csv(uploaded_file)
            st.markdown(f"The uploaded file has {len(list(dataframe))} columns which are: {list(dataframe.columns)}.")
            st.write(dataframe.head(5))

            # Loading model to compare the results
            tree_model = pickle.load(open('https://github.com/isichei-nnamdi/bc-predictor/blob/main/streamlit_app/DecisionTreeClassifier().pkl','rb'))
            gauss_model = pickle.load(open('https://github.com/isichei-nnamdi/bc-predictor/blob/main/streamlit_app/GaussianNB().pkl','rb'))
            knn_model = pickle.load(open('https://github.com/isichei-nnamdi/bc-predictor/blob/main/streamlit_app/KNeighborsClassifier(n_neighbors%3D3).pkl','rb'))
            lr_model = pickle.load(open('https://github.com/isichei-nnamdi/bc-predictor/blob/main/streamlit_app/lr_model.pkl','rb'))
            svc_model = pickle.load(open('https://github.com/isichei-nnamdi/bc-predictor/blob/main/streamlit_app/svc_fit.pkl','rb'))
            
            options = st.selectbox(
                    'Select a trained model from the list to classify the type of breast cancer.',
                    ('Choose a model','Logistic Regression', 'Decision Tree Classifier', 'GaussianNB Classifier', 'KNeighbor Classifier', 'Support Vector Classifier'))
            
            if options == 'Choose a model':
                st.write(' ')
            
            elif options == 'Logistic Regression':
                prediction = pd.DataFrame(lr_model.predict(dataframe))
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(prediction.rename(columns={0: 'Diagnosis'}))
                with col2:
                    pred = pd.DataFrame(prediction.value_counts(), columns=['Frequency']).reset_index().rename(columns={0: 'Diagnosis'}) 
                    fig = px.bar(data_frame=pred,
                        x='Diagnosis',
                        y='Frequency',
                        color= 'Diagnosis',
                        title = 'Plot of Predicted Cancer Type')
                    st.plotly_chart(fig, use_container_width=True)
                confu_matrix = confusion_matrix(test_data, prediction)

                st.write(f'The classification was done using {options} classifier with an accuracy of {accuracy_score(test_data, prediction)*100:.2f}%. {confu_matrix[0][0]} of the {confu_matrix[0][0] + confu_matrix[0][1]} benign cases were correctly classified as benign, while {confu_matrix[1][1]} of the {confu_matrix[1][0] + confu_matrix[1][1]} malignant cases were accurately classified as malignant.')
            elif options == 'Decision Tree Classifier':
                prediction = pd.DataFrame(tree_model.predict(dataframe))
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(prediction.rename(columns={0: 'Diagnosis'}))
                with col2:
                    pred = pd.DataFrame(prediction.value_counts(), columns=['Frequency']).reset_index().rename(columns={0: 'Diagnosis'}) 
                    fig = px.bar(data_frame=pred,
                        x='Diagnosis',
                        y='Frequency',
                        color= 'Diagnosis',
                        title = 'Plot of Predicted Cancer Type')
                    st.plotly_chart(fig, use_container_width=True)
                confu_matrix = confusion_matrix(test_data, prediction)

                st.write(f'The classification was done using {options} classifier with an accuracy of {accuracy_score(test_data, prediction)*100:.2f}%. {confu_matrix[0][0]} of the {confu_matrix[0][0] + confu_matrix[0][1]} benign cases were correctly classified as benign, while {confu_matrix[1][1]} of the {confu_matrix[1][0] + confu_matrix[1][1]} malignant cases were accurately classified as malignant.')
            elif options == 'GaussianNB Classifier':
                prediction = pd.DataFrame(gauss_model.predict(dataframe))
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(prediction.rename(columns={0: 'Diagnosis'}))
                with col2:
                    pred = pd.DataFrame(prediction.value_counts(), columns=['Frequency']).reset_index().rename(columns={0: 'Diagnosis'}) 
                    fig = px.bar(data_frame=pred,
                        x='Diagnosis',
                        y='Frequency',
                        color= 'Diagnosis',
                        title = 'Plot of Predicted Cancer Type')
                    st.plotly_chart(fig, use_container_width=True)
                confu_matrix = confusion_matrix(test_data, prediction)

                st.write(f'The classification was done using {options} classifier with an accuracy of {accuracy_score(test_data, prediction)*100:.2f}%. {confu_matrix[0][0]} of the {confu_matrix[0][0] + confu_matrix[0][1]} benign cases were correctly classified as benign, while {confu_matrix[1][1]} of the {confu_matrix[1][0] + confu_matrix[1][1]} malignant cases were accurately classified as malignant.')
            elif options == 'KNeighbor Classifier':
                prediction = pd.DataFrame(knn_model.predict(dataframe))
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(prediction.rename(columns={0: 'Diagnosis'}))
                with col2:
                    pred = pd.DataFrame(prediction.value_counts(), columns=['Frequency']).reset_index().rename(columns={0: 'Diagnosis'}) 
                    fig = px.bar(data_frame=pred,
                        x='Diagnosis',
                        y='Frequency',
                        color= 'Diagnosis',
                        title = 'Plot of Predicted Cancer Type')
                    st.plotly_chart(fig, use_container_width=True)
                confu_matrix = confusion_matrix(test_data, prediction)

                st.write(f'The classification was done using {options} classifier with an accuracy of {accuracy_score(test_data, prediction)*100:.2f}%. {confu_matrix[0][0]} of the {confu_matrix[0][0] + confu_matrix[0][1]} benign cases were correctly classified as benign, while {confu_matrix[1][1]} of the {confu_matrix[1][0] + confu_matrix[1][1]} malignant cases were accurately classified as malignant.')
            else:
                prediction = pd.DataFrame(svc_model.predict(dataframe))
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(prediction.rename(columns={0: 'Diagnosis'}))
                with col2:
                    pred = pd.DataFrame(prediction.value_counts(), columns=['Frequency']).reset_index().rename(columns={0: 'Diagnosis'}) 
                    fig = px.bar(data_frame=pred,
                        x='Diagnosis',
                        y='Frequency',
                        color= 'Diagnosis',
                        title = 'Plot of Predicted Cancer Type')
                    st.plotly_chart(fig, use_container_width=True)
                confu_matrix = confusion_matrix(test_data, prediction)

                st.write(f'The classification was done using {options} classifier with an accuracy of {accuracy_score(test_data, prediction)*100:.2f}%. {confu_matrix[0][0]} of the {confu_matrix[0][0] + confu_matrix[0][1]} benign cases were correctly classified as benign, while {confu_matrix[1][1]} of the {confu_matrix[1][0] + confu_matrix[1][1]} malignant cases were accurately classified as malignant.')

if __name__ == '__main__':
    main()