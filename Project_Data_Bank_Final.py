#"""
#Title: "Predicting the Success of a Bank Marketing Campaign"  
#Author: Toobasadat Sarmadi  
#Date: 2024-2025  
#Description: Predicting term deposit subscriptions using data analysis and machine learning  
#Dependencies: pandas, numpy, seaborn, matplotlib, plotly, scikit-learn, imbalanced-learn, xgboost  
#Usage: Run this script in a Python environment with the required libraries installed.
#"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

from sklearn.metrics import classification_report

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("bank-additional-full.csv", sep=";")  # Correct usage inside a function

df = load_data()  # Call the function to get the cached data

# Page Title & Sidebar
st.title("Data Bank : binary classification project")
st.sidebar.title("Table of contents")
pages=["Introduction", "Exploration", "DataVizualization", "Modelling", "Conclusions"]
page=st.sidebar.radio("Go to", pages)





if page == pages[0]:
    #st.image('https://assets-datascientest.s3.eu-west-1.amazonaws.com/mc5_da_streamlit/ecom_image.png', width=350)
    st.header("Context")
    
    st.markdown("""
        Welcome to our **analysis of Predicting the Success of a Bank Marketing Campaign**. This interactive project explores the challenge of optimizing bank 
        marketing campaigns by predicting customer subscriptions to term deposits. 
        The outcomes of this analysis will directly enhance decision-making skills and the ability to apply data science techniques to 
        real-world business problems. This project focuses on predicting whether a customer will subscribe to a term deposit using machine 
        learning techniques. By identifying customers most likely to subscribe, this project helps optimize resource allocation in marketing 
        campaigns, improving efficiency and profitability.
        Use this dashboard to explore the data dynamically and gain actionable insights! 🚀
    """)

    st.header("Objectives")

    st.markdown("""
        The main objectives of the project consist of:
    •	Performing exploratory data analysis to understand key Parameters
    •	Identify key factors influencing customer decisions.
    •	Preparing the dataset for modeling by cleaning and preprocessing.
    •	Using machine learning models to predict customer subscription behavior

    """)


if page == pages[1] : 
    st.header("Data Exploration")
  #st.write("### Data Exploration")  
    st.markdown("""
        This section presents a clear overview of the dataset’s variables, including their definitions, data types, 
        and significance. Understanding these variables helps interpret trends, correlations, and patterns in the data. 
        A structured table or list ensures clarity and quick reference.The dataset used is the publicly available Bank Marketing Dataset.
        This dataset consists of thousands of records with key variables and the target variable y. The dataset also consists of customer 
        demographics, economic conditions, social and economic features, and marketing interaction details:

    """)
    if st.checkbox("Display Variables & Their Meaning"):
        variables = {
        "Variables Name": [ 'age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 
        'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 
        'cons.conf.idx', 'euribor3m', 'nr.employed', 'y'],
        "Meaning": [
        'Age of the client',
        'Type of job',
        'Marital status',
        'Level of education',
        'Has credit in default?',
        'Has housing loan?',
        'Has personal loan?',
        'Contact communication type',
        'Last contact month of the year',
        'Last contact day of the week',
        'Last contact duration (in seconds)',
        'Number of contacts performed during this campaign (includes last contact)',
        'Number of days since the client was last contacted from a previous campaign (-1 means client was not previously contacted)',
        'Number of contacts performed before this campaign for this client',
        'Outcome of the previous marketing campaign',
        'Employment variation rate (quarterly indicator)',
        'Consumer price index (monthly indicator)',
        'Consumer confidence index (monthly indicator)',
        'Euribor 3-month rate (daily indicator)',
        'Number of employees (quarterly indicator)',
        'Has the client subscribed to a term deposit?'],
         "Type of Variable": ['Quantitative', 'Categorical', 'Categorical', 'Categorical', 'Categorical-Binary', 'Categorical-Binary', 
         'Categorical-Binary', 'Categorical', 'Categorical', 'Categorical', 'Quantitative', 'Quantitative', 'Quantitative', 'Quantitative', 
         'Categorical', 'Quantitavie', 'Quantitative', 'Quantitative', 'Quantitative', 'Quantitative', 'Categorical-Binary']
    }
        st.dataframe(pd.DataFrame(variables))

    if st.checkbox("Display Dataset"):
        st.dataframe(df.head(10))

    st.header("Pre-Processing")
    st.markdown("""
        The dataset underwent the following preprocessing steps:

        •	Handling Missing Values: Imputed or removed missing data to ensure completeness.
        •	Encoding Categorical Variables: Used One-Hot Encoding and Label Encoding for categorical features.
        •	Feature Scaling: Applied StandardScaler for numerical variables.
    """)
    if st.checkbox("Show NA") :
        st.dataframe(df.isna().sum())




if page == pages[2] : 
    st.markdown("""
        ### Visualisation and Statistics
        In this section, we present three key data visualizations that provide insights into the project's core metrics. 
        These visualizations focus on a detailed visual analysis of the dataset to explore patterns, relationships, 
        and distributions of some variables. The goal of this analysis is to identify the factors that might influence 
        whether a customer subscribes to the term deposit product.
    """)
    col1, col2, col3 = st.columns(3
    )

    if col1.button("Descriptive Statistics of Quantitative Variables"):        
        st.write(df.shape)
        st.dataframe(df.describe())

    if col1.button("Distribution of Categorical Variables"):


        st.markdown("## Distribution of Categorical Variables")

# Create tabs
        tab1, tab2, tab3 = st.tabs(["Personal Info", "Financial Info", "Marketing Info"])

        with tab1:
            st.markdown("### Personal Information")
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            sns.countplot(x='y', data=df, ax=axes[0, 0])
            axes[0, 0].set_title("Deposit Outcome")

            sns.countplot(x='job', data=df, ax=axes[0, 1])
            axes[0, 1].set_title("Type of Job")
            axes[0, 1].tick_params(axis='x', rotation=45)

            sns.countplot(x='marital', data=df, ax=axes[1, 0])
            axes[1, 0].set_title("Marital Status")

            sns.countplot(x='education', data=df, ax=axes[1, 1])
            axes[1, 1].set_title("Education")
            axes[1, 1].tick_params(axis='x', rotation=45)
            #Correct Rotation and Alignment of X-axis labels
            for ax in axes.flat:  
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            #Adjust tables:
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("""
            The "Deposit Outcome" chart shows an imbalanced dataset where the majority class ("no") significantly outnumbers the minority class 
            ("yes"). In the "Type of Job" chart, the most common job category is "admin.," while the least common is "housemaid." Additionally, 
            in "Marital Status," "Married" is the most frequent status, whereas "unknown" is the least. In "Education," "University degree" 
            appears most often, while "illiterate" is the least represented.
                    """)

        with tab2:
            st.markdown("### Financial Information")
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            sns.countplot(x='default', data=df, ax=axes[0])
            axes[0].set_title("Credit Default")

            sns.countplot(x='housing', data=df, ax=axes[1])
            axes[1].set_title("Housing Loan")

            sns.countplot(x='loan', data=df, ax=axes[2])
            axes[2].set_title("Personal Loan")
            st.pyplot(fig)
            st.markdown("""
            The image contains three bar charts displaying data distributions. In the Credit Default and Personal Loan charts, the majority of 
            customers have no credit default and no personal loan. In contrast, the Housing Loan chart shows that more customers have a housing 
            loan (yes) than those without.
            """)

        with tab3:
            st.markdown("### Marketing Campaign Information")
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            sns.countplot(x='contact', data=df, ax=axes[0, 0])
            axes[0, 0].set_title("Contact Type")

            month_order = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
            sns.countplot(x='month', data=df, order=month_order, ax=axes[0, 1])
            axes[0, 1].set_title("Last Contact Month")

            sns.countplot(x='day_of_week', data=df, ax=axes[1, 0])
            axes[1, 0].set_title("Last Contact Day")

            sns.countplot(x='poutcome', data=df, ax=axes[1, 1])
            axes[1, 1].set_title("Previous Campaign Outcome")
            #Adjust tables:
            plt.tight_layout() 
            st.pyplot(fig)
            st.markdown("""
            The Contact Type chart indicates that 
            most interactions were conducted via cellular rather than telephone. The Last Contact Month chart reveals that the highest number 
            of contacts occurred in May, followed by June and July, while other months saw significantly lower activity. The Last Contact Day 
            chart shows a relatively even distribution of contacts across weekdays, suggesting no strong preference for any particular day. 
            Lastly, the Previous Campaign Outcome chart highlights that the majority of clients had no prior campaign interaction, with a 
            smaller proportion having experienced a failed or successful previous campaign.
            """)



    if col1.button("Distribution of Quantitative Variables"):        

    ##Categorical Vars:

        bins = [18,20, 25,30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]  
        labels = ["18-19","20-24", "25-29","30,34", "35-39","40-44", "45-49", "50,54", "55-59", "60-64" ,"65-69", "70-74", "+75"]
    # Create a new column for age groups
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)
        fig = plt.figure(figsize=(8, 4))  
    # Plot using age groups instead of individual ages
        sns.countplot(x="age_group", data=df, order=labels)
        plt.title("Distribution of Client's Age Groups")
        plt.xlabel("Age Group")
        plt.xticks(rotation=45, fontsize=12)  # Rotate labels for better readability
        st.pyplot(fig)
        st.markdown("""This graph illustrates the age distribution of customers targeted in the telemarketing campaign. The majority of 
        customers fall within the 30 to 34-year-old range, where the distribution peaks. In contrast, there are significantly fewer 
        customers over the age of 60 or under 20. This trend suggests that individuals aged 30 and above represent the primary demographic 
        for this campaign, likely due to their financial stability and higher potential interest in term deposit products.
                 """)

        fig, ax = plt.subplots(figsize=(10, 4)) 
        sns.histplot(df["duration"], bins=30, kde=True, ax=ax)  
        ax.set_title("Distribution of the Last Contact Duration (in Sec.)")
        ax.set_xlim(0, 1500) 
        ax.set_xlabel("Duration (seconds)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        st.markdown("""This histogram shows the distribution of the last contact duration (in seconds) during the telemarketing campaign. 
        The distribution is right-skewed, with most calls lasting under 200 seconds. The frequency decreases as the duration increases, 
        indicating that longer calls are less common. The presence of a density curve further emphasizes the steep decline in call frequency 
        as duration increases.
                """)


##########Some visualisations:
    if col1.button("Relationship between Important Variables"):        

     #   fig = plt.figure()
      #  sns.countplot(x = 'y', hue='marital', data = df)
       # st.pyplot(fig)



        marital_statuses = df['marital'].unique()
        fig = go.Figure()
# Add traces for each marital category
        for status in marital_statuses:
            filtered_df = df[df['marital'] == status]
            counts = filtered_df['y'].value_counts()
    
            fig.add_trace(go.Bar(
            x=counts.index,  # 'yes' or 'no'
            y=counts.values,  # Count of occurrences
            name=status
            ))
        fig.update_layout(
        title='Distribution of Y by Marital Status',
        xaxis_title='Y (Outcome)',
        yaxis_title='Count',
        barmode='group'  # 'group' for side-by-side bars, 'stack' for stacked bars
        )
        st.plotly_chart(fig)
        st.markdown("""This bar chart displays the distribution of the target variable Y  across different marital statuses. 
        The majority of customers in the dataset are married, followed by single and divorced individuals. Most customers, 
        regardless of marital status, did not subscribe to the term deposit. However, among those who did, married individuals 
        still represent the largest group. The "unknown" category is minimal in comparison. 
                """)

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['y'], name='y', opacity=0.4))
        fig.add_trace(go.Histogram(x=df['housing'], name='housing', opacity=0.6))

        fig.update_layout(
        title='Distribution of Housing and Y',
        xaxis_title='Value',
        yaxis_title='Count',
        barmode='overlay'
        )
        st.plotly_chart(fig)
        st.markdown("""illustrates the distribution of housing loan ownership (housing) in relation to the term deposit subscription status (y). 
            It categorizes customers into three groups: those without a housing loan ("no"), those with a housing loan ("yes"), and those with an 
            unknown status. The majority of customers fall into the "no" category, which includes a significant number of both subscribers (y=yes) 
            and non-subscribers (y=no). Customers with a housing loan ("yes") also contribute notably to the dataset, though the proportion of 
            subscribers within this group appears slightly smaller than non-subscribers. The "unknown" category has minimal representation, 
            suggesting that housing loan data is well-documented for most customers. This analysis highlights potential trends in how housing 
            loan status may influence the likelihood of subscribing to a term deposit.
                    """)

        fig=go.Figure()
        fig.add_trace(go.Scatter(x = df.age,
                         y = df.duration,
                        name = 'does last contact take more based on age?',
                         text=df.marital,
                        mode='markers'))
        fig.update_layout(title='Relationship Between Age and Last Contact Duration', 
                   xaxis_title='age',   
                   yaxis_title='last contact duration')        
        st.plotly_chart(fig)
        st.markdown("""examines whether the 'Last Contact Duration' is influenced by age, using a scatter plot to represent the relationshin
            between customer age and the duration of the last contact. The graph suggests that call durations tend to be longer for individuals 
            under 60 years old, while those aged 60 and above are less likely to engage in long conversations.
                    """)

    ####Heatmap
    #### y is binary: yes/no ---> 0/1:
        df['y_num'] = df['y'].map({'no': 0, 'yes': 1})
    #### and other binary Vars:
        df['default_num'] = df['default'].map({'no': 0, 'yes': 1})
        df['housing_num'] = df['housing'].map({'no': 0, 'yes': 1})
        df['loan_num'] = df['loan'].map({'no': 0, 'yes': 1})

        df_numeric = df[['y_num', 'default_num', 'housing_num', 'loan_num', 'age','duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
        'cons.price.idx', 'cons.conf.idx', 'euribor3m','nr.employed']]
        fig, ax = plt.subplots()
        sns.heatmap(df_numeric.corr(), annot=True, annot_kws={"size": 6},fmt=".2f", cmap="crest", ax=ax)
        st.pyplot(fig) 
        st.markdown("""This heatmap illustrates the correlation between different variables in the dataset. The 'duration' variable shows 
        the highest positive correlation with 'y' (0.41), indicating that longer call durations are strongly associated with a higher 
        likelihood of a positive outcome. 'pdays' (-0.32), 'euribor3m' (-0.31), and 'nr.employed' (-0.35) have also the strongest (negative) 
        correlations with 'y', indicating that a higher number of days since last contact, higher euribor rates, and a larger number of 
        employed individuals are associated with a higher likelihood of subscription.
                """)



        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # First subplot
        sns.barplot(x='y', y='duration', hue='marital', data=df, ci=None, ax=axes[0])
        axes[0].set_title("Duration by Marital Status")
        axes[0].legend(fontsize='small')
    # Second subplot
        sns.barplot(x='y', y='duration', hue='education', data=df, ci=None, ax=axes[1])
        axes[1].set_title("Duration by Education Level")
        axes[1].legend(fontsize='small')
        st.pyplot(fig)
        st.markdown("""These graphs illustrate that among those who have subscribed to the campaign, individuals with lower education levels 
            tend to have longer last call durations. However, for those who have not subscribed, there are no significant differences, even when 
            considering different marital statuses. It also appears that single individuals are less likely to engage in these calls compared to 
            others.
                    """)
 

if page == pages[3] : 
    st.write("## Modelling")

    #separate data to featurs and target:
    feats = df.drop('y', axis=1)
    target = df['y']

    #split the data to train and test:
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state = 42)
    #There is no missing values so there is no need to use simpleimputer

    #######Preprocessing:  Standardize the numerical variables:
    sc=StandardScaler()
    cols = ['age', 'duration', 'campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed',]
    X_train[cols] = sc.fit_transform(X_train[cols])
    X_test[cols] = sc.transform(X_test[cols])

    ######Encoding categorical target var:
    le=LabelEncoder()

    y_train=le.fit_transform(y_train)

    y_test=le.transform(y_test)



    ######Encoding explanatory Variables:

    # The drop = 'first' parameter allows to delete one of the columns created by the OneHotEncoder and thus avoid a multicolinearity problem
    #in new version: sparse--->sparse_output
    oneh = OneHotEncoder(drop = 'first', sparse_output=False)

    cols= ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']


    # Fit and transform on the training data, and transform on the test data
    X_train_encoded = oneh.fit_transform(X_train[cols])
    X_test_encoded = oneh.transform(X_test[cols])

    # Convert the encoded arrays into DataFrames
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=oneh.get_feature_names_out(cols), index=X_train.index)
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=oneh.get_feature_names_out(cols), index=X_test.index)

    # Drop the original categorical columns from X_train and X_test
    X_train = X_train.drop(columns=cols)
    X_test = X_test.drop(columns=cols)

    # Concatenate the encoded DataFrame with the original DataFrame
    X_train = pd.concat([X_train, X_train_encoded_df], axis=1)
    X_test = pd.concat([X_test, X_test_encoded_df], axis=1)


    #################################################################



    # Model selection
    choice = ['-', 'Logistic Regression', 'DecisionTreeClassifier', 'Random Forest', 'XGBOOST']
    option = st.selectbox('Choice of the model', choice)

    st.write('The chosen model is:', option)

    # Define a function to train and evaluate the selected model
    def train_and_evaluate(model, model_name):
        # Train the model
        model.fit(X_train, y_train)

        # Predictions
        predictions = model.predict(X_test)

        # Streamlit Output
        st.write(f"### {model_name} Results")

        # Print scores on training and test data
        st.write("### Score")
        st.write(f"**Score on the train set:** {model.score(X_train, y_train):.2f}")
        st.write(f"**Score on the test set:** {model.score(X_test, y_test):.2f}")

        # Display Confusion Matrix
        st.write("### Confusion Matrix")
        st.dataframe(pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Prediction']))

        # Convert classification report to a DataFrame
        report_dict = classification_report(y_test, predictions, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()

        # Display Classification Report as a table
        st.write("### Classification Report")
        st.dataframe(report_df)



    # Check which model was selected and call the function

    if option == 'Logistic Regression':
        train_and_evaluate(LogisticRegression(random_state=42), "Logistic Regression")

    elif option == 'DecisionTreeClassifier':
        train_and_evaluate(DecisionTreeClassifier(max_depth=3, min_samples_leaf=25, random_state=42), "Decision Tree Classifier")

    elif option == 'Random Forest':
        train_and_evaluate(RandomForestClassifier(random_state=42), "Random Forest")

    elif option == 'XGBOOST':
        train_and_evaluate(XGBClassifier(random_state=42), "XGBOOST")
    elif option == '-':
        st.write('-')

    st.write("## Precision for Class 1")

    # Data
    data1 = {
        "Model": [
            "Logistic Regression", 
            "DecisionTreeClassifier",
            "Random Forest",
            "XGBOOST",
        ],
        "Precision Class-1": [0.6694, 0.6004, 0.6461, 0.6362]
    }

    # Convert to DataFrame
    df = pd.DataFrame(data1)

    # Format Precision column to 3 decimal places
    df["Precision Class-1"] = df["Precision Class-1"].map(lambda x: f"{x:.3f}")

    # Function to apply highlighting to a specific row
    def highlight_row(s):
        return ['background-color: gray' if s["Model"] == "DecisionTreeClassifier" else '' for _ in s]

    # Display table with color highlight
    if st.checkbox("Precision Class-1"):
        st.dataframe(df.style.apply(highlight_row, axis=1))
        
            

    
############ Imbalance ################
    st.write('## Fixing Imbalance')
    choice = ['-', 'Random Over Sampler-DecisionTreeClassifier', 'Random Under Sampler-DecisionTreeClassifier', 'Mix Random Over & Under Sampler-DecisionTreeClassifier', 
    'Random Over Sampler-XGBOOST', 'Random Under Sampler-XGBOOST', 'Mix Over- & Under- Sampler_XGBOOST-scale_pos_weight-max_delta_step']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is:', option)

    if option == '-':
        st.write('-')
    if option == 'Random Over Sampler-DecisionTreeClassifier':
        # Apply RandomOverSampler
        st.write("## Random Over Sampler-DecisionTreeClassifier")
        rOs = RandomOverSampler(random_state=42)
        X_ro, y_ro = rOs.fit_resample(X_train, y_train)

        # Display class distribution after oversampling
        oversampled_distribution = pd.Series(y_ro).value_counts(normalize=True)
        st.write("Oversampled sample classes:")
        st.dataframe(oversampled_distribution)

        # Train DecisionTreeClassifier Model on Resampled Data
        st.write("### Training DecisionTreeClassifier on Resampled Data")
        dtc = DecisionTreeClassifier(max_depth=3, min_samples_leaf=25, random_state=42)
        dtc.fit(X_ro, y_ro)

        # Display scores
        st.write(f"**Score on train set:** {dtc.score(X_ro, y_ro):.2f}")
        st.write(f"**Score on test set:** {dtc.score(X_test, y_test):.2f}")

        # Predictions
        pred_ro = dtc.predict(X_test)

        # Display Confusion Matrix
        st.write("### Confusion Matrix")
        st.dataframe(pd.crosstab(y_test, pred_ro, rownames=['True'], colnames=['Prediction']))

        # Display Classification Report
        st.write("### Classification Report")
        report_dict = classification_report(y_test, pred_ro, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df)



    if option == 'Random Under Sampler-DecisionTreeClassifier':
        # Apply RandomOverSampler
        st.write("## Random Under Sampler-DecisionTreeClassifier")
        rUs = RandomUnderSampler(random_state=42)
        X_ru, y_ru = rUs.fit_resample(X_train, y_train)

        # Display class distribution after oversampling
        undersampled_distribution = pd.Series(y_ru).value_counts(normalize=True)
        st.write("Undersampled sample classes:")
        st.dataframe(undersampled_distribution)

        # Train DecisionTreeClassifier Model on Resampled Data
        st.write("### Training DecisionTreeClassifier on Resampled Data")
        dtc = DecisionTreeClassifier(max_depth=3, min_samples_leaf=25, random_state=42)
        dtc.fit(X_ru, y_ru)

        # Display scores
        st.write(f"**Score on train set:** {dtc.score(X_ru, y_ru):.2f}")
        st.write(f"**Score on test set:** {dtc.score(X_test, y_test):.2f}")

        # Predictions
        pred_ru = dtc.predict(X_test)

        # Display Confusion Matrix
        st.write("### Confusion Matrix")
        st.dataframe(pd.crosstab(y_test, pred_ru, rownames=['True'], colnames=['Prediction']))

        # Display Classification Report
        st.write("### Classification Report")
        report_dict = classification_report(y_test, pred_ru, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df)



    if option == 'Random Over Sampler-XGBOOST':
        # Apply RandomOverSampler
        st.write("## Random Over Sampler-XGBOOST")
        rOs = RandomOverSampler(random_state=42)
        X_ro, y_ro = rOs.fit_resample(X_train, y_train)

        # Display class distribution after oversampling
        oversampled_distribution = pd.Series(y_ro).value_counts(normalize=True)
        st.write("oversampled sample classes:")
        st.dataframe(oversampled_distribution)

        # Train XGBOOST Model on Resampled Data
        st.write("### Training XGBOOST on Resampled Data")
        bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
        bst.fit(X_ro, y_ro)

        # Display scores
        st.write(f"**Score on train set:** {bst.score(X_ro, y_ro):.2f}")
        st.write(f"**Score on test set:** {bst.score(X_test, y_test):.2f}")

        # Predictions
        pred_ro = bst.predict(X_test)

        # Display Confusion Matrix
        st.write("### Confusion Matrix")
        st.dataframe(pd.crosstab(y_test, pred_ro, rownames=['True'], colnames=['Prediction']))

        # Display Classification Report
        st.write("### Classification Report")
        report_dict = classification_report(y_test, pred_ro, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df)




    if option == 'Random Under Sampler-XGBOOST':
        # Apply RandomUnderSampler
        st.write("## Random Under Sampler-XGBOOST")
        rUs = RandomUnderSampler(random_state=42)
        X_ru, y_ru = rUs.fit_resample(X_train, y_train)

        # Display class distribution after oversampling
        undersampled_distribution = pd.Series(y_ru).value_counts(normalize=True)
        st.write("## undersampled sample classes:")
        st.dataframe(undersampled_distribution)

        # Train Logistic Regression Model on Resampled Data
        st.write("### Training XGBOOST on Resampled Data")
        bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
        bst.fit(X_ru, y_ru)

        # Display scores
        st.write(f"**Score on train set:** {bst.score(X_ru, y_ru):.2f}")
        st.write(f"**Score on test set:** {bst.score(X_test, y_test):.2f}")

        # Predictions
        pred_bst = bst.predict(X_test)

        # Display Confusion Matrix
        st.write("### Confusion Matrix")
        st.dataframe(pd.crosstab(y_test, pred_bst, rownames=['True'], colnames=['Prediction']))

        # Display Classification Report
        st.write("### Classification Report")
        report_dict = classification_report(y_test, pred_bst, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df)



    if option == 'Mix Random Over & Under Sampler-DecisionTreeClassifier':
        # Apply Mix RandomSampler
        st.write("## Mix Over & Under Sampler - DecisionTreeClassifier")

        # Apply Random Over Sampler
        st.write("### Step 1: Oversampling the Minority Class")
        oversampler = RandomOverSampler(random_state=42)
        X_oversampled, y_oversampled = oversampler.fit_resample(X_train, y_train)

        # Display class distribution after oversampling
        oversampled_distribution = pd.Series(y_oversampled).value_counts(normalize=True)
        st.write("## Oversampled Sample Classes:")
        st.dataframe(oversampled_distribution.to_frame(name="Proportion"))

        # Apply Random Under Sampler
        st.write("### Step 2: Undersampling the Majority Class")
        undersampler = RandomUnderSampler()
        X_resampled, y_resampled = undersampler.fit_resample(X_oversampled, y_oversampled)

        # Display class distribution after undersampling
        undersampled_distribution = pd.Series(y_resampled).value_counts(normalize=True)
        st.write("Undersampled Sample Classes:")
        st.dataframe(undersampled_distribution.to_frame(name="Proportion"))

        # Train DecisionTreeClassifier Model on Resampled Data
        st.write("### Training DecisionTreeClassifier on Resampled Data")
        dtc = DecisionTreeClassifier(max_depth=3, min_samples_leaf=25, random_state=42)
        dtc.fit(X_resampled, y_resampled)

        # Display scores
        st.write(f"**Score on train set (mixed resampling):** {dtc.score(X_resampled, y_resampled):.2f}")
        st.write(f"**Score on test set (mixed resampling):** {dtc.score(X_test, y_test):.2f}")

        # Predictions
        pred_resampled = dtc.predict(X_test)

        # Display Confusion Matrix
        st.write("### Confusion Matrix")
        st.dataframe(pd.crosstab(y_test, pred_resampled, rownames=['True'], colnames=['Prediction']))

        # Display Classification Report
        st.write("### Classification Report")
        report_dict = classification_report(y_test, pred_resampled, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df)


    if option == 'Mix Over- & Under- Sampler_XGBOOST-scale_pos_weight-max_delta_step':
        st.write("## Mix Over- & Under- Sampler with XGBOOST")

        # Step 1: Oversampling the Minority Class
        st.write("### Step 1: Oversampling the Minority Class")
        oversampler = RandomOverSampler(random_state=42)
        X_oversampled, y_oversampled = oversampler.fit_resample(X_train, y_train)

        # Display class distribution after oversampling
        oversampled_distribution = pd.Series(y_oversampled).value_counts(normalize=True)
        st.write("## Oversampled Sample Classes:")
        st.dataframe(oversampled_distribution.to_frame(name="Proportion"))

        # Step 2: Undersampling the Majority Class
        st.write("### Step 2: Undersampling the Majority Class")
        undersampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = undersampler.fit_resample(X_oversampled, y_oversampled)

        # Display class distribution after undersampling
        undersampled_distribution = pd.Series(y_resampled).value_counts(normalize=True)
        st.write("Undersampled Sample Classes:")
        st.dataframe(undersampled_distribution.to_frame(name="Proportion"))

        # Compute the positive class weight
        pos_class_weight = (len(y_resampled) - np.sum(y_resampled)) / np.sum(y_resampled)
        st.write(f"**Computed scale_pos_weight:** {pos_class_weight:.2f}")

        # Initialize and Train XGBoost Model
        st.write("### Training XGBOOST Model with scale_pos_weight and max_delta_step")
        model = XGBClassifier(
            n_estimators=100,
            objective='binary:logistic',
            scale_pos_weight=pos_class_weight,
            max_delta_step=1,
            random_state=42
        )
        model.fit(X_resampled, y_resampled)

        # Predictions
        predictions = model.predict(X_test)

        # Display Confusion Matrix
        st.write("### Confusion Matrix")
        st.dataframe(pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Prediction']))

        # Display Classification Report
        st.write("### Classification Report")
        report_dict = classification_report(y_test, predictions, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df)


    st.write("## Summary of Precision for Class 1")

    # Data
    data = {
        "Model": [
            "Logistic Regression", 
            "DecisionTreeClassifier",
            "Random Forest",
            "XGBOOST",
            "Random Over Sampler - DecisionTreeClassifier",
            "Random Under Sampler - DecisionTreeClassifier",
            "Mix Random Over & Under Sampler - DecisionTreeClassifier",
            "Random Over Sampler - XGBOOST",
            "Random Under Sampler - XGBOOST",
            "Mix Random Over & Under Sampler - XGBOOST - Scale"
        ],
        "Precision Class 1": [0.6694, 0.6004, 0.6576, 0.6362, 0.3989, 0.4055, 0.3989, 0.3541, 0.376, 0.4805]
    }

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Format Precision column to 3 decimal places
    df["Precision Class 1"] = df["Precision Class 1"].map(lambda x: f"{x:.3f}")

    # Function to apply highlighting to a specific row
    def highlight_row(s):
        return ['background-color: gray' if s["Model"] == "Random Over Sampler - XGBOOST" else '' for _ in s]

    # Display table with color highlight
    if st.checkbox("Precision Class 1"):
        st.dataframe(df.style.apply(highlight_row, axis=1))


##################### Feature Importance ############
    # Apply Random Over Sampler
    rOs = RandomOverSampler(random_state=42)
    X_ro, y_ro = rUs.fit_resample(X_train, y_train)

    # Train XGBOOST Model on Resampled Data
    bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic', random_state=42)
    bst.fit(X_ro, y_ro)

    # Feature Importance
    feat_importance = pd.DataFrame({
        'variables': bst.feature_names_in_,
        'importance': bst.feature_importances_
    }).sort_values(by='importance', ascending=False)

    # Plotting Feature Importance
    st.write("## Feature Importance Plot")
    fig, ax = plt.subplots(figsize=(11, 5))
    feat_importance.nlargest(4, 'importance').plot.bar(x='variables', y='importance', ax=ax, color='#4529de')

    if st.checkbox("Show Feature Importance Plot"):
        st.pyplot(fig)

    # Interpretation
    if st.checkbox("Show Interpretation"):
        st.write("""
    The duration variable is the most important feature in the model, followed closely by the number of employees (nr.employed) and the 
    employment variation rate (emp.var.rate). This indicates that both client interaction details and macroeconomic indicators significantly 
    impact the likelihood of a client subscribing to a term deposit. Meanwhile, the age variable appears to have little to no influence on 
    the model's predictions, suggesting that demographic factors like age may not be strong predictors compared to other economic and 
    behavioral indicators.
        """)





if page == pages[4]:
    st.header("Results")
    
    st.markdown("""
    This study demonstrates the effectiveness of machine learning in predicting customer behavior in banking campaigns. The insights gained 
    can be directly applied to enhance targeting strategies, optimize marketing budgets, and improve customer engagement.

    Among the models tested, the XGBOOST combined with a Random Over Sampling outperformed others in precision, 
    making it the most effective predictor of term deposit subscriptions.

    Key findings include:

    Duration of the last contact was the strongest predictor of subscription.
    Higher contact frequency increased the likelihood of subscription.
    Customer employment status played a significant role in decision-making.
    """)

    st.header("Difficulties Encounter during the Project")

    st.markdown("""
    One of the most challenging aspects of this project was handling imbalanced data. The dataset contained significantly more 'no' than 'yes' 
    responses, requiring techniques such as oversampling, undersampling, and a combination of both. These methods, particularly the mixed 
    approach, improved the model's performance as anticipated.
    """)


    st.header("Continuation of the Project:")

    st.markdown("""
    For future work, it is recommended to explore the use of XGBoost with class weighting instead of using oversampling or undersampling. 
    Changing the class weights might help the model handle the minority class better without adding any bias from resampling, which could 
    lead to better performance.
    """)






