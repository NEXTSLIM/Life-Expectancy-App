import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import os # accessing directory structure
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plotCorrelationMatrix

from sklearn import datasets, ensemble

from bokeh.plotting import figure, show
from bokeh.io import output_notebook

#-------------
#theme 
#--

primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"

#-------------
#layout design 
#-------------

st.title('LIFE EXPECTANCY ESTIMATOR TOOL')
st.markdown("---")
st.write('''
         This app will estimate life expectancy base on 
World Development Indicators| Data from World Bank Open Datandata.
         
Please fill in the attributes below, then hit the life expectancy Estimate buttonto get the estimate. 
''')
st.markdown("---")

#---------------------------------#
# Model building
st.header('Input Attributes')
att_electric = st.slider('Access to electricity (% of population)', min_value=0, max_value=100, value= 80, step=10)
att_ndiseases = st.slider('Cause of death, by non-communicable diseases (% of total)', min_value= 16, max_value= 96, value=69, step=10)
att_healthexp = st.slider('Current health expenditure (% of GDP))', min_value= 1, max_value= 24, value=6, step=2)
att_diabetes = st.slider('Diabetes prevalence (% of population ages 20 to 79)', min_value= 0, max_value= 30, value=8, step=5)
att_eduexp = st.slider('Government expenditure on education, total (% of GDP)', min_value= 0, max_value= 14, value=4, step=2)

att_hospbeds = st.slider('Hospital beds (per 1,000 people))', min_value= 1, max_value= 16, value=3, step=1)
att_hepB3 = st.slider('Immunization, HepB3 (% of one-year-old children)', min_value= 1, max_value= 99, value=85, step=10)
att_measles = st.slider('Immunization, measles (% of children ages 12-23 months))', min_value= 8, max_value= 99, value=86, step=10)
att_inflation = st.slider('Inflation, consumer prices (annual %)', min_value= -18, max_value= 513, value=7, step=4)

att_cellular = st.slider('Mobile cellular subscriptions (per 100 people)', min_value= 0, max_value= 345, value=72, step=10)
att_traffic = st.slider('Mortality caused by road traffic injury (per 100,000 population)', min_value= 0, max_value= 65, value=17, step=5)
att_sanitation = st.slider('Mortality rate attributed to unsafe water, unsafe sanitation and lack of hygiene (per 100,000 population)', min_value= 0.1, max_value= 101e2, value=12e2, step=5e2)

att_infant= st.number_input('Number of infant deaths(world average 390k)', min_value=1000, max_value=2000000, value=10000) 
att_population= st.number_input('Population, total(billions)', min_value=3214, max_value=1397715000, value=31432454) 

att_below_income = st.slider('Proportion of people living below 50 percent of median income (%)', min_value= 1, max_value= 31, value=13, step=5)
att_interest_rate = st.slider('Real interest rate (%)', min_value= -74, max_value= 94, value=6, step=2)
att_suicide = st.slider('Suicide mortality rate (per 100,000 population)', min_value= 0, max_value= 93, value=10, step=5)
att_alcohol = st.slider('Total alcohol consumption per capita (liters of pure alcohol, projected estimates, 15+ years of age)', min_value= 0, max_value= 20, value=10, step=2)
att_unemployment = st.slider('Unemployment, total (% of total labor force) (modeled ILO estimate)', min_value= 1, max_value= 36, value=8, step=2)

att_regn = st.selectbox('Region', options=(1,2,3,4,5,6,7))
st.write('''
         * 1: East Asia & Pacific
         * 2: Europe & Central Asia
         * 3: Latin America & Caribbean
         * 4: Middle East & North Africa
         * 5: North America
         * 6: South Asia
         * 7: Sub-Saharan Africa
         '''
         )

if att_regn == 1:
    att_regn_1 = 1
    att_regn_2 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = 0
elif att_regn == 2: 
    att_regn_2 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = 0
elif att_regn == 3: 
    att_regn_3 = 1
    att_regn_1 = att_regn_2 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 =  0
elif att_regn == 4: 
    att_regn_4 = 1
    att_regn_1 = att_regn_3 = att_regn_2 = att_regn_5 = att_regn_6 = att_regn_7 =  0
elif att_regn == 5: 
    att_regn_5 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_2 = att_regn_6 = att_regn_7 =  0
elif att_regn == 6: 
    att_regn_6 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_2 = att_regn_7 = 0
else:
    att_regn_7 = 1
    att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_2  = 0


user_input = np.array([att_electric , att_ndiseases, att_healthexp, att_diabetes , att_eduexp , 
                       att_hospbeds , att_hepB3, att_measles , att_inflation , att_cellular , att_traffic , 
                       att_sanitation , att_infant, att_population, att_below_income , att_interest_rate , 
                       att_suicide , att_alcohol , att_unemployment, att_regn_1, att_regn_2, att_regn_3,
                       att_regn_4, att_regn_5, att_regn_6, att_regn_7, 
                       ]).reshape(1,-1)


    # AN Sidebar - Specify parameter settings
with st.sidebar.header('Set Parameters'):
        split_size = st.sidebar.slider('Data split ratio (percentage for Training Set)', min_value=10, max_value=90, value= 20, step=10) #use
        learning_rate = st.sidebar.select_slider('Learning rate (trade-off with n_estimators)', options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]) #done
        parameter_n_estimators = st.sidebar.slider('Number of estimators (number of trees)', 100, 500, 1000) #done
        parameter_max_depth = st.sidebar.slider('Max depth (maximum number of levels in each trees)', min_value=1, max_value=9, value= 3, step= 1) #done
        parameter_max_features = st.sidebar.select_slider('Max features (Max number of features to consider at each split)', options=['auto', 'sqrt' , 'log2']) #done
        parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', min_value=1, max_value=10, value= 2, step= 1) #done
        parameter_min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', min_value=1, max_value=8, value= 1, step= 1) #done
        #AN addition
        parameter_max_leaf_node = st.sidebar.slider('Grow trees with max_leaf_nodes in best-first fashion', min_value=8, max_value=32, value= 8, step= 2) #done

        parameter_subsample = st.sidebar.select_slider('Subsample (percentage of samples per tree) ', options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]) #done
        parameter_random_state = st.sidebar.slider('random_state (Controls the random seed given to each Tree estimator at each boosting iteration)',  min_value=0, max_value=100, value=100, step= 10) #done
        parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['friedman_mse', 'mse','mae']) #done

#------
# Model
#------

#import dataset
def get_dataset():
    data= pd.read_csv('https://renzo-test1.s3.amazonaws.com/life_expectancy/ML_data_SL.csv')
    # data = pd.read_csv('ML_data_SL.csv')
    return data

life_df = get_dataset()
df = life_df.copy()
st.markdown("---")

if st.button('Estimate LIFE EXPECTANCY'):
    data = get_dataset()
    
    #fix column names
    data.columns = (["countryname","countrycode","year","region","life_expectancy","access_electricity",
    "non_communicable_diseases","health_expenditure","diabetes","education_expenditure","hospital_beds",
    "hepb3","measles","inflation", "cellular_subscriptions","road_mortality" ,"sanitation","infant_deaths","population",
    "below_median_income","interest_rate","suicide_rate","alcohol_consumption","unemployment"])
    
    #Fix data types
    data.countryname = data.countryname.astype('category')
    data.countrycode = data.countrycode.astype('category')
    data.year = data.year.astype('category')
    data.region = data.region.astype('category')
    data.life_expectancy = data.life_expectancy.astype(int)
    data.access_electricity = data.access_electricity.astype(float)
    data.non_communicable_diseases = data.non_communicable_diseases.astype(int)
    data.health_expenditure = data.health_expenditure.astype(float)
    data.diabetes = data.diabetes.astype(float)
    data.education_expenditure = data.education_expenditure.astype(float)
    data.hospital_beds = data.hospital_beds.astype(float)
    data.hepb3 = data.hepb3.astype(int)
    data.measles = data.measles.astype(int)
    data.inflation = data.inflation.astype(int)
    data.cellular_subscriptions = data.cellular_subscriptions.astype(int)
    data.road_mortality = data.road_mortality.astype(float)
    data.sanitation = data.sanitation.astype(float)
    data.infant_deaths = data.infant_deaths.astype(int)
    data.population = data.population.astype(int)
    data.below_median_income = data.below_median_income.astype(float)
    data.interest_rate = data.interest_rate.astype(float)
    data.suicide_rate = data.suicide_rate.astype(float)
    data.alcohol_consumption = data.alcohol_consumption.astype(float)
    data.unemployment = data.unemployment.astype(float)

    
    #Region Transform
    data_final = pd.concat([data,pd.get_dummies(data['region'], prefix='region')], axis=1).drop(['region'],axis=1)
    
    #Data Split
    y = data_final['life_expectancy']
    X = data_final.drop(['life_expectancy','countryname','countrycode', 'year'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-split_size)/100, random_state=parameter_random_state)

    gbm_opt = GradientBoostingRegressor(
        learning_rate =learning_rate,#use
        n_estimators=parameter_n_estimators, #use
        random_state=parameter_random_state, #use
        max_depth=parameter_max_depth, #use
        max_features=parameter_max_features, #use
        subsample= parameter_subsample, #use
        criterion=parameter_criterion, #use
        min_samples_split=parameter_min_samples_split, #use
        min_samples_leaf=parameter_min_samples_leaf,#use
        #AN Additional
        max_leaf_nodes=parameter_max_leaf_node)
    
    #model training

    # gbm_opt = GradientBoostingRegressor(learning_rate=0.01, n_estimators=500,
    #                                         max_depth=5, min_samples_split=10, 
    #                                         min_samples_leaf=1, subsample=0.7,
    #                                         max_features= 18, random_state=101, criterion='friedman_mse')
    gbm_opt.fit(X_train,y_train)

    ##AN - New
    st.markdown('**Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**Variable details**:')
    st.write('X variable - Attributes')
    st.info(list(X.columns))
    st.write('Y variable - Prediction')
    st.info(y.name)
    ##AN 
    
    #making a prediction
    gbm_predictions = gbm_opt.predict(user_input) #user_input is taken from input attributes 
    gbm_score = gbm_opt.score(X_test,y_test) #R2 of the prediction from user input
    gbm_mse = mean_squared_error(y_test, gbm_opt.predict(X_test))
    gbm_rmse = gbm_mse**(1/2)

    gbm_mse_train = mean_squared_error(y_train, gbm_opt.predict(X_train))
    gbm_rmse_train = gbm_mse_train**(1/2)

    st.write('Based on the user input the estimated Life Expectancy for this region is: ')
    st.info((gbm_predictions))

    st.subheader('Model Performance')

    st.write('With an ($R^2$) score of: ', gbm_score)
    
    st.write('Error (MSE or MAE) for testing:')
    st.info(gbm_mse)
    st.write("The root mean squared error (RMSE) on test set: {:.4f}".format(gbm_rmse))

    st.write('Error (MSE or MAE) for training:')
    st.info(gbm_mse_train)
    st.write("The root mean squared error (RMSE) on train set: {:.4f}".format(gbm_rmse_train))


    st.subheader('Model Parameters')
    st.write(gbm_opt.get_params())
    


# # Graphing Function #####
st.markdown("---")
z_data = pd.read_csv('https://renzo-test1.s3.amazonaws.com/life_expectancy/ML_data_SL.csv')

z = z_data.values
sh_0, sh_1 = z.shape
x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
fig.update_layout(title='3D DATA VISUALITATION', autosize=False,
                  width=800, height=800,
                  margin=dict(l=40, r=40, b=40, t=40))
st.plotly_chart(fig)


# display data
st.markdown("---")
with st.beta_container():
    show_data = st.checkbox("See the raw data?")

    if show_data:
        df
st.markdown("---")
# Life Expectancy Data.csv has 2939 rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('https://renzo-test1.s3.amazonaws.com/life_expectancy/ML_data_SL.csv')
df1.dataframeName = 'https://renzo-test1.s3.amazonaws.com/life_expectancy/ML_data_SL.csv'
nRow, nCol = df1.shape
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()

#Building Correlation Matrix Model for data
st.subheader('Correlation between features')
fig3 = plt.figure()
sns.heatmap(df1.corr())
st.pyplot(fig3)