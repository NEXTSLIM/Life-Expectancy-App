import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

#-------------
#layout design 
#-------------

st.title('LIFE EXPENCTACY Estimatior Tool')
st.write('''
         This app will estimate life expectancy for a countr, given some 
         indicator for that specific country as input.
         
         Please fill in the attributes below, then hit the life expectancy Estimate button
         to get the estimate. 
         ''')

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

#------
# Model
#------

#import dataset
def get_dataset():
    data = pd.read_csv('ML_data_SL.csv')
    return data

if st.button('Estimate LIFE EXPENCTACY'):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    
    #model training
    gbm_opt = GradientBoostingRegressor(learning_rate=0.01, n_estimators=500,
                                        max_depth=5, min_samples_split=10, 
                                        min_samples_leaf=1, subsample=0.7,
                                        max_features=7, random_state=101)
    gbm_opt.fit(X_train,y_train)
    
    #making a prediction
    gbm_predictions = gbm_opt.predict(user_input) #user_input is taken from input attrebutes
    gmb_score = gbm_opt.score(X,y) 
    st.write('The LIFE EXPENDACASY BASE ON YOUR IMPUTS is: ', gbm_predictions)
    # st.write('with an R2 score of: ', gmb_score)




