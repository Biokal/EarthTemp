import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models.annotations import Span
from joblib import load
from xgboost import XGBRegressor

st.set_page_config(page_title="Réchauffement Climatique",page_icon="pics/datascientest.png",layout="wide")

#TEST CARTE
import requests
import json
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook
import bokeh.palettes as palette
from bokeh.models import Range1d
from bokeh.models import FixedTicker
from bokeh.layouts import column
from bokeh.models import (ColorBar, LinearColorMapper)
from bokeh.palettes import brewer

def find_color_based_on_country_pop(index,country,year):
    temp_population = 0
    country_to_search = country
    if country[-1] in "0123456789":
      country_to_search = country[:-1]
    if country[-2] in "123456789":
      country_to_search = country[:-2]

    #CAS PAR CAS DES PAYS CAR LES FICHIERS N'ONT PAS LA MEME BASE
    #ON POURRAIT AUSSI RENOMMER DANS LE DF
    if country_to_search == 'The Bahamas' : country_to_search = 'Bahamas'
    if country_to_search == 'United States of America' : country_to_search = 'United States'
    if country_to_search == 'Democratic Republic of the Congo' : country_to_search = 'Democratic Republic of Congo'
    if country_to_search == 'Republic of the Congo' : country_to_search = 'Congo'
    if country_to_search == 'Northern Cyprus' : country_to_search = 'Cyprus'
    if country_to_search == 'Czech Republic' : country_to_search = 'Czechia'
    if country_to_search == 'Guinea Bissau' : country_to_search = 'Guinea-Bissau'
    if country_to_search == 'Macedonia' : country_to_search = 'North Macedonia'
    if country_to_search == 'Somaliland' : country_to_search = 'Somalia'
    if country_to_search == 'Republic of Serbia' : country_to_search = 'Serbia'
    if country_to_search == 'Swaziland' : country_to_search = 'Eswatini'
    if country_to_search == 'East Timor' : country_to_search = 'Timor'
    if country_to_search == 'United Republic of Tanzania' : country_to_search = 'Tanzania'
    if country_to_search == 'Ivory Coast' : country_to_search = 'Cote d\'Ivoire'
    if country_to_search == 'West Bank' : country_to_search = 'Israel'


    if country_to_search in df_population['Entity'].values:
      temp_population = df_population[ (df_population['Entity']==country_to_search)  &  (df_population['Year']==year) ]['Population (historical estimates)']
      temp_index = temp_population.index.values.astype(int)[0]
      temp_population = temp_population[temp_index]
    
    #TRI POUR ASSIGNER UNE COULEUR SUIVANT LA VALEUR DE POPULATION TROUVEE
    if temp_population <= 5000000 :
      temp_color = '#f7fcfd'
    elif temp_population <=10000000 :
      temp_color = '#e5f5f9'
    elif temp_population <= 15000000 :
      temp_color = '#ccece6'
    elif temp_population <= 20000000 :
      temp_color = '#99d8c9'
    elif temp_population <= 25000000 :
      temp_color = '#66c2a4'
    elif temp_population <= 30000000 :
      temp_color = '#41ae76'
    elif temp_population <= 35000000 :
      temp_color = '#238b45'
    else :
      temp_color = '#005824'
    return temp_color

def find_color_based_on_country_co2(index,country,year):
    temp_co2 = 0
    country_to_search = country
    if country[-1] in "0123456789":
      country_to_search = country[:-1]
    if country[-2] in "123456789":
      country_to_search = country[:-2]

    #CAS PAR CAS DES PAYS
    if country_to_search == 'The Bahamas' : country_to_search = 'Bahamas'
    if country_to_search == 'United States of America' : country_to_search = 'United States'
    if country_to_search == 'Democratic Republic of the Congo' : country_to_search = 'Democratic Republic of Congo'
    if country_to_search == 'Republic of the Congo' : country_to_search = 'Congo'
    if country_to_search == 'Northern Cyprus' : country_to_search = 'Cyprus'
    if country_to_search == 'Czech Republic' : country_to_search = 'Czechia'
    if country_to_search == 'Guinea Bissau' : country_to_search = 'Guinea-Bissau'
    if country_to_search == 'Macedonia' : country_to_search = 'North Macedonia'
    if country_to_search == 'Somaliland' : country_to_search = 'Somalia'
    if country_to_search == 'Republic of Serbia' : country_to_search = 'Serbia'
    if country_to_search == 'Swaziland' : country_to_search = 'Eswatini'
    if country_to_search == 'East Timor' : country_to_search = 'Timor'
    if country_to_search == 'United Republic of Tanzania' : country_to_search = 'Tanzania'
    if country_to_search == 'Ivory Coast' : country_to_search = 'Cote d\'Ivoire'
    if country_to_search == 'West Bank' : country_to_search = 'Israel'

    #Ici on a rajouté une sécurité car le dataset CO2 est moins complet que celui de la population
    #Ainsi si on ne trouve pas de valeur pour le pays concerné on affecte la couleur comme si la valeur était de 0
    if country_to_search in df_co2['Entity'].values:
      temp_co2 = df_co2[ (df_co2['Entity']==country_to_search)  &  (df_co2['Year']==year) ]['Annual CO₂ emissions']
      if len(temp_co2) != 0 :
        temp_index = temp_co2.index.values.astype(int)[0]
        temp_co2 = temp_co2[temp_index]
      else :
        return '#fff7ec'
    
    if temp_co2 <= 5000000 :
      temp_color = '#fff7ec'
    elif temp_co2 <=10000000 :
      temp_color = '#fee8c8'
    elif temp_co2 <= 150000000 :
      temp_color = '#fdd49e'
    elif temp_co2 <= 200000000 :
      temp_color = '#fdbb84'
    elif temp_co2 <= 250000000 :
      temp_color = '#fc8d59'
    elif temp_co2 <= 300000000 :
      temp_color = '#ef6548'
    elif temp_co2 <= 350000000 :
      temp_color = '#d7301f'
    else :
      temp_color = '#990000'
    return temp_color

countries = requests.get('https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json').json()

countryObject = {}
for country in countries['features']:
    index = 0
    if(country['geometry']['type']=='Polygon'):
      countryObject[country['properties']['name']] = {'x': [x[0] for x in country['geometry']['coordinates'][0]],
                                                      'y': [x[1] for x in country['geometry']['coordinates'][0]],}
    else:
      for polygons in country['geometry']['coordinates'] :
        countryObject[country['properties']['name']+str(index)] = {'x': [x[0] for x in country['geometry']['coordinates'][index][0]],
                                                      'y': [x[1] for x in country['geometry']['coordinates'][index][0]],}
        index = index + 1
                                   
df_countries_map = pd.DataFrame(countryObject)
#TEST CARTE


#FILES
df_global = pd.read_csv('csv/GLB.Ts+dSST.csv',header=1)
df_global = df_global.replace('***',np.nan)
df_global['DJF'] = df_global['DJF'].astype(np.float64)
df_north = pd.read_csv('csv/NH.Ts+dSST.csv',header=1)
df_north = df_north.replace ('***',np.nan)
df_north['DJF'] = df_north['DJF'].astype(np.float64)
df_south = pd.read_csv('csv/SH.Ts+dSST.csv',header=1)
df_south = df_south.replace ('***',np.nan)
df_south['DJF'] = df_south['DJF'].astype(np.float64)
df_zone = pd.read_csv('csv/ZonAnn.Ts+dSST.csv',header=1)

df_co2 = pd.read_csv('csv/CO2percountry.csv', header=0)
df_co2_new = df_co2[(df_co2['Year'] >= 1880) & (df_co2['Entity'] == 'World')]

df_CH4 = pd.read_csv('csv/methane-emissions.csv',header=0)
df_CH4_new = df_CH4[(df_CH4['Year'] >= 1880) & (df_CH4['Entity'] == 'World')]

df_N2O = pd.read_csv('csv/nitrous-oxide-emissions.csv',header=0)
df_N2O_new = df_N2O[(df_N2O['Year'] >= 1880) & (df_N2O['Entity'] == 'World')]

df_energy = pd.read_csv('csv/per-capita-energy-use.csv',header=0)
df_energy_new = df_energy[(df_energy['Year'] >= 1965) & (df_energy['Entity'] == 'World')]

df_forest = pd.read_csv('csv/forest.csv',header=0)
df_forest_new = df_forest[(df_forest['Year'] >= 1990) & (df_forest['Entity'] == 'World')]

df_population = pd.read_csv('csv/population.csv', header=0)
df_population_new = df_population[(df_population['Year'] >= 1880) & (df_population['Entity'] == 'World')]

#SAVED CSV
df_score_ml = pd.read_csv('csv/score_ml.csv',header=0)
df_score_ml_opti = pd.read_csv('csv/score_opti.csv',header=0)
df_time_ml = pd.read_csv('csv/time.csv',header=0)
df_time_opti = pd.read_csv('csv/time_opti.csv',header=0)
df_time_ml_with_opti = pd.read_csv('csv/time_with_optimized.csv',header=0)
df_test = pd.read_csv('csv/df_test.csv',header=0)
df_train = pd.read_csv('csv/df_train.csv',header=0)
df_rmse = pd.read_csv('csv/rmse_opti.csv',header=0)

#
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_test_scaled = scaler.fit_transform(df_test)

#GLOBAL VALUES
doc = curdoc()
doc.theme = 'dark_minimal'


#PAGES
pages = ["Présentation","Exploration des données","Cartes population interactives","Cartes CO2 interactives","Traitement des données","Modélisation","Conclusion"]
st.sidebar.image("pics/climate_icon.png",width=200)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller vers:",pages)
st.sidebar.header("Promotion Data Analyst 'Bootcamp' Novembre 2022")
st.sidebar.markdown("""---""")
st.sidebar.subheader('Membres de l\'équipe : ')
col1, col2 = st.sidebar.columns([1,1])
with col1:
    col1.write('DELTOUR Rémy')
    col1.write('SEU Flora')
with col2:
    col2.write('GOUVERNEUR Pauline')
    col2.write('SOUCOURS Eloise')

#ORGA DES PAGES
if page == pages[0]:
  col1,col2 = st.columns([1,5])
  with col2:
    st.title("Projet d'analyse de la Température Terrestre")
  st.image("pics/intro_image.jpg")
  st.write("“Réchauffement climatique”, “hausse des températures”, “fonte des glaces” : la question climatique est depuis quelques années au cœur des préoccupations géopolitiques, et bel et bien au centre de toutes nos discussions quotidiennes. Largement relayée par les médias, la notion de réchauffement climatique se lit tous les jours dans des articles de presse, magazines scientifiques ou se “regarde” à la télévision aux heures de grandes écoutes, et force à une prise de conscience inéluctable. 2022, est, pour de nombreux chercheurs “un avant-goût des climats futurs”. ")
  st.header("Problématique")
  st.write("Constater le réchauffement (et le dérèglement) climatique global à l’échelle de la planète sur les derniers siècles et dernières décennies.")
  st.header("Plan d'étude")
  col1, col2 = st.columns([1,40])
  with col2:
    st.write("1. Exploration de nos données")
    st.write("2. Ajout de données externes")
    st.write("3. Regroupement des données utiles")
    st.write("4. Visualisation des données")
    st.write("5. Mise en place d'algorithmes de Machine Learning")
    st.write("6. Prédictions")

if page == pages[1]:
    st.title("Exploration des données")
    st.header("Données de la NASA")
    st.write("Nos données initiales proviennent d'un site mis en place par la NASA à cette adresse : [https://data.giss.nasa.gov/gistemp/](https://data.giss.nasa.gov/gistemp/)")
    st.write("Sur ce site nous récupérons 3 fichiers au format CSV : ")
    col1, col2 = st.columns([1,40])
    with col2:
      st.write("1. [Global-mean monthly, seasonal, and annual means](https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv)")
      st.write("2. [Northern Hemisphere-mean monthly, seasonal, and annual means](https://data.giss.nasa.gov/gistemp/tabledata_v4/NH.Ts+dSST.csv)")
      st.write("3. [Southern Hemisphere-mean monthly, seasonal, and annual means](https://data.giss.nasa.gov/gistemp/tabledata_v4/SH.Ts+dSST.csv)")
    expander = st.expander("Afficher les 10 premières et dernières lignes du jeu de donnée Global")
    expander.dataframe(df_global.head())
    expander.dataframe(df_global.tail())
    expander2 = st.expander("Afficher les 10 premières et dernières lignes du jeu de donnée Northern")
    expander2.dataframe(df_north.head())
    expander2.dataframe(df_north.tail())
    expander3 = st.expander("Afficher les 10 premières et dernières lignes du jeu de donnée Southern")
    expander3.dataframe(df_south.head())
    expander3.dataframe(df_south.tail())

    st.write("Nos données sont réparties par mois, par periode de 12 mois ainsi que par période de 3 mois.")
    st.write("Dimension de nos données : ")
    st.write("• Nombre de lignes : " + str(df_global.shape[0]) + ' / ' + str(df_north.shape[0]) + ' / ' + str(df_south.shape[0]))
    st.write("• Nombre de colonnes : " + str(df_global.shape[1]) + ' / ' + str(df_north.shape[1]) + ' / ' + str(df_south.shape[1]))
    st.write("• Nombre de valeurs nulles : " + str(df_global.isna().sum().sum()) + ' / ' + str(df_north.isna().sum().sum()) +' / '+ str(df_south.isna().sum().sum()))
    st.write(pd.DataFrame(data=[df_global.isna().sum(),df_north.isna().sum(),df_south.isna().sum()],columns=df_global.columns,index=['Global','North','South']))
    st.write("Ces valeurs nulles sont explicables par leurs dates. En effet, nos données ont été récupérées milieu Novembre 2022 et nous n'avons aucune données d'avant 1880.")

    expander4 = st.expander("Afficher le graphique des données de la NASA")
    p_global = figure(width=700, height=400, title = "Déviation de la température terrestre")
    p_global.line(x = df_north['Year'], y = df_north['DJF'], color = 'pink',legend_label='Hem. Nord')
    p_global.line(x = df_global['Year'], y = df_global['DJF'],color='cyan',legend_label='Global')
    p_global.line(x = df_south['Year'], y = df_south['DJF'], color = 'lime',legend_label='Hem. Sud')
    p_global.legend.location = "top_left"
    p_global.yaxis.axis_label = "Déviation en °C"
    p_global.xaxis.axis_label = "Année"
    p_global.xaxis.major_label_orientation = 0.33
    p_global.legend.click_policy="hide"
    p_global_span = Span(location=0,dimension='width',line_width=2,line_color='grey')
    p_global.add_layout(p_global_span)
    #1ère légende
    label1 = Span(location= 1880,
                  dimension='height', line_color='orange',
                  line_dash='dashed', line_width=1)
    p_global.line(x = 1880, y =-0.18,
    line_color='orange', line_dash='dashed', line_width=1,
    legend_label="Seconde révolution industrielle (pétrole, électricité, automobile)")
    #2ème légende
    label2 = Span(location= 1918,
                  dimension='height', line_color='red',
                  line_dash='dashed', line_width=1)
    p_global.line(x = 1918, y = -0.15,
    line_color='red', line_dash='dashed', line_width=1,
    legend_label="Fin de la première guerre mondiale")
    #3ème légende
    label3 = Span(location= 1945,
                  dimension='height', line_color='lime',
                  line_dash='dashed', line_width=1)
    p_global.line(x = 1945, y = -0.02,
    line_color='lime', line_dash='dashed', line_width=1,
    legend_label="Fin de la seconde guerre mondiale")
    #4ème légende
    label4 = Span(location= 1970,
                  dimension='height', line_color='pink',
                  line_dash='dashed', line_width=1)
    p_global.line(x = 1970, y =-0.03,
    line_color='pink', line_dash='dashed', line_width=1,
    legend_label="Révolution numérique")
    #5ème légende
    label5 = Span(location= 2020,
                  dimension='height', line_color='yellow',
                  line_dash='dashed', line_width=1)
    p_global.line(x = 2020, y = 1.02,
    line_color='yellow', line_dash='dashed', line_width=1,
    legend_label="Apparition du Covid-19")
    p_global.add_layout(label1)
    p_global.add_layout(label2)
    p_global.add_layout(label3)
    p_global.add_layout(label4)
    p_global.add_layout(label5)
    
    p_global.add_layout(p_global.legend[0], 'right')

    doc.add_root(p_global)
    expander4.bokeh_chart(p_global,use_container_width=True)


    st.header("Données additionnelles")
    st.write("Afin de compléter nos données initiales, nous avons choisi et récupéré les données au format csv, de 6 facteurs complémentaires pouvant expliquer le réchauffement climatique, \n et trouvées sur le site : [https://ourworldindata.org/](https://ourworldindata.org/)")
    col1, col2 = st.columns([1,40])
    with col2:
      st.write('1.[Annual CO2 emissions](https://ourworldindata.org/grapher/annual-co2-emissions-per-country)')
      st.write('2.[Annual Methane emissions (CH4)](https://ourworldindata.org/grapher/methane-emissions)')
      st.write('3.[Annual Nitrous oxide emissions (N2O)](https://ourworldindata.org/grapher/nitrous-oxide-emissions)')
      st.write('4.[Energy use per person](https://ourworldindata.org/grapher/per-capita-energy-use)')
      st.write('5.[Forest Area](https://ourworldindata.org/grapher/forest-area-km)')
      st.write('6.[Population, 1800 to 2021](https://ourworldindata.org/grapher/population-since-1800)')
    expander = st.expander("Afficher les DataFrames")
    option = expander.selectbox('Quel DataFrame souhaitez vous afficher ?',('CO2','CH4','N2O','Energie','Forêts','Population'))
    if option == 'CO2':
      expander.dataframe(df_co2.head())
    elif option == 'CH4':
      expander.dataframe(df_CH4.head())
    elif option == 'N2O':
      expander.dataframe(df_N2O.head())
    elif option == 'Energie':
      expander.dataframe(df_energy.head())
    elif option == 'Forêts':
      expander.dataframe(df_forest.head())
    elif option == 'Population':
      expander.dataframe(df_population.head())
    st.write("Pour la suite du projet nous nous concentrons sur des valeurs mondiales. Un nettoyage et un filtre de nos données seront alors à effectuer en amont de notre analyse.")
    st.header("Visualisation de nos données")
    st.subheader("CO2")
    p_co2 = figure(width=700, height=400, title = "Emission mondiale de CO2 au cours des années")
    p_co2.line(x = df_co2_new['Year'], y = df_co2_new['Annual CO₂ emissions'], color = "orange")
    p_co2.xaxis.axis_label = "Année"              
    p_co2.yaxis.axis_label = "Emission de CO2 en tonnes"
    #1ère légende
    label1 = Span(location= 1880,
                  dimension='height', line_color='orange',
                  line_dash='dashed', line_width=1)
    p_co2.line(x = 1880, y =-0.18,
    line_color='orange', line_dash='dashed', line_width=1,
    legend_label="Seconde révolution industrielle (pétrole, électricité, automobile)")
    #2ème légende
    label2 = Span(location= 1918,
                  dimension='height', line_color='red',
                  line_dash='dashed', line_width=1)
    p_co2.line(x = 1918, y = -0.15,
    line_color='red', line_dash='dashed', line_width=1,
    legend_label="Fin de la première guerre mondiale")
    #3ème légende
    label3 = Span(location= 1945,
                  dimension='height', line_color='lime',
                  line_dash='dashed', line_width=1)
    p_co2.line(x = 1945, y = -0.02,
    line_color='lime', line_dash='dashed', line_width=1,
    legend_label="Fin de la seconde guerre mondiale")
    #4ème légende
    label4 = Span(location= 1970,
                  dimension='height', line_color='pink',
                  line_dash='dashed', line_width=1)
    p_co2.line(x = 1970, y =-0.03,
    line_color='pink', line_dash='dashed', line_width=1,
    legend_label="Révolution numérique")
    #5ème légende
    label5 = Span(location= 2020,
                  dimension='height', line_color='yellow',
                  line_dash='dashed', line_width=1)
    p_co2.line(x = 2020, y = 1.02,
    line_color='yellow', line_dash='dashed', line_width=1,
    legend_label="Apparition du Covid-19")
    p_co2.add_layout(label1)
    p_co2.add_layout(label2)
    p_co2.add_layout(label3)
    p_co2.add_layout(label4)
    p_co2.add_layout(label5)
    p_co2.add_layout(p_co2.legend[0], 'right')
    #Fonction linéaire nous permettant de mettre en avant les valeurs jusqu'en 2050
    par = np.polyfit(df_co2_new['Year'], df_co2_new['Annual CO₂ emissions'], 1, full=True)
    slope=par[0][0]
    intercept=par[0][1]
    X_range = np.arange(1880,2051,1)
    y_co2_predicted = [slope*i + intercept  for i in X_range]

    p_co2.line(x = X_range, y = y_co2_predicted,color='grey',legend_label='Prédiction linéaire')
    doc.add_root(p_co2)
    st.bokeh_chart(p_co2,use_container_width=True)

    expander = st.expander("Afficher les informations du DataFrame Emission de CO2 mondial : ")
    expander.write("Dimension de nos données : ")
    expander.write("• Nombre de lignes : " + str(df_co2_new.shape[0]))
    expander.write("• Nombre de colonnes : " + str(df_co2_new.shape[1]))
    expander.write("• Nombre de valeurs nulles : " + str(df_co2_new.isna().sum().sum()))
    expander.dataframe(df_co2_new.head())

    st.write("Mise en avant des emissions de CO2 par pays à partir des années 1980 car c'est l'année où la plupart des données sont disponibles : ")
    col1, col2 = st.columns([1,1])
    with col1:
      st.image("pics/map_co2_1980.png")
    with col2:
      st.image("pics/map_co2_2021.png")

    st.subheader("CH4")
    p_ch4 = figure(width=700, height=400, title = "Emission mondiale de CH4 au cours des années")
    p_ch4.line(x = df_CH4_new ['Year'], y = df_CH4_new ['Total including LUCF'], color = "orange")
    p_ch4.xaxis.axis_label = "Année"              
    p_ch4.yaxis.axis_label = "Emission de méthane en tonnes"
    
    #Fonction linéaire nous permettant de mettre en avant les valeurs jusqu'en 2050
    par = np.polyfit(df_CH4_new['Year'], df_CH4_new['Total including LUCF'], 1, full=True)
    slope=par[0][0]
    intercept=par[0][1]
    X_range = np.arange(1880,2051,1)
    y_ch4_predicted = [slope*i + intercept  for i in X_range]
    p_ch4.line(x = X_range, y = y_ch4_predicted,color='grey',legend_label='Prédiction linéaire')
    p_ch4.add_layout(p_ch4.legend[0], 'right')
    p_ch4.x_range = Range1d(start = 1985, end = 2025)
    p_ch4.y_range = Range1d(start = 6e+9, end = 9e+9)
    doc.add_root(p_ch4)
    st.bokeh_chart(p_ch4,use_container_width=True)

    expander = st.expander("Afficher les informations du DataFrame Emission de CH4 mondial : ")
    expander.write("Dimension de nos données : ")
    expander.write("• Nombre de lignes : " + str(df_CH4_new.shape[0]))
    expander.write("• Nombre de colonnes : " + str(df_CH4_new.shape[1]))
    expander.write("• Nombre de valeurs nulles : " + str(df_CH4_new.isna().sum().sum()))
    expander.dataframe(df_CH4_new.head())


    st.subheader("N2O")
    p_n2o = figure(width=700, height=400, title = "Emission mondiale de N2O au cours des années")
    p_n2o.line(x = df_N2O_new ['Year'], y = df_N2O_new ['Total including LUCF'], color = "pink")
    p_n2o.xaxis.axis_label = "Année"              
    p_n2o.yaxis.axis_label = "Emission de protoxyde d'azote en tonnes"

    #Fonction linéaire nous permettant de mettre en avant les valeurs jusqu'en 2050
    par = np.polyfit(df_N2O_new['Year'], df_N2O_new['Total including LUCF'], 1, full=True)
    slope=par[0][0]
    intercept=par[0][1]
    X_range = np.arange(1880,2051,1)
    y_n2O_predicted = [slope*i + intercept  for i in X_range]

    p_n2o.line(x = X_range, y = y_n2O_predicted,color='grey',legend_label='Prédiction linéaire')
    p_n2o.add_layout(p_n2o.legend[0], 'right')
    p_n2o.x_range = Range1d(start = 1985, end = 2025)
    p_n2o.y_range = Range1d(start = 2e+9, end = 3.4e+9)
    doc.add_root(p_n2o)
    st.bokeh_chart(p_n2o,use_container_width=True)

    expander = st.expander("Afficher les informations du DataFrame Emission de N2O mondial : ")
    expander.write("Dimension de nos données : ")
    expander.write("• Nombre de lignes : " + str(df_N2O_new.shape[0]))
    expander.write("• Nombre de colonnes : " + str(df_N2O_new.shape[1]))
    expander.write("• Nombre de valeurs nulles : " + str(df_N2O_new.isna().sum().sum()))
    expander.dataframe(df_N2O_new.head())

    st.subheader("Energie")
    p_energy = figure(width=700, height=400, title = "Consommation mondiale de l'énergie au cours des années")
    p_energy.line(x = df_energy_new['Year'], y = df_energy_new['Primary energy consumption per capita (kWh/person)'], color = "yellow")
    p_energy.xaxis.axis_label = "Année"              
    p_energy.yaxis.axis_label = "Consommation d'énergie en kWh/personne"
    #4ème légende
    label4 = Span(location= 1970,
                  dimension='height', line_color='pink',
                  line_dash='dashed', line_width=1)
    p_energy.add_layout(label4)
    p_energy.line(x = 1970, y = -0.03,
    line_color='pink', line_dash='dashed', line_width=1,
    legend_label="Révolution numérique")
    #5ème légende
    label5 = Span(location= 2020,
                  dimension='height', line_color='yellow',
                  line_dash='dashed', line_width=1)
    p_energy.add_layout(label5)
    p_energy.line(x = 2020, y = 1.02,
      line_color='yellow', line_dash='dashed', line_width=1,
      legend_label="Apparition du Covid-19")
    #Fonction linéaire nous permettant de mettre en avant les valeurs jusqu'en 2050
    par = np.polyfit(df_energy_new['Year'], df_energy_new['Primary energy consumption per capita (kWh/person)'], 1, full=True)
    slope=par[0][0]
    intercept=par[0][1]
    X_range = np.arange(1880,2051,1)
    y_energy_predicted = [slope*i + intercept  for i in X_range]

    p_energy.line(x = X_range, y = y_energy_predicted,color='grey',legend_label='Prédiction linéaire')
    p_energy.add_layout(p_energy.legend[0], 'right')
    p_energy.x_range = Range1d(start = 1960, end = 2025)
    p_energy.y_range = Range1d(start = 10000, end = 24000)
    doc.add_root(p_energy)
    st.bokeh_chart(p_energy,use_container_width=True)

    expander = st.expander("Afficher les informations du DataFrame Consommation d'énergie mondiale : ")
    expander.write("Dimension de nos données : ")
    expander.write("• Nombre de lignes : " + str(df_energy_new.shape[0]))
    expander.write("• Nombre de colonnes : " + str(df_energy_new.shape[1]))
    expander.write("• Nombre de valeurs nulles : " + str(df_energy_new.isna().sum().sum()))
    expander.dataframe(df_energy_new.head())

    st.subheader("Forêts")
    p_forest = figure(width=700, height=400, title = "Surface Forestière mondiale au cours des années")
    p_forest.line(x = df_forest_new['Year'], y = df_forest_new['Forest area'], color = "lime")
    p_forest.xaxis.axis_label = "Année"              
    p_forest.yaxis.axis_label = "Surface forestière en ha"
    #Fonction linéaire nous permettant de mettre en avant les valeurs jusqu'en 2050
    par = np.polyfit(df_forest_new['Year'], df_forest_new['Forest area'], 1, full=True)
    slope=par[0][0]
    intercept=par[0][1]
    X_range = np.arange(1880,2051,1)
    y_forest_predicted = [slope*i + intercept  for i in X_range]

    p_forest.line(x = X_range, y = y_forest_predicted,color='grey',legend_label='Prédiction linéaire')
    p_forest.x_range = Range1d(start = 1975, end = 2035)
    p_forest.y_range = Range1d(start = 3.9e+9, end = 4.5e+9)
    p_forest.add_layout(p_forest.legend[0], 'right')
    doc.add_root(p_forest)
    st.bokeh_chart(p_forest,use_container_width=True)

    expander = st.expander("Afficher les informations du DataFrame Surface Forestière mondiale : ")
    expander.write("Dimension de nos données : ")
    expander.write("• Nombre de lignes : " + str(df_forest_new.shape[0]))
    expander.write("• Nombre de colonnes : " + str(df_forest_new.shape[1]))
    expander.write("• Nombre de valeurs nulles : " + str(df_forest_new.isna().sum().sum()))
    expander.dataframe(df_forest_new.head())

    st.subheader("Population")
    p_pop = figure(width=700, height=400, title = "Population mondiale au cours des années")
    p_pop.line(x = df_population_new['Year'], y = df_population_new['Population (historical estimates)'], color = "red")
    p_pop.xaxis.axis_label = "Année"              
    p_pop.yaxis.axis_label = "Population" 
    #1ère légende
    label1 = Span(location= 1880,
                  dimension='height', line_color='orange',
                  line_dash='dashed', line_width=1)
    p_pop.line(x = 1880, y =-0.18,
    line_color='orange', line_dash='dashed', line_width=1,
    legend_label="Seconde révolution industrielle (pétrole, électricité, automobile)")
    #2ème légende
    label2 = Span(location= 1918,
                  dimension='height', line_color='red',
                  line_dash='dashed', line_width=1)
    p_pop.line(x = 1918, y = -0.15,
    line_color='red', line_dash='dashed', line_width=1,
    legend_label="Fin de la première guerre mondiale")
    #3ème légende
    label3 = Span(location= 1945,
                  dimension='height', line_color='lime',
                  line_dash='dashed', line_width=1)
    p_pop.line(x = 1945, y = -0.02,
    line_color='lime', line_dash='dashed', line_width=1,
    legend_label="Fin de la seconde guerre mondiale")
    #4ème légende
    label4 = Span(location= 1970,
                  dimension='height', line_color='pink',
                  line_dash='dashed', line_width=1)
    p_pop.line(x = 1970, y =-0.03,
    line_color='pink', line_dash='dashed', line_width=1,
    legend_label="Révolution numérique")
    #5ème légende
    label5 = Span(location= 2020,
                  dimension='height', line_color='yellow',
                  line_dash='dashed', line_width=1)
    p_pop.line(x = 2020, y = 1.02,
    line_color='yellow', line_dash='dashed', line_width=1,
    legend_label="Apparition du Covid-19")
    p_pop.add_layout(label1)
    p_pop.add_layout(label2)
    p_pop.add_layout(label3)
    p_pop.add_layout(label4)
    p_pop.add_layout(label5)

    #Fonction linéaire nous permettant de mettre en avant les valeurs jusqu'en 2050
    par = np.polyfit(df_population_new['Year'], df_population_new['Population (historical estimates)'], 1, full=True)
    slope=par[0][0]
    intercept=par[0][1]
    X_range = np.arange(1880,2051,1)
    y_pop_predicted = [slope*i + intercept  for i in X_range]

    p_pop.line(x = X_range, y = y_pop_predicted,color='grey',legend_label='Prédiction linéaire')
    p_pop.add_layout(p_pop.legend[0], 'right')
    doc.add_root(p_pop)
    st.bokeh_chart(p_pop,use_container_width=True)

    expander = st.expander("Afficher les informations du DataFrame Population mondiale : ")
    expander.write("Dimension de nos données : ")
    expander.write("• Nombre de lignes : " + str(df_population_new.shape[0]))
    expander.write("• Nombre de colonnes : " + str(df_population_new.shape[1]))
    expander.write("• Nombre de valeurs nulles : " + str(df_population_new.isna().sum().sum()))
    expander.dataframe(df_population_new.head())

    st.write("Mise en avant de la Population par pays à partir des années 1980 car c'est l'année où la plupart des données sont disponibles : ")
    col1, col2 = st.columns([1,1])
    with col1:
      st.image("pics/map_pop_1980.png")
    with col2:
      st.image("pics/map_pop_2021.png")

if page == pages[4]:
  st.title("Traitement des données")
  st.header("Regroupement dans un DataFrame d'entraînement")
  st.write("Après nettoyage et jointures de nos différentes données nous obtenons le DataFrame suivant : ")
  st.dataframe(df_train)
  st.write("Correlation de nos valeurs : ")
  st.image("pics/heatmap.png",)
  st.header("Regroupement dans un DataFrame de prédictions")
  st.write("Afin de construire ce DataFrame qui nous servira à effectuer des prédictions de déviations de température, nous utilisons nos valeurs des prédictions linéaires basiques de 2023 à 2050.")
  st.dataframe(df_test)
    
if page == pages[2]:
    st.title("Cartes de Population")
    p_countries_map_pop_1980 = figure(
    width = 800, 
    height=400, 
    title='Pays du monde colorés en fonction de leur population en 1980', 
    x_axis_label='Longitude',
    y_axis_label='Latitude',
    )

    p_countries_map_pop_2021 = figure(
    width = 800, 
    height=400, 
    title='Pays du monde colorés en fonction de leur population en 2021', 
    x_axis_label='Longitude',
    y_axis_label='Latitude',
    )

    palette = brewer['BuGn'][8]
    palette = palette[::-1]

    color_mapper = LinearColorMapper(palette = palette, low = 0, high = 40000000)
 
    tick_labels = {'0': '0', '5000000': '5,000,000',
 '10000000':'10,000,000', '15000000':'15,000,000',
 '20000000':'20,000,000', '25000000':'25,000,000',
 '30000000':'30,000,000', '35000000':'35,000,000',
 '40000000':'40,000,000+'}

    for (index,country) in enumerate(df_countries_map):

        p_countries_map_pop_1980.patch(
        x=df_countries_map[country]['x'],
        y=df_countries_map[country]['y'],
        fill_color = find_color_based_on_country_pop(index,country,1980),
        alpha = 0.8
        )
        p_countries_map_pop_2021.patch(
        x=df_countries_map[country]['x'],
        y=df_countries_map[country]['y'],
        fill_color = find_color_based_on_country_pop(index,country,2021),
        alpha = 0.8
        )

    color_bar = ColorBar(color_mapper = color_mapper, 
                     label_standoff = 8,
                     width = 500, height = 20,
                     border_line_color = None,
                     location = (0,0), 
                     orientation = 'horizontal',
                     major_label_overrides = tick_labels)   
    p_countries_map_pop_1980.add_layout(color_bar, 'below')
    p_countries_map_pop_2021.add_layout(color_bar, 'below')

    #On limite nos range pour l'affichage de base
    p_countries_map_pop_2021.x_range = Range1d(start = -180, end = 180)
    p_countries_map_pop_2021.y_range = Range1d(start = -90, end = 90)

    p_countries_map_pop_1980.x_range = Range1d(start = -180, end = 180)
    p_countries_map_pop_1980.y_range = Range1d(start = -90, end = 90)


    st.header("Carte de la population par pays en 2021")
    doc.add_root(p_countries_map_pop_2021)
    st.bokeh_chart(p_countries_map_pop_2021,use_container_width=False)
    st.header("Carte de la population par pays en 1980")
    doc.add_root(p_countries_map_pop_1980)
    st.bokeh_chart(p_countries_map_pop_1980,use_container_width=False)
    

    expander_pop = st.expander("Afficher la Population de différents Pays : ")
    expander_pop.write("La population de la France comparée à celle d’autres pays.")
    option2 = expander_pop.selectbox('Choix du pays à comparer :',('Argentine','Canada','Maroc','Afrique du Sud','Colombie'),key=12)
    if option2 == 'Argentine':
      recherche = 'Argentina'
    elif option2 == 'Afrique du Sud':
      recherche = 'South Africa'
    elif option2 == 'Canada':
      recherche = 'Canada'
    elif option2 == 'Maroc':
      recherche = 'Morocco'
    elif option2 == 'Colombie':
      recherche = 'Colombia'
    col1, col2 = expander_pop.columns([1,1])
    with col1:
      col1.write("France")
      col1.dataframe(df_population[df_population['Entity']=='France'])
    with col2:
      col2.write(option2)
      col2.dataframe(df_population[(df_population['Entity']==recherche) & (df_population['Population (historical estimates)']>0)])

if page == pages[3]:
    st.title("Cartes des emissions de CO2")
    p_countries_map_co2_1980 = figure(
    width = 800, 
    height=400, 
    title='Pays du monde colorés en fonction de leur emission de CO2 en 1980', 
    x_axis_label='Longitude',
    y_axis_label='Latitude'
    )
    p_countries_map_co2_2021 = figure(
    width = 800, 
    height=400, 
    title='Pays du monde colorés en fonction de leur emission de CO2 en 2021', 
    x_axis_label='Longitude',
    y_axis_label='Latitude'
    )

    palette = brewer['OrRd'][9]
    palette = palette[::-1]

    color_mapper = LinearColorMapper(palette = palette, low = 0, high = 100000000)
 
    tick_labels = {'0': '0', '5000000': '5,000,000',
 '10000000':'10,000,000', '150000000':'150,000,000',
 '200000000':'200,000,000', '250000000':'250,000,000',
 '300000000':'300,000,000', '350000000':'350,000,000',
 '400000000':'400,000,000', '1000000000':'1,000,000,000+'}

    for (index,country) in enumerate(df_countries_map):
        p_countries_map_co2_2021.patch(
        x=df_countries_map[country]['x'],
        y=df_countries_map[country]['y'],
        fill_color = find_color_based_on_country_co2(index,country,2021),
        alpha = 0.8
        )

        p_countries_map_co2_1980.patch(
        x=df_countries_map[country]['x'],
        y=df_countries_map[country]['y'],
        fill_color = find_color_based_on_country_co2(index,country,1980),
        alpha = 0.8
        )
    ticks = np.linspace(0, 100000000, 10).tolist()
    fT = FixedTicker(ticks=ticks)

    color_bar = ColorBar(color_mapper = color_mapper, 
                     label_standoff = 5,
                     width = 600, height = 20,
                     border_line_color = None,
                     location = (0,0), 
                     orientation = 'horizontal',
                     ticker=fT,
                     major_label_overrides = tick_labels)
    p_countries_map_co2_2021.x_range = Range1d(start = -180, end = 180)
    p_countries_map_co2_2021.y_range = Range1d(start = -90, end = 90)

    p_countries_map_co2_1980.x_range = Range1d(start = -180, end = 180)
    p_countries_map_co2_1980.y_range = Range1d(start = -90, end = 90)

    p_countries_map_co2_2021.add_layout(color_bar, 'below')
    p_countries_map_co2_1980.add_layout(color_bar, 'below')

    st.header("Carte de l'emission de CO2 par pays en 2021")
    doc.add_root(p_countries_map_co2_2021)
    st.bokeh_chart(p_countries_map_co2_2021,use_container_width=False)
    st.header("Carte de l'emission de CO2 par pays en 1980")
    doc.add_root(p_countries_map_co2_1980)
    st.bokeh_chart(p_countries_map_co2_1980,use_container_width=False)
    

    

    

    expander = st.expander("Afficher les emissions de CO2 de différents Pays : ")
    expander.write("L'emission de CO2 de la France comparée à celle d’autres pays.")
    option = expander.selectbox('Choix du pays à comparer :',('Brésil','Afrique du Sud','Australie','Inde','Mexique'))
    if option == 'Brésil':
      recherche = 'Brazil'
    elif option == 'Afrique du Sud':
      recherche = 'South Africa'
    elif option == 'Australie':
      recherche = 'Australia'
    elif option == 'Inde':
      recherche = 'India'
    elif option == 'Mexique':
      recherche = 'Mexico'
    col1, col2 = expander.columns([1,1])
    with col1:
      col1.write("France")
      col1.dataframe(df_co2[df_co2['Entity']=='France'])
    with col2:
      col2.write(option)
      col2.dataframe(df_co2[(df_co2['Entity']==recherche) & (df_co2['Annual CO₂ emissions']>0)])

if page == pages[5]:

    linear_model = load('joblib/linear_model.joblib')
    decisiontree_model = load('joblib/decisiontree_model.joblib')
    randomforest_model = load('joblib/randomforest_model.joblib')
    lasso_model = load('joblib/lasso_model.joblib')
    ridge_model = load('joblib/ridge_model.joblib')
    ridge_model_opti = load('joblib/ridge_opti.joblib')
    elasticnet_model = load('joblib/elasticnet_model.joblib')

    st.title("Modélisation")
    st.header("Nos modèles")
    st.write("Nous implémentons 7 modèles différents : ")
    col1, col2 = st.columns([1,40])
    with col2:
      st.write("• Linear Regressor")
      st.write("• Decision Tree Regressor")
      st.write("• Random Forest Regressor")
      st.write("• Lasso")
      st.write("• Ridge")
      st.write("• ElasticNet")
      st.write("• Extreme Gradient Boosting Regressor (XGB Regressor)")
    st.subheader("Importances des variables pour quelques modèles")
    st.write("Random Forest Regressor")
    st.image('pics/Importance_Feats_RandomForest.png')
    st.write("XGB Regressor")
    st.image('pics/XGB_Feats_Importance.png')


    st.header("Score (R2) de nos modèles sans optimisation")
    df_score_ml.index = ['train','test']
    df_score_ml.loc['train','Random Forest'] = 0.9710
    df_score_ml.loc['test','Random Forest'] = 0.5085
    df_score_ml.loc['train','XGB Regressor'] = 0.7436
    df_score_ml.loc['test','XGB Regressor'] = 0.4406
    st.dataframe(df_score_ml)

    st.write("Entraînement de nos modèles sans optimisation")
    df_time_ml.index =['Temps (s)']
    st.dataframe(df_time_ml)



    st.header("Score (R2) de nos modèles avec optimisation")
    df_score_ml_opti.index = ['train','test']
    df_score_ml_opti.loc['train','Random Forest'] = 0.9710
    df_score_ml_opti.loc['test','Random Forest'] = 0.5085
    df_score_ml_opti.loc['train','XGB Regressor'] = 0.7436
    df_score_ml_opti.loc['test','XGB Regressor'] = 0.4406
    df_score_ml_opti.loc['train','XGB Opti'] = 0.9999
    df_score_ml_opti.loc['test','XGB Opti'] = 0.4463
    st.dataframe(df_score_ml_opti)
    
    col1, col2, col3= st.columns([1,1,2])
    with col1:
      st.write("Recherche de paramètres pour nos modèles")
      df_time_opti.index =['Temps (s)']
      st.dataframe(df_time_opti)
    with col2:
      st.write("Score RMSE pour nos modèles")
      df_rmse = df_rmse.rename(columns={'Lowest RMSE Ridge':'Ridge Opti','Lowest RMSE XGB':'XGB Opti'})
      df_rmse.index =['Lowest RMSE']
      st.dataframe(df_rmse)

    st.write("Temps d'entraînement de nos modèles avec optimisation")
    df_time_ml_with_opti.index =['Temps (s)']
    st.dataframe(df_time_ml_with_opti)

    st.header("Prédictions")
    st.write("Affichage de la courbe de déviation de la température terrestre des données de la NASA et nos différentes prédictions grâce à nos modèles")

    

    predictions_linear = linear_model.predict(df_test_scaled)
    predictions_decisiontree = decisiontree_model.predict(df_test_scaled)
    predictions_forest = randomforest_model.predict(df_test_scaled)
    predictions_lasso = lasso_model.predict(df_test_scaled)
    predictions_ridge = ridge_model.predict(df_test_scaled)
    predictions_ridge_opti = ridge_model_opti.predict(df_test_scaled)

    predictions_elasticnet = elasticnet_model.predict(df_test_scaled)

    p_predictions = figure(width=1000, height=600, title = "Déviation de la température terrestre au cours des années")

    p_predictions.line(x = df_global['Year'], y = df_global['J-D'])
    p_predictions.line(x = range(2023,2051,1), y = predictions_linear, color='red',legend_label='Linear Regression')
    p_predictions.line(x = range(2023,2051,1), y = predictions_decisiontree, color='green',legend_label='Decision Tree')
    p_predictions.line(x = range(2023,2051,1), y = predictions_forest, color='cyan',legend_label='Random Forest')
    p_predictions.line(x = range(2023,2051,1), y = predictions_lasso, color='purple',legend_label='Lasso')
    p_predictions.line(x = range(2023,2051,1), y = predictions_ridge, color='orange',legend_label='Ridge')
    p_predictions.line(x = range(2023,2051,1), y = predictions_elasticnet, color='blue',legend_label='ElasticNet')
    p_predictions.legend.click_policy="hide"
    p_predictions.yaxis.axis_label = "Déviation en °C"
    p_predictions.xaxis.axis_label = "Année"
    p_predictions_span = Span(location=0,dimension='width',line_width=2,line_color='grey')
    p_predictions.add_layout(p_predictions_span)
    p_predictions.add_layout(p_predictions.legend[0], 'right')
    doc.add_root(p_predictions)
    st.bokeh_chart(p_predictions,use_container_width=True)
    

    st.write("Après interprétation de nos résultats, nous décidons d\'instaurer un biais de **0.5°C** sur la majorité de nos prédictions, le résultat est donc le suivant")

    p_predictions2 = figure(width=1000, height=600, title = "Déviation de la température terrestre au cours des années avec biais")

    p_predictions2.line(x = df_global['Year'], y = df_global['J-D'])
    p_predictions2.line(x = range(2023,2051,1), y = predictions_linear + 0.5, color='red',legend_label='Linear Regression')
    p_predictions2.line(x = range(2023,2051,1), y = predictions_decisiontree + 0.5, color='green',legend_label='Decision Tree')
    p_predictions2.line(x = range(2023,2051,1), y = predictions_forest + 0.5, color='cyan',legend_label='Random Forest')
    p_predictions2.line(x = range(2023,2051,1), y = predictions_lasso + 0.5, color='purple',legend_label='Lasso')
    p_predictions2.line(x = range(2023,2051,1), y = predictions_ridge + 0.5, color='orange',legend_label='Ridge')
    p_predictions2.line(x = range(2023,2051,1), y = predictions_elasticnet + 0.5, color='blue',legend_label='ElasticNet')
    p_predictions2.legend.click_policy="hide"
    p_predictions2.yaxis.axis_label = "Déviation en °C"
    p_predictions2.xaxis.axis_label = "Année"
    p_predictions2_span = Span(location=0,dimension='width',line_width=2,line_color='grey')
    p_predictions2.add_layout(p_predictions2_span)
    p_predictions2.add_layout(p_predictions2.legend[0], 'right')
    doc.add_root(p_predictions2)
    st.bokeh_chart(p_predictions2,use_container_width=True)

    st.write("On isole nos modèles choisis pour optimisation")

    p_predictions3 = figure(width=1000, height=600, title = "Déviation de la température terrestre au cours des années avec biais")

    p_predictions3.line(x = df_global['Year'], y = df_global['J-D'])
    p_predictions3.line(x = range(2023,2051,1), y = predictions_ridge + 0.5, color='orange',legend_label='Ridge')
    p_predictions3.line(x = range(2023,2051,1), y = predictions_ridge_opti + 0.5, color='orange',width=2,legend_label='Ridge Opti')
    p_predictions3.line(x = range(2023,2051,1), y = predictions_xgb + 0.5, color='grey',legend_label='XGB')
    p_predictions3.line(x = range(2023,2051,1), y = predictions_xgb_opti + 0.5, color='grey',width=2,legend_label='XGB Opti')
    p_predictions3.legend.click_policy="hide"
    p_predictions3.yaxis.axis_label = "Déviation en °C"
    p_predictions3.xaxis.axis_label = "Année"
    p_predictions3_span = Span(location=0,dimension='width',line_width=2,line_color='grey')
    p_predictions3.add_layout(p_predictions3_span)
    p_predictions3.add_layout(p_predictions3.legend[0], 'right')
    doc.add_root(p_predictions3)
    st.bokeh_chart(p_predictions3,use_container_width=True)

    st.header("Résultats de nos prédictions")
    predictions = {'Linear Regressor':[predictions_linear[-1]+0.5],
                'Decision Tree':[predictions_decisiontree[-1]+0.5],
                'Random Forest':[predictions_forest[-1]+0.5],
                'Lasso':[predictions_lasso[-1]+0.5],
                'Ridge':[predictions_ridge[-1]+0.5],
                'Ridge Opti':[predictions_ridge_opti[-1]+0.5],
                'ElasticNet':[predictions_elasticnet[-1]+0.5]
    }
    df_predictions = pd.DataFrame(data = predictions)
    df_predictions.index = ['Déviation de la température terrestre en °C en 2050 ']
    st.dataframe(df_predictions)

if page == pages[6]:

  linear_model = load('joblib/linear_model.joblib')
  decisiontree_model = load('joblib/decisiontree_model.joblib')
  randomforest_model = load('joblib/randomforest_model.joblib')
  lasso_model = load('joblib/lasso_model.joblib')
  ridge_model = load('joblib/ridge_model.joblib')
  ridge_model_opti = load('joblib/ridge_opti.joblib')
  elasticnet_model = load('joblib/elasticnet_model.joblib')

  predictions_linear = linear_model.predict(df_test_scaled)
  predictions_decisiontree = decisiontree_model.predict(df_test_scaled)
  predictions_forest = randomforest_model.predict(df_test_scaled)
  predictions_lasso = lasso_model.predict(df_test_scaled)
  predictions_ridge = ridge_model.predict(df_test_scaled)
  predictions_ridge_opti = ridge_model_opti.predict(df_test_scaled)

  predictions_elasticnet = elasticnet_model.predict(df_test_scaled)


  st.title("Conclusion") 
  st.write("En qualité de Data Analyst, nous avons appliqué les étapes nécessaires à la bonne analyse de données :")
  col1, col2 = st.columns([1,40])
  with col2:
    st.write("• Recueil des données")
    st.write("• Extraction des données")
    st.write("• Analyse et Prédictions")
    st.write("• Communication et recommandations stratégiques")
  
  st.subheader("Prédictions")
  
  st.write("Il apparaît clairement dans notre analyse que les températures mondiales augmenteront à horizon 2050, de 1.4 degrés celsius. Ce qui est cohérent avec les prédictions des scientifiques et notamment celles du GIEC.")
  st.write("[Site du gouvernement Français, rapport du GIEC](https://www.ecologie.gouv.fr/hausse-temperature-globale-sest-encore-accentuee-selon-dernier-rapport-du-giec#:~:text=Le%20GIEC%20constate%20que%20la,pr%C3%A9industrielle%20entre%202021%20et%202040.)")
  st.image("pics/giec.png")

  st.header("Ouvertures")
  st.write("A l’heure actuelle, les solutions généralement évoquées pour lutter contre ce dérèglement climatique sont :")
  col1, col2 = st.columns([1,40])
  with col2:
    st.write("• Lutte contre la déforestation")
    st.write("• Réduction de la consommation énergétique")
    st.write("• Préservation des océans")
    st.write("• Et autres...")
  st.subheader("Saurons-nous renverser ces prédictions alarmistes grâce à des actions globales ?")
  st.subheader("Et comment le big data nous permettra de prévoir des événements climatiques et réduire les victimes de catastrophes ?")
