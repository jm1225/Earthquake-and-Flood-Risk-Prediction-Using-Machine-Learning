import streamlit as st
from PIL import Image
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=UserWarning)
#importing dataset
df = pd.read_csv("Guwahati.csv")
city_map= Image.open("city_map2_guwahati.jpg")

#SVM-ML Model1

df['min_dist'] = df[['prox_kopili','prox_hft','prox_epicenter','prox_sp']].min(axis=1)
bins = [-0.000001, 1, float("inf")]
labels = ['High', 'Low']
df['Zone'] = pd.cut(df['min_dist'], bins=bins, labels=labels)
features = df[['Latitude', 'Longitude', 'prox_kopili','prox_hft','prox_epicenter','prox_sp']]
labels = df['Zone']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

#SVM-ML Model2

df['min_dist2'] = df[['prox_bhramaputra','prox_diporbil','prox_bharalu']].min(axis=1)
bins2 = [-0.000001, 1, float("inf")]
labels2 = ['High', 'Low']
df['Zone2'] = pd.cut(df['min_dist2'], bins=bins2, labels=labels2)
features2 = df[['Latitude', 'Longitude', 'prox_bhramaputra','prox_diporbil','prox_bharalu']]
labels2 = df['Zone2']
X_train2, X_test2, y_train2, y_test2 = train_test_split(features2, labels2, test_size=0.2, random_state=42)
scaler2 = StandardScaler()
X_train2 = scaler2.fit_transform(X_train2)
X_test2=scaler2.transform(X_test2)
svm_model2 = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model2.fit(X_train2, y_train2)



#Proximity distance formula
def dist(lat1, lon1, lati, long):
    lat1, lon1, lati, long = map(math.radians,[lat1, lon1, lati, long])
    dlon = long - lon1
    dlat = lati - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1)*math.cos(lati)*math.sin(dlon/2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371.0
    distance = r * c
    return distance

def main():
    st.write("<h1 style='text-align: center;'>Earthquake Risk Predictor Using GPS Coordinates</h1>", unsafe_allow_html=True)
    st.write("<h2 style='text-align: center;'>FOR GUWAHATI</h2>", unsafe_allow_html=True)

    lati_str = st.text_input("Enter Latitude:")
    long_str = st.text_input("Enter Longitude:")     
    lati = round(float(lati_str), 9) if lati_str else None
    long = round(float(long_str), 9) if long_str else None
    
     
    if lati is not None and long is not None:
        if 26.069366<=lati<=26.217928 and 91.529162<= long<=91.875617:

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_title('Selected Coordinate')
            image_path = 'marker.png'
            image = OffsetImage(plt.imread(image_path), zoom=0.01)
            ab = AnnotationBbox(image, (long, lati), frameon=False)
            ax.add_artist(ab)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True)
            ax.imshow(city_map, extent=[91.529162, 91.875617,26.069366,26.217928])

            st.pyplot(fig)

            #nearest coord and distance from bhramputra
            if 26.135623<=lati<=26.215799:
                lat1=lati
            else:
                if lati<26.135623:
                    lat1=26.135623
                else:
                    lat1=26.215799
            if 91.557963<=long<=91.840108:
                lon1=long
            else:
                if long<91.557963:
                    lon1=91.557963
                else:
                    lon1=91.840108  
            d_bhramputra=dist(lat1, lon1, lati, long)
            
            #nearest coord and distance from diporbil     
            if 26.102514<=lati<=26.140576:
                lat1=lati
            else:
                if lati<26.102514:
                    lat1=26.102514
                else:
                    lat1=26.140576
            if 91.615154<=long<=91.680987:
                lon1=long
            else:
                if long<91.615154:
                    lon1=91.615154
                else:
                    lon1=91.680987

            d_diporbil=dist(lat1, lon1, lati, long)
            
            #nearest coord and distance from bharalu     
            if 26.165939<=lati<=26.167230:
                lat1=lati
            else:
                if lati<26.165939:
                    lat1=26.165939
                else:
                    lat1=26.167230
            if 91.731077<=long<=91.779473:
                lon1=long
            else:
                if long<91.731077:
                    lon1=91.731077
                else:
                    lon1=91.779473
            d_bharalu=dist(lat1, lon1, lati, long)
            
            #nearest coord and distance from kopili     
            if 25.75345<=lati<=27:
                lat1=lati
            else:
                if lati<25.75345:
                    lat1=25.75345
                else:
                    lat1=27
            if 92.1923434<=long<=93:
                lon1=long
            else:
                if long<92.1923434:
                    lon1=92.1923434
                else:
                    lon1=93   
            d_kopili=dist(lat1, lon1, lati, long)

            #nearest coord and distance from epicenter     
            lat1=26.690
            lon1=92.360
            d_epicenter=dist(lat1, lon1, lati, long)

            #nearest coord and distance from hft     
            lat1=26.830087
            if 91.529162<=long<=91.875617:
                lon1=long
            else:
                if long<91.529162:
                    lon1=91.529162
                else:
                    lon1=91.875617 

            d_hft=dist(lat1, lon1, lati, long)

            #nearest coord and distance from sp     
            if 26.065325<=lati<=26.111384:
                lat1=lati
            else:
                if lati<26.065325:
                    lat1=26.065325
                else:
                    lat1=26.111384
            if 91.741400<=long<=91.855663:
                lon1=long
            else:
                if long<91.741400:
                    lon1=91.741400
                else:
                    lon1=91.855663  

            d_sp=dist(lat1, lon1, lati, long)

            new_data = pd.DataFrame({
                'Latitude': [lati],
                'Longitude': [long],
                'prox_kopili': [d_kopili],
                'prox_hft': [d_hft],
                'prox_epicenter': [d_epicenter],
                'prox_sp': [d_sp]
            })

            new_data2 = pd.DataFrame({
                'Latitude': [lati],
                'Longitude': [long],
                'prox_bhramaputra': [d_bhramputra],
                'prox_diporbil': [d_diporbil],
                'prox_bharalu': [d_bharalu]
            })
        
            new_data_scaled1 = scaler.transform(new_data)
            new_data_scaled2 = scaler2.transform(new_data2[['Latitude', 'Longitude', 'prox_bhramaputra', 'prox_diporbil', 'prox_bharalu']])
            predicted_risk = svm_model.predict(new_data_scaled1)
            predicted_risk2 = svm_model2.predict(new_data_scaled2)
            st.markdown(f"<p style='font-size: 30px; text-align: center;'>Predicted Earthquake Epicenter Risk: <b><i>{predicted_risk[0]}</i></b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 30px; text-align: center;'>Predicted Flood Risk: <b><i>{predicted_risk2[0]}</i></b></p>", unsafe_allow_html=True)
            st.write("<p style='text-align: center;'>Wanna predict in danger areas if this coordinate become an epicenter?</h1>", unsafe_allow_html=True)
            k = 0
            mag_str = st.text_input("Enter the Magnitude of Earthquake from the Epicenter:")
            mag = round(float(mag_str), 1) if mag_str else None
            if st.button("Predict"):            
                st.subheader("Prediction Result:")
                if mag is not None:
                    if 0 <= mag <= 5.5:
                        if predicted_risk[0] == "High":
                            k = 3
                            st.markdown(f"{mag} magnitude of Earthquake is COMMON and HARMLESS, but since it falls in the High Risk Zone, these cities need to be observed.")
                        else:
                            st.markdown(f"{mag} magnitude of Earthquake is COMMON and HARMLESS under Low Risk Zone, so no need to worry.")
                            return
                    elif 5.5 < mag <= 6:
                        st.markdown("That's a MODERATE shock with slight damage to infrastructure of these Areas.")
                        k = 4
                    elif 6.0 < mag <= 8.0:
                        st.markdown("That's a GREAT shock with high damage in these areas:")
                        k =6
                    elif 8 < mag <= 9:
                        st.markdown("Major Earthquake ALERT!! Serious damage expected in these Areas.")
                        k = 10
                    elif mag > 9:
                        st.markdown("Bye Bye...This magnitude will collapse human civilisation here for good. See you in next Life in another world :)")
                        if st.button("Exit"):
                            st.write("Thank You for the visit! :)")
                        return
                                
                    else:
                        st.markdown("Invalid Magnitude! Try Again.")
                        return 
                                
                    input_coord = [lati, long]
                    nn_model = NearestNeighbors(n_neighbors=k, metric='haversine')
                    nn_model.fit(np.radians(df[['Latitude', 'Longitude']]))
                    distances, indices = nn_model.kneighbors([np.radians(input_coord)])
                    nearest_coords = df.iloc[indices[0]][['Latitude', 'Longitude']]
                    area_names = df.iloc[indices[0]]['Area_Name']
                    nearest_df = pd.DataFrame({'Area_Name': area_names.values, 'Latitude': nearest_coords['Latitude'].values, 'Longitude': nearest_coords['Longitude'].values})
                    st.subheader(f"Names of Nearest {k} Indanger Areas:")
                    st.write(nearest_df)

                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.scatter(nearest_df['Longitude'], nearest_df['Latitude'], c='b', marker='*', label='Nearest in Danger')
                    ax.scatter(long, lati, marker='o', c='r', label='Selected Coordinate/ Epicenter')
                    ax.set_title('Epicenter and Indanger Areas')
                    image_path = 'marker.png'
                    image = OffsetImage(plt.imread(image_path), zoom=0.01)
                    ab = AnnotationBbox(image, (long, lati), frameon=False)
                    ax.add_artist(ab)
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')
                    ax.legend(loc='upper left')
                    ax.grid(True)
                    ax.imshow(city_map, extent=[91.529162, 91.875617,26.069366,26.217928])

                    st.pyplot(fig)              
                
                    if st.button("Exit"):
                        st.write("Thank You for the visit! :)")
                                    
                else:
                    st.write("Enter the magnitude...") 
                    
            elif st.button("Exit"):
                st.write("Thank You for the visit! :)")
                    
            else:
                st.write("Click on either above of them to continue.")

        else:
            st.write("These coordinated are out of Guwahati. Please enter in the range.")
            st.write("For Your reference, Guwahati falls inside:")
            st.write("Latitude: 26.069366 - 26.217928")
            st.write("Longitude: 91.529162 - 91.875617")

    else:
        st.write("Enter the coordinates above.")
        st.write("For Your reference, Guwahati falls inside:")
        st.write("Latitude: 26.069366 - 26.217928")
        st.write("Longitude: 91.529162 - 91.875617")
        return

    
       
if __name__ == "__main__":
    main()