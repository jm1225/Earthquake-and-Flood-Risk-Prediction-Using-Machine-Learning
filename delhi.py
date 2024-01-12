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
df = pd.read_csv("Delhi.csv")
city_map= Image.open("city_map3.jpg")

#SVM-ML Model(epicenter risk)
df['min_dist'] = df[['prox_Sohna', 'prox_Mathura', 'prox_Moradabad']].min(axis=1)
bins = [-0.000001, 2, float("inf")]
labels = ['High', 'Low']
df['Zone'] = pd.cut(df['min_dist'], bins=bins, labels=labels)
features = df[['Latitude', 'Longitude','prox_Sohna', 'prox_Mathura', 'prox_Moradabad']]
labels = df['Zone']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

#SVM-ML Model2(flood risk)
bins2 = [-0.000001, 5, float("inf")]
labels2 = ['High', 'Low']
df['Zone2'] = pd.cut(df['proximity_yamuna'], bins=bins2, labels=labels2)
features2 = df[['Latitude', 'Longitude', 'proximity_yamuna']]
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
    st.write("<h2 style='text-align: center;'>FOR DELHI</h2>", unsafe_allow_html=True)

    lati_str = st.text_input("Enter Latitude:")
    long_str = st.text_input("Enter Longitude:")     
    lati = round(float(lati_str), 9) if lati_str else None
    long = round(float(long_str), 9) if long_str else None
    
    

    if lati is not None and long is not None:
        if 28.406288<=lati<=28.88565 and 76.839081<= long<=77.345843:

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_title('Selected Coordinate')
            image_path = 'marker.png'
            image = OffsetImage(plt.imread(image_path), zoom=0.01)
            ab = AnnotationBbox(image, (long, lati), frameon=False)
            ax.add_artist(ab)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True)
            ax.imshow(city_map, extent=[76.839081, 77.345843, 28.406288, 28.88565])

            st.pyplot(fig) 
            
            if 28.57047<=lati<=28.8333:
                lat1=lati
            else:
                if lati<28.57047:
                    lat1=28.57047
                else:
                    lat1=28.8333
                    
            if 77.2<=long<=77.3:
                lon1=long
            else:
                if long<77.2:
                    lon1=77.2
                else:
                    lon1=77.3
            y_distance=dist(lat1, lon1, lati, long)
            
            #nearest coord and distance from sohna     
            if 28.4000<=lati<=28.7000:
                lat1=lati
            else:
                if lati<28.4:
                    lat1=28.4
                else:
                    lat1=28.7
            lon1=77.00

            s_distance=dist(lat1, lon1, lati, long)
            
            #nearest coord and distance from moradabad     
            if 28.6000<=lati<=28.7000:
                lat1=lati
            else:
                if lati<28.6:
                    lat1=28.6
                else:
                    lat1=28.7
                    
            if 77.2000<=long<=77.3200:
                lon1=long
            else:
                if long<77.2:
                    lon1=77.2
                else:
                    lon1=77.32
            dm_distance=dist(lat1, lon1, lati, long)
            
            #nearest coord and distance from mathura     
            if 28.5000<=lati<=28.5000:
                lat1=lati
            else:
                if lati<28.5:
                    lat1=28.5
                else:
                    lat1=28.6
                    
            if 77.1000<=long<=77.2000:
                lon1=long
            else:
                if long<77.1:
                    lon1=77.1
                else:
                    lon1=77.2
            m_distance=dist(lat1, lon1, lati, long)

            new_data = pd.DataFrame({
                'Latitude': [lati],
                'Longitude': [long],
                'prox_Sohna': [s_distance],
                'prox_Mathura': [m_distance],
                'prox_Moradabad': [dm_distance]
            })
            new_data2 = pd.DataFrame({
                'Latitude': [lati],
                'Longitude': [long],
                'proximity_yamuna':[y_distance]
            })
        
            new_data_scaled = scaler.transform(new_data)
            predicted_zone = svm_model.predict(new_data_scaled)
            new_data_scaled2 = scaler2.transform(new_data2)
            predicted_zone2 = svm_model2.predict(new_data_scaled2)
            
            st.markdown(f"<p style='font-size: 30px; text-align: center;'>Predicted Earthquake Epicenter Risk: <b><i>{predicted_zone[0]}</i></b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 30px; text-align: center;'>Predicted Flood Risk: <b><i>{predicted_zone2[0]}</i></b></p>", unsafe_allow_html=True)
            
            st.write("<p style='text-align: center;'>Wanna predict in danger areas if this coordinate become an epicenter?</h1>", unsafe_allow_html=True)
            
            k = 0
            mag_str = st.text_input("Enter the Magnitude of Earthquake from the Epicenter:")
            mag = round(float(mag_str), 1) if mag_str else None
            if st.button("Predict"):            
                st.subheader("Prediction Result:")
                if mag is not None:
                    if 0 <= mag <= 5.5:
                        if predicted_zone[0] == "High":
                            k = 5
                            st.markdown(f"{mag} magnitude of Earthquake is COMMON and HARMLESS, but since it falls in High Risk Zone, these cities need to be observed.")
                        else:
                            st.markdown(f"{mag} magnitude of Earthquake is COMMON and HARMLESS under Low Risk Zone, so no need to worry.")
                            return
                    elif 5.5 < mag <= 6:
                        st.markdown("That's a MODERATE shock with slight damage to infrastructure of these Areas.")
                        k = 7
                    elif 6.0 < mag <= 7.0:
                        st.markdown("That's a GREAT shock with high damage in these areas:")
                        k = 8
                    elif 7 < mag <= 8.0:
                        st.markdown("Major Earthquake ALERT!! Serious damage expected in these Areas.")
                        k = 10
                    elif 8 < mag <= 9:
                        st.markdown("Major Earthquake ALERT!! Serious damage expected in these Areas.")
                        k = 15
                    elif mag > 9:
                        st.markdown("Bye Bye...This magnitude will collapse human civilisation here for good. See You in next Life in another world :)")
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
                    st.subheader(f"Names of Nearest {k} Effected Areas:")
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
                    ax.imshow(city_map, extent=[76.839081, 77.345843, 28.406288, 28.88565])

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
            st.write("These coordinates are out of Delhi. Please enter in range.")
            st.write("For Your reference, Delhi falls inside:")
            st.write("Latitude: 28.406288 - 28.88565")
            st.write("Longitude: 76.839081 - 77.345843")

        
    else:
        st.write("Enter the coordinates above.")
        st.write("For Your reference, Delhi falls inside:")
        st.write("Latitude: 28.406288 - 28.88565")
        st.write("Longitude: 76.839081 - 77.345843")
        return

    
if __name__ == "__main__":
    main()