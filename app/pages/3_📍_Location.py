import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import folium
from folium.plugins import MarkerCluster
from folium.plugins import Search
from streamlit_folium import st_folium

import os

cwd = os.path.dirname(__file__)

def load_map(cluster, df):
    X = df.iloc[:, -2:].values
    kmeans = KMeans(
        n_clusters = cluster,
        init = 'k-means++',
        random_state = 42
    )
    y_kmeans = kmeans.fit_predict(X)
    # y_kmeans
    df['cluster'] = y_kmeans + 1
    m = folium.Map(
        [
            df['Latitude'].mean(),
            df['Longitude'].mean()
        ],
        zoom_start = 5.6
    )
    # Layers for cluster
    layer1 = folium.FeatureGroup(
        name= '<u><b>G1</b></u>',
        show= True
    )
    m.add_child(layer1)
    layer2 = folium.FeatureGroup(
        name= '<u><b>G2</b></u>',
        show= True
    )
    m.add_child(layer2)
    layer3 = folium.FeatureGroup(
        name= '<u><b>G3</b></u>',
        show= True
    )
    m.add_child(layer3)
    layer4 = folium.FeatureGroup(
        name= '<u><b>G4</b></u>',
        show= True
    )
    m.add_child(layer4)
    layer5 = folium.FeatureGroup(
        name= '<u><b>G5</b></u>',
        show= True
    )
    m.add_child(layer5)
    layer6 = folium.FeatureGroup(
        name= '<u><b>G6</b></u>',
        show= True
    )
    m.add_child(layer6)
    layer7 = folium.FeatureGroup(
        name= '<u><b>G7</b></u>',
        show= True
    )
    m.add_child(layer7)

    my_symbol_css_class= """ <style>
        .fa-g1:before {
            font-family: Arial; 
            font-weight: bold;
            font-size: 8px;
            color: black;
            background-color:white;
            border-radius: 10px; 
            white-space: pre;
            content: ' G1 ';
            }
        .fa-g2:before {
            font-family: Arial; 
            font-weight: bold;
            font-size: 8px;
            color: black;
            background-color:white;
            border-radius: 10px; 
            white-space: pre;
            content: ' G2 ';
            }
        .fa-g3:before {
            font-family: Arial; 
            font-weight: bold;
            font-size: 8px;
            color: black;
            background-color:white;
            border-radius: 10px; 
            white-space: pre;
            content: ' G3 ';
            }
        .fa-g4:before {
            font-family: Arial; 
            font-weight: bold;
            font-size: 8px;
            color: black;
            background-color:white;
            border-radius: 10px; 
            white-space: pre;
            content: ' G4 ';
            }
            .fa-g5:before {
            font-family: Arial; 
            font-weight: bold;
            font-size: 8px;
            color: black;
            background-color:white;
            border-radius: 10px; 
            white-space: pre;
            content: ' G5 ';
            }
            .fa-g6:before {
            font-family: Arial; 
            font-weight: bold;
            font-size: 8px;
            color: black;
            background-color:white;
            border-radius: 10px; 
            white-space: pre;
            content: ' G6 ';
            }
            .fa-g7:before {
            font-family: Arial; 
            font-weight: bold;
            font-size: 8px;
            color: black;
            background-color:white;
            border-radius: 10px; 
            white-space: pre;
            content: ' G7 ';
            }
        </style>
    """
    # the below is just add above  CSS class to folium root map      
    m.get_root().html.add_child(
        folium.Element(
            my_symbol_css_class
        )
    )
    # then we just create marker and specific your css class in icon like below
    for index, row in df.iterrows():
        if row['cluster'] == 1 :
            color='black'
            fa_symbol = 'fa-g1'
            lay = layer1
        elif row['cluster'] == 2:
            color='purple'
            fa_symbol = 'fa-g2'
            lay = layer2     
        elif row['cluster'] == 3:
            color='orange'
            fa_symbol = 'fa-g3'
            lay = layer3
        elif row['cluster'] == 4:
            color='blue'
            fa_symbol = 'fa-g4'
            lay = layer4
        elif row['cluster'] == 5:
            color='green'
            fa_symbol = 'fa-g5'
            lay = layer4
        elif row['cluster'] == 6:
            color='gray'
            fa_symbol = 'fa-g6'
            lay = layer4
        elif row['cluster'] == 7:
            color='yellow'
            fa_symbol = 'fa-g7'
            lay = layer4
            
        folium.Marker(
            location=[
                row['Latitude'],
                row['Longitude']
            ],
            title = row['Store Name']+ ' group:{}'.format(str(row['cluster'])),
            popup = row['Store Name']+ ' group:{}'.format(str(row['cluster'])),
            icon= folium.Icon(color=color, icon=fa_symbol, prefix='fa')
        ).add_to(lay)
        
        
    layer_list = [
        layer1,
        layer2,
        layer3,
        layer4,
        layer5,
        layer6,
        layer7
    ]
    color_list = [
        'black',
        'purple',
        'orange',
        'blue',
        'green',
        'beige',
        'red'
    ]
    for g in df['cluster'].unique():
    # this part we apply ConvexHull theory to find the boundary of each group
        # first, we have to cut the lat lon in each group 
        latlon_cut = df[
            df['cluster'] == g
        ].iloc[:, -3:-1]

        hull = ConvexHull(latlon_cut.values)

        Lat = latlon_cut.values[hull.vertices,0]
        Long = latlon_cut.values[hull.vertices,1] 
        
        median_hq_lat = sum(Lat)/len(Lat)
        median_hq_long = sum(Long)/len(Long)

        cluster = pd.DataFrame({'lat':Lat,'lon':Long })       
        area = list(zip(cluster['lat'],cluster['lon']))
        # plot polygon
        list_index = g - 1 # minus 1 to get the same index 
        lay_cluster = layer_list[list_index] 
        
        bean_pin = folium.CustomIcon(
            icon_image= cwd + "/img/mrbeans.png",
            icon_size=(60, 60)
        )
        folium.Marker(
            location=[
                median_hq_long,
                median_hq_lat
            ],
            title = "Mr Beans HQ " + str(g),
            popup = "Mr Beans HQ " + str(g),
            icon= bean_pin
        ).add_to(lay_cluster)
        
        
        folium.Polygon(
            locations=area,
            color=color_list[list_index],
            weight=2,
            fill=True,
            fill_opacity=0.1,
            opacity=0.8
        ).add_to(lay_cluster) 

        # folium.LayerControl(collapsed=False,position= 'bottomright').add_to()
        
    st_folium(m, width=725, returned_objects=[])
        
        

def config():
    st.set_page_config(
        layout="wide",
        page_title="Mr Beans | Location",
        page_icon="🌍",
    )

    # Customize page title
    st.title("Location Optimizer ☕️")

    st.info(
        "Optimize where to build Mr Beans Hub"
    )


def render_view():
    df_malaysia = pd.read_csv(cwd + '/data/location/starbucks_malaysia.csv',encoding='latin-1')
    number = st.number_input(
        "Choose Cluster",
        min_value = 1,
        max_value = 7,
        value = 5
    )
    load_map(
        number,
        df_malaysia
    )

if __name__ == "__main__":
    config()
    render_view()