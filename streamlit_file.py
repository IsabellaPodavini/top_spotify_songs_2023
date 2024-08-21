# import librearies
import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import folium 
import io
from io import BytesIO
#from streamlit_folium import st_folium

##################
#Importing Dataset
##################
spotify_songs_df = pd.read_csv('spotify-2023.csv', encoding='latin-1')

##########
#set tabs for the chapters 
##########

tab_names = ["📄 Introduction", "🗑️ Cleaning", "🔗 Correlation", "📊 Exploratory Data Analysis", "🤖 Modeling with ML algorithms"]
current_tab = st.sidebar.selectbox('Summary', tab_names)
st.sidebar.markdown(
    """
    **Isabella Podavini** \n
    My page on GitHub: [GitHub](https://github.com/isabellapodavini)   
    My LinkedIn profile: [LinkedIn](linkedin.com/in/isabella-podavini)
    """
)

######################
#Functions and dataset
######################
    
def clean_data(df):
     
    cleaned_df = df.copy()
    
    # columns
    cleaned_df.columns = cleaned_df.columns.map(lambda x: x.lower().replace(' ', '_'))
    
    # mode
    cleaned_df['in_shazam_charts'] = cleaned_df['in_shazam_charts'].fillna(cleaned_df['in_shazam_charts'].mode()[0], inplace=True)
    cleaned_df['key'] = cleaned_df['in_shazam_charts'].fillna(cleaned_df['key'].mode()[0], inplace=True)
    
    #Incorrect values
    cleaned_df['streams'] = pd.to_numeric(cleaned_df['streams'], errors='coerce')
    mean_streams = cleaned_df['streams'].mean()
    cleaned_df['streams'].fillna(mean_streams, inplace=True)
    
    cleaned_df['in_deezer_playlists'] = pd.to_numeric(cleaned_df['in_deezer_playlists'], errors='coerce')
    mean_in_deezer_playlists = cleaned_df['in_deezer_playlists'].mean()
    cleaned_df['in_deezer_playlists'].fillna(mean_in_deezer_playlists, inplace=True)
    
    cleaned_df=cleaned_df.replace('#NAME?', np.nan)
    
    return cleaned_df

##########introduction
if current_tab == '📄 Introduction':
    st.markdown('<h1 style = "text-align: center;"> Top songs in 2023 </h1>', unsafe_allow_html = True)
    st.subheader('Programming for Data Science: Final Project')
    st.markdown('''
                **Author:** Isabella Podavini
                ''')
    st.markdown('''
                The dataset is about the best songs in 2023 registrated in different platforms as Spotify, Apple Music, Shazam and Deezer. \n 
                **Data Source:** https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023
                ''')

    
    selected_columns = st.multiselect('Explore the dataset by selecting columns', spotify_songs_df.columns)
    if selected_columns:
        columns_df = spotify_songs_df.loc[:, selected_columns]
        st.dataframe(columns_df.head(15))
    else: 
        st.dataframe(spotify_songs_df.head(15))
        
    
    st.write('General informations about the DataFrame')
    # Creating a buffer to capture information on the Airbnb DataFrame
    buffer = io.StringIO()
    spotify_songs_df.info(buf=buffer)
    s = buffer.getvalue()
    # Show multiselect to select columns to display
    selected_columns1 = st.multiselect("Select the variables", spotify_songs_df.columns.tolist(), default=spotify_songs_df.columns.tolist())

    # If columns are selected, it shows information only for those columns
    if selected_columns1:
        selected_info_buffer = io.StringIO()
        spotify_songs_df[selected_columns1].info(buf=selected_info_buffer)
        selected_info = selected_info_buffer.getvalue()
        st.text(selected_info)
    else:
        # Otherwise, it shows the information for all columns
        st.text(s)


##########cleaning        
elif current_tab == "🗑️ Cleaning":
    st.title("Cleaning NA values")
   
    st.write('Before proceeding with the analysis, the null values in the dataset were analyzed and then replaced or eliminated.')

    cleaned_df = clean_data(spotify_songs_df)

    tab1, tab2, tab3, tab4 = st.tabs(["NA values", "Cleaning", "-", "-"]) 
    
    with tab1:
    #calculates the count of missing values and the percentage of missing values for each variable
        missing_values_count = spotify_songs_df.isna().sum()
        total_values = spotify_songs_df.shape[0]
        missing_values_percentage = (missing_values_count / total_values) * 100
    
        # Round the percentage to two decimal places
        missing_values_percentage = missing_values_percentage.round(2)
            
        #create a new DataFrame with the count and percentage of missing values 
        missing_df = pd.DataFrame({
            'Variable': missing_values_count.index,
            'NA Values': missing_values_count.values,
            '% NA values': missing_values_percentage.values
        })
        
        #show the DataFrame of missing values 
        st.write(missing_df)
    with tab2:  
        st.markdown('''
                In this case, since the 'in_shazam_charts' and 'key' columns contain categorical variables and not numeric variables, it would not be correct to replace missing values with mathematical operations such as mode or median. So the null values have been dropped.
                ''')
        
        with st.expander('Resulted DataFrame Preview'):
            st.write(cleaned_df.head(15)) 


############
#Correlation
############
elif current_tab == "🔗 Correlation": 
    st.title("Correlations between values")
    
    # Pulisci i dati
    cleaned_df = clean_data(spotify_songs_df)
    
    st.write("A preliminary graphical analysis through heatmap proved useful in exploring the correlations between numerical variables in the dataset.")
    #Select only the columns of audio features
    audio_features = cleaned_df[['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']]
    #Calculates the correlation between audio characteristics
    correlation_matrix = audio_features.corr()
    
    # Heatmap 
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Audio Features')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    st.pyplot(plt.gcf())
    
    st.markdown("""
    It can be seen that "acousticness_%" and "energy_%" have a negative correlation, so it indicates that acoustic songs tend to have a lower energy level.
    While it can be seen that "valence_%" and "energy_%" have a strong positive correlation, so songs with a high positivity level also tend to have a high energy level.
    For example:
    * Acousticness - Energy: -0.58
    * Valence - Energy: 0.36
    * Valence - Danceability: 0.41
    """)
    
    ##########
    #Function
    ##########   
    #Define the plot_scatter function 
    def plot_scatter(cleaned_df):
        plt.figure(figsize=(6, 3))
        sns.set_palette("husl")
        sns.scatterplot(data=cleaned_df, x='service_fee', y='price', color='purple')
        plt.xlabel('Service Fee')
        plt.ylabel('Price')
        plt.title('Relationship between Service Fee and Price')
        st.pyplot(plt.gcf())  # 

    # Initializes or updates the state when the button is pressed
    if 'show_scatter' not in st.session_state:
        st.session_state.show_scatter = False

    if st.button('Click to see the Scatterplot'):
        st.session_state.show_scatter = not st.session_state.show_scatter

    
    # Shows or hides the graph based on the state of the session
    if st.session_state.show_scatter:
        plot_scatter(cleaned_df)  
    
    st.divider() 
    st.write('To observe how the main categorical variables relate to each other, contingency tables were used, which help identify areas of higher or lower frequency of combinations between types of variables:')

    tab1, tab2, tab3 = st.tabs(["Acousticness vs Energy", "Valence vs Energy", "Valence vs Danceability"]) 
    
    with tab1:
        contingency_table = pd.crosstab(cleaned_df['acousticness_%'], cleaned_df['energy_%'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(contingency_table, annot=True, cmap='BuPu', fmt='d')
        plt.xlabel('Acousticness')
        plt.ylabel('Energy')
        st.pyplot(plt.gcf())
            
        st.markdown('''
                    It can be seen that "acousticness_%" and "energy_%" have a negative correlation, so it indicates that acoustic songs tend to have a lower energy level.
                    ''')
    
    with tab2:
        contingency_table = pd.crosstab(cleaned_df['valence_%'], cleaned_df['energy_%'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(contingency_table, annot=True, cmap='BuPu', fmt='d')
        plt.xlabel('Valence')
        plt.ylabel('Energy')
        st.pyplot(plt.gcf())
            
        st.markdown('''
                    It can be seen that "valence_%" and "energy_%" have a strong positive correlation, so songs with a high positivity level also tend to have a high energy level
                    ''')
    
    with tab3:
        contingency_table = pd.crosstab(cleaned_df['valence_%'], cleaned_df['danceability_%'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(contingency_table, annot=True, cmap='BuPu', fmt='d')
        plt.xlabel('Valence')
        plt.ylabel('Danceability')
        st.pyplot(plt.gcf())
            
        st.markdown('''
                    It can be seen that "valence_%" and "danceability_%" have a strong positive correlation, so songs with a high positivity level also tend to have a high danceability level
                    ''')
    
    #fig_1 = plt.figure(figsize=(20, 16))
    #ax_1 = fig_1.add_subplot(2, 2, 1)
    #ax_2 = fig_1.add_subplot(2, 2, 2)
        
    #ax_1.scatter(cleaned_df['energy_%'],cleaned_df['acousticness_%'])
    #ax_1.title.set_text('Relation between Energy - Acousticness')
        
    #ax_2.scatter(cleaned_df['energy_%'],cleaned_df['valence_%'])
    #ax_2.title.set_text('Relation between Energy - Valence')
    
    #st.pyplot(fig_1)