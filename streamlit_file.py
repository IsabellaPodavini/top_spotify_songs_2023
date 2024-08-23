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
    
    cleaned_df = cleaned_df[cleaned_df['streams'] !='BPM110KeyAModeMajorDanceability53Valence75Energy69Acousticness7Instrumentalness0Liveness17Speechiness3']
    cleaned_df['streams'] = cleaned_df['streams'].astype('int64')
    cleaned_df['in_deezer_playlists'] = cleaned_df['in_deezer_playlists'].str.replace(',','').astype('int64')
    
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

    tab1, tab2 = st.tabs(["NA values", "Cleaning"]) 
    
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
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
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
    
    fig_1 = plt.figure(figsize=(20, 16))
    ax_1 = fig_1.add_subplot(2, 2, 1)
    ax_2 = fig_1.add_subplot(2, 2, 2)

    ax_1.scatter(cleaned_df['energy_%'], cleaned_df['acousticness_%'])
    ax_1.title.set_text('Relation between Energy - Acousticness')
    ax_1.set_xlabel('Energy %')
    ax_1.set_ylabel('Acousticness %')

    ax_2.scatter(cleaned_df['energy_%'], cleaned_df['valence_%'])
    ax_2.title.set_text('Relation between Energy - Valence')
    ax_2.set_xlabel('Energy %')
    ax_2.set_ylabel('Valence %')


    st.pyplot(fig_1)

    st.markdown('''
        It can be seen that "acousticness_%" and "energy_%" have a negative correlation, indicating that acoustic songs tend to have a lower energy level and also that "valence_%" and "energy_%" have a strong positive correlation, indicating that songs with a high positivity level also tend to have a high energy level.    
    ''')

############
#EDA
############
elif current_tab == "📊 Exploratory Data Analysis": 
    st.title("Exploratory Data Analysis")
    
    st.write('First of all, I offer a complementary view of the temporal distribution of music, exploring both the production and popularity of songs over time.')
    
    #tab1, tab2 = st.tabs(["Number of songs in each year", "Distribution of Streams in different Months"]) 
    
    cleaned_df = clean_data(spotify_songs_df)
    
    st.write('The following graph represents the trend in the number of songs released each year in the dataset. Each bar corresponds to a specific year and its height indicates how many songs were released in that year. This allows us to observe how music production has varied over time, highlighting any trends. By analyzing this graph, we can draw conclusions about the development and evolution of music over time, noting a greater trend in recent years.')
    count_by_year = cleaned_df['released_year'].value_counts().sort_index()
    plt.figure(figsize=(10,6))
    count_by_year.plot(kind='bar', color='orange')
    plt.title('Number of songs released by year')
    plt.xlabel('Released year')
    plt.ylabel('Number of songs')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7) #grid only for axis y, style: fotted line, trasparency: 70%
    plt.tight_layout() #Automatically adjusts the position of the graph axes so that they do not overlap and the content is evenly distributed
    st.pyplot(plt.gcf())
    
    ##############################
    st.write('We can also examine the distribution of music streams in different months of the year. Using this dataset, this bar graph shows how the number of streams varies according to the month of release. Each bar represents a specific month and its height indicates the average amount of streams received by the songs released in that month. Through this graph, we can identify any seasonality or trends that affect the popularity of songs throughout the year')
    MonthDict={ 1 : "January",
            2 : "February",
            3 : "March",
            4 : "April",
            5 : "May",
            6 : "June",
            7 : "July",
            8 : "August",
            9 : "September",
           10 : "October",
           11 : "November",
           12 : "December"
        }
    #cleaned_df['month_name'] = cleaned_df['released_month'].map(MonthDict)
    #order = list(MonthDict.values())
    sns.barplot(x='released_month', y='streams', data=cleaned_df, palette="coolwarm")
    plt.title('Distribution of Streams Across Different Months')
    plt.xlabel('Month')
    plt.ylabel('Streams')
    st.pyplot(plt.gcf())
    
    #########################à
    st.write('The number of artists in this dataset is very large, the top 10 who made a larger number of songs are:')
    artist_counts = cleaned_df['artist(s)_name'].value_counts().head(10)
    plt.figure(figsize=(12,6))
    sns.barplot(x=artist_counts.values,y=artist_counts.index,palette='viridis')
    plt.xlabel('No. of songs')
    plt.ylabel('Artist(s)')
    plt.title('Top 10 artist with most number of songs')
    st.pyplot(plt.gcf())
    st.write('The singer(s) with the highest number is Taylor Swift')
    
    ##############################
    st.write('The top 10 songs with the most streams are:')
    song_streamh = cleaned_df[['track_name','artist(s)_name','released_year','streams']].\
               sort_values(by = 'streams',ascending=False)
    song_count = song_streamh.head(10)
    plt.figure(figsize=(12,6))
    sns.barplot(x=song_count.streams,y=song_count.track_name,palette='magma')
    plt.xlabel('Streams(in billions)')
    plt.ylabel('Track Name')
    plt.title('Top 10 song with most stream hour')
    st.pyplot(plt.gcf())
    
    st.write('The song with highest strems is: Blinding Lights. The song with lowest streams is: Que Vuelvas')
    
    ##############################
    st.write('Each song can be played by multiple artists')
    artist_count_counts = cleaned_df['artist_count'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.bar(artist_count_counts.index, artist_count_counts.values, color='blue')
    plt.xlabel('Number of Artists Involved')
    plt.ylabel('Number of Songs')
    plt.title('Distribution of Songs by Number of Artists Involved')
    plt.xticks(artist_count_counts.index)
    st.pyplot(plt.gcf())
    
    st.write('In a song, the maximum number of artists involved is 8 and the minimum number is 1. It can be seen from the graph that most songs involve only one artist.')
    
    ################################
    st.write('We can now analyze some important tools ')
    
    tab1, tab2, tab3 =st.tabs(["Distribution of BPM", "Distribution of Keys", "Distribution of Mode"])
    with tab1:
        plt.figure(figsize=(8, 6))
        sns.histplot(cleaned_df['bpm'], bins=20, kde=True, color='skyblue')
        plt.xlabel('BPM', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        st.pyplot(plt.gcf())
    
    with tab2:
        plt.figure(figsize=(8, 6))
        sns.countplot(x="key", data=cleaned_df, palette="Set2")
        plt.xlabel("Keys", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        st.pyplot(plt.gcf())
    
    with tab3:
        cleaned_df['mode'].value_counts().plot.pie(autopct = '%1.2f%%', legend = True);
        plt.tight_layout()
        st.pyplot(plt.gcf())
