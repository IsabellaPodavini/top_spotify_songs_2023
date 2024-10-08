# import librearies
import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from wordcloud import WordCloud

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
    mode_value = cleaned_df['key'].mode()[0]
    cleaned_df['key'].fillna(mode_value, inplace=True)

    
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
        
    
    st.subheader('General informations about the DataFrame')
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
    
    st.subheader('Temporal distribution of music')
    st.write('First of all, I offer a complementary view of the temporal distribution of music, exploring both the production and popularity of songs over time.')
    
    tab1, tab2 = st.tabs(["Number of songs in each year", "Distribution of Streams in different Months"]) 
    
    cleaned_df = clean_data(spotify_songs_df)
    with tab1:
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
    with tab2:
        st.write('We can also examine the distribution of music streams in different months of the year. Using this dataset, this bar graph shows how the number of streams varies according to the month of release. Each bar represents a specific month and its height indicates the average amount of streams received by the songs released in that month. Through this graph, we can identify any seasonality or trends that affect the popularity of songs throughout the year')
        count_by_month = cleaned_df['released_month'].value_counts().sort_index()
        plt.figure(figsize=(10,6))
        count_by_month.plot(kind='bar', color='lightblue')
        plt.title('Distribution of Streams Across Different Months')
        plt.xlabel('Month')
        plt.ylabel('Streams')
        st.pyplot(plt.gcf())
    
    #########################
    
    st.subheader('The top10')
    st.write('The number of artists in this dataset is very large, the top 10 who made a larger number of songs are:')
    artist_counts = cleaned_df['artist(s)_name'].value_counts().head(10)
    plt.figure(figsize=(12,6))
    sns.barplot(x=artist_counts.values, y=artist_counts.index, hue=artist_counts.index, palette='viridis', legend=False)
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
    sns.barplot(x=song_count.streams,y=song_count.track_name,palette='magma', hue=song_count.track_name, dodge=False)
    plt.xlabel('Streams(in billions)')
    plt.ylabel('Track Name')
    plt.title('Top 10 song with most stream hour')
    st.pyplot(plt.gcf())
    
    st.write('The song with highest strems is: Blinding Lights. The song with lowest streams is: Que Vuelvas')
    
    ##############################
    
    st.subheader('Number of artists involved in each song')
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
    
    st.subheader('BPM, Key and Mode')
    st.write('We can now analyze some important tools:')
    
    tab1, tab2, tab3, tab4 =st.tabs(["Distribution of BPM", "Distribution of Keys", "Distribution of Mode", "Streams Distribution by Key and Mode"])
    with tab1:
        plt.figure(figsize=(8, 6))
        sns.histplot(cleaned_df['bpm'], bins=20, kde=True, color='lightcoral')
        plt.xlabel('BPM', fontsize=10)
        plt.ylabel('Count', fontsize=10)
        st.pyplot(plt.gcf())
    
    with tab2:
        plt.figure(figsize=(8, 6))
        sns.countplot(x="key", data=cleaned_df, palette="Set2")
        plt.xlabel("Key", fontsize=10)
        plt.ylabel("Count", fontsize=10)
        st.pyplot(plt.gcf())
        
        st.write('The key that is most commonly used is C#')
    
    with tab3:
        cleaned_df['mode'].value_counts().plot.pie(autopct = '%1.2f%%', legend = True);
        st.pyplot(plt.gcf())
        
        st.write('The Mode of the song that is most commonly used is Major')
    
    with tab4:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='key', y='streams', hue='mode', data=cleaned_df)
        plt.xlabel('Key')
        plt.ylabel('Streams')
        st.pyplot(plt.gcf())
        
        st.write('In the graph we see, we have a boxplot representing streaming streams broken down by music key and mode. Each boxplot, colored by mode, allows us to compare the distributions of streams among different keys and modes. Taking key D as an example, we note that the major mode has a much larger box than the minor mode, indicating greater variability in flows for the major mode. However, the average values of the flows between the two modes are similar. This suggests that although the major mode has a more variable distribution, the average number of flows does not differ significantly from that of the minor mode.')
    #############################
    st.divider()
    st.subheader("The musical characteristics")
    
    st.write('We can also analyze top songs according to the trend of musical characteristics, namely: danceability, valence, energy, acousticness, instrumentalness, liveness and finally speechiness.')
    
    tab1, tab2, tab3,tab4, tab5, tab6, tab7 = st.tabs(["Distribution of Danceability", "Distribution of Valence", "Distribution of energy", "Distribution of Acousticness", "Distribution of Instrumentalness", "Distribution of Liveness", "Distribution of Speechiness"])
    
    with tab1:
        plt.figure(figsize=(10, 6))
        sns.histplot(cleaned_df['danceability_%'], bins=30, kde=True, color='skyblue')
        plt.xlabel('Danceability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Danceability')
        plt.grid(True)
        st.pyplot(plt.gcf())
        
        top_danceable_songs = cleaned_df.nlargest(3, 'danceability_%')[['track_name', 'artist(s)_name', 'danceability_%']]
        top_danceable_songs
        
        st.write('These are the top3 songs with higher percentage of danceability, that indicate how suitable the song is for dancing')
        
    with tab2:
        plt.figure(figsize=(10, 6))
        sns.histplot(cleaned_df['valence_%'], bins=30, kde=True, color='lightcoral')
        plt.xlabel('Valence')
        plt.ylabel('Frequency')
        plt.title('Distribution of Valence')
        plt.grid(True)
        st.pyplot(plt.gcf())
        
        top_positive_songs = spotify_songs_df.nlargest(3, 'valence_%')[['track_name', 'artist(s)_name', 'valence_%']]
        top_positive_songs
        
        st.write('These are the top3 songs with higher percentage of Valence, that indicate the positivity of the song''s musical content')
        
    with tab3:
        plt.figure(figsize=(10, 6))
        sns.histplot(cleaned_df['energy_%'], bins=30, kde=True, color='lightgreen')
        plt.xlabel('Energy')
        plt.ylabel('Frequency')
        plt.title('Distribution of Energy')
        plt.grid(True)
        st.pyplot(plt.gcf())
        
        top_energetic_songs = cleaned_df.nlargest(3, 'energy_%')[['track_name', 'artist(s)_name', 'energy_%']]
        top_energetic_songs
        
        st.write('These are the top3 songs with higher percentage of Energy, that indicate the perceived energy level of the song')
    
    with tab4:
        plt.figure(figsize=(10, 6))
        sns.histplot(cleaned_df['acousticness_%'], bins=30, kde=True, color='orange')
        plt.xlabel('Acousticness')
        plt.ylabel('Frequency')
        plt.title('Distribution of Acousticness')
        plt.grid(True)
        st.pyplot(plt.gcf())
        
        top_acoustic_songs = cleaned_df.nlargest(3, 'acousticness_%')[['track_name', 'artist(s)_name', 'acousticness_%']]
        top_acoustic_songs
        
        st.write('There are the top3 songs with higher percentage of Acousticness, that indicated the amount of acoustic sound in the song')

    with tab5:
        plt.figure(figsize=(10, 6))
        sns.histplot(cleaned_df['instrumentalness_%'], bins=30, kde=True, color='yellow')
        plt.xlabel('Instrumentalness')
        plt.ylabel('Frequency')
        plt.title('Distribution of Instrumentalness')
        plt.grid(True)
        st.pyplot(plt.gcf())
        
        top_instrumental_songs = cleaned_df.nlargest(3, 'instrumentalness_%')[['track_name', 'artist(s)_name', 'instrumentalness_%']]
        top_instrumental_songs
        
        st.write('There are the top3 songs with higher percentage of Instrumentalness, that indicated the amount of instrumental content in the song')

    with tab6:
        plt.figure(figsize=(10, 6))
        sns.histplot(cleaned_df['liveness_%'], bins=30, kde=True, color='red')
        plt.xlabel('Liveness')
        plt.ylabel('Frequency')
        plt.title('Distribution of Liveness')
        plt.grid(True)
        st.pyplot(plt.gcf())
        
        top_liveness_songs = cleaned_df.nlargest(3, 'liveness_%')[['track_name', 'artist(s)_name', 'liveness_%']]
        top_liveness_songs
        
        st.write('There are the top3 songs with higher percentage of Liveness, that indicated the presence of live performance elements')
        
    with tab7:
        plt.figure(figsize=(10, 6))
        sns.histplot(cleaned_df['speechiness_%'], bins=30, kde=True, color='purple')
        plt.xlabel('Speechiness')
        plt.ylabel('Frequency')
        plt.title('Distribution of Speechiness')
        plt.grid(True)
        st.pyplot(plt.gcf())
        
        top_speech_songs = cleaned_df.nlargest(3, 'speechiness_%')[['track_name', 'artist(s)_name', 'speechiness_%']]
        top_speech_songs
        
        st.write('There are the top3 songs with higher percentage of Speechiness, that indicated the amount of spoken words in the song')

    

    ##################################### WordCloud graphs

    st.divider()
    st.subheader("WordCloud graph on dataset's note")
    
    words_in_title = ''.join(spotify_songs_df['track_name'].astype(str))
    wordcloud = WordCloud(width=1200, height=800, min_font_size=10, max_font_size=150).generate(words_in_title)
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear') 
    plt.axis('off')
    plt.title('Most Common Words in song titles')
    st.pyplot(plt.gcf())
    st.write('This graph is a WordCloud, which is a visual way to represent the most frequent words in a given dataset. In this case, the WordCloud was created based on the variable "track_name". This variable contains information about the songs, and the largest words in the are those that appear most frequently in the notes. This can help to quickly identify the most common themes or terms associated with the music.')


#########
#Modeling
#########
elif current_tab == "🤖 Modeling with ML algorithms":
    st.title("Modeling with Machine Learning algorithms")
    
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    st.write('Having almost exclusively categorical variables in the dataset, the data relating to the encoding of the categorical variables were used in the modeling phase. They were first processed through PCA Analysis to reduce the size of the data, and then processed through the KMeans clustering algorithm to identify groups based on similarities.')
    
    st.subheader('PCA Analysis using Categorical Values')
    st.write('Principal Component Analysis (PCA) is a dimensionality reduction technique that is often used to simplify data while retaining the most meaningful information.')
    
    cleaned_df = clean_data(spotify_songs_df)
    cleaned2_df = cleaned_df.copy()
    
    cleaned2_df.drop(columns= ['track_name', 'artist(s)_name','artist_count','released_year','released_month', 'released_day','in_spotify_playlists', 'in_spotify_charts','in_apple_playlists', 'in_apple_charts',
       'in_deezer_playlists', 'in_deezer_charts', 'in_shazam_charts','key','mode'], inplace=True)  
    for x in cleaned2_df.columns:
        print(f"column {x} has nulls: {cleaned_df[x].isnull().any()}, count: {cleaned_df[x].isnull().sum()}")
        cleaned_df[x + "_cat"] = pd.Categorical(cleaned_df[x]).codes
    
    Sum_of_squared_distances = []
    K = range(1,len(cleaned2_df.columns)+1)
    for n in K:
        pca = PCA(n_components=n)
        pca.fit(cleaned2_df)
        print(n,"components, variance ratio=",pca.explained_variance_ratio_)
    
    ########
    
    st.write('Using PCA to find the correct value of clusters')
    pca = PCA(n_components=len(cleaned2_df.columns))
    pca.fit(cleaned2_df)
    explained_variance=pca.explained_variance_ratio_
    cumulative_explained_variance=np.cumsum(pca.explained_variance_ratio_)
    plt.plot(K, explained_variance,marker='o', label='Explained Variance per Component')
    plt.plot(K, cumulative_explained_variance,marker='x', label='Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Elbow Diagram for fatalities PCA')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt.gcf())
    
    st.write('The elbow graph shows how much of the total variance is explained by the first N principal components. In this case, the N number of principal components is 2.')
    
    ########################################### KMeans clusters analysis

    st.subheader('K-means Clustering')
    st.write('To evaluate the actual effectiveness of the clustering algorithm, we examine the silhouette coefficient to evaluate the cohesion [-1, 1].')
  
    tab1, tab2 = st.tabs(["PCA2", "PCA9"])
    
    with tab1:
        pca = PCA(n_components=2)
        pca.fit(cleaned2_df)
        pca_data=pca.fit_transform(cleaned2_df)
    
        kmeans_2 = KMeans(n_clusters=2, random_state=20)
        kmeans_2.fit(cleaned2_df)
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_2.labels_, cmap='Accent')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title('K-means Clustering 2')
        plt.legend().set_visible(False)
        plt.show()
        st.pyplot(plt.gcf())

        from sklearn.metrics import silhouette_score
        silhouette_result = silhouette_score(cleaned2_df, kmeans_2.labels_)
        st.write("Silhouette coefficient for 2 clusters on data:", silhouette_result)
        st.write('Using N = 2 as the number of clusters as suggested by the analysis done earlier, we note how the coefficient si silhouette acquires a high average value.')


    with tab2:
        pca = PCA(n_components=9)
        pca.fit(cleaned2_df)
        pca_data9 = pca.fit_transform(cleaned2_df)
        
        kmeans_9 = KMeans(n_clusters=9, random_state=20, n_init=10)
        kmeans_9.fit(cleaned2_df)
        plt.scatter(pca_data9[:, 0], pca_data9[:, 1], c=kmeans_9.labels_, cmap='Set2')
        plt.xlabel('Main component PCA1')
        plt.ylabel('Main component PCA2')
        plt.title('K-means Clustering (n = 9)')
        plt.show()
        st.pyplot(plt.gcf())
        silhouette_result = silhouette_score(cleaned2_df, kmeans_9.labels_)
        st.write("Silhouette coefficient for 9 clusters on data:", silhouette_result)
        st.write('Test to try the maximum number (9) of pca components and then setting 9 as the number of clusters.')
    
    st.divider()
    st.subheader('Finding the best silhouette coefficient using high values.')
    st.write('By using a high values, numerous attempts can be made in order to find the number of clusters that maximizes the coefficient. Only a few tests are given below (maximum value N = 900), as the computational power using too high values of N is too much.')

    tab1, tab2 = st.tabs(["PCA100", "PCA900"])

    with tab1:
        kmeans_100 = KMeans(n_clusters=100, random_state=20)
        kmeans_100.fit(cleaned2_df)
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_100.labels_, cmap='Set2')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title('K-means Clustering 100')
        plt.show()
        st.pyplot(plt.gcf())
        silhouette_result = silhouette_score(cleaned2_df, kmeans_100.labels_)
        st.write("Silhouette coefficient for 100 clusters on data:", silhouette_result)
    
    with tab2:
        kmeans_900 = KMeans(n_clusters=900, random_state=20)
        kmeans_900.fit(cleaned2_df)
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_900.labels_, cmap='Set2')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title('K-means Clustering 900')
        plt.show()
        st.pyplot(plt.gcf())
        silhouette_result = silhouette_score(cleaned2_df, kmeans_900.labels_)
        st.write("Silhouette coefficient for 900 clusters on data:", silhouette_result)
    
    st.write('''
            The high silhouette coefficient suggests a valid separation between clusters and internal consistency. The numerosity of the clusters, however, raises questions about the true complexity of the data.
             ''')
        