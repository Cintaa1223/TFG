import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import nltk
nltk.download('stopwords') #Dowload list of stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re #Library used to remove certain symbols / characters from a text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from pathlib import Path


def load_data(filename):
    data = pd.read_csv(filename, sep=",")
    data = data.drop_duplicates(subset=['Project Name'], keep = 'first')
    return data

projectsCS = load_data(Path(__file__).parents[1]/'VS_Projects.csv')

def barcelona_proj(web):
    if str(web) == 'Ciencia Ciudadana Ayuntamiento de Barcelona':
        return 'Oficina de la Ciència Ciutadana'
    return str(web)
    

projectsCS['Citizen Science Web Name'] = projectsCS.apply(lambda row: barcelona_proj(row['Citizen Science Web Name']), axis=1)


def separate_scopes(row_scopes, scopes):
        row_scopes = str(row_scopes).replace('.', '')
        proj_scopes = row_scopes.split(', ')
        for i in proj_scopes:
            if i not in scopes:
                scopes.append(i)


def scope_in_project(proj_sco, scope):
    if str(scope) in str(proj_sco):
        return str(scope)
    return str(proj_sco)


#This function receives a line of text as input and returns a list of the words contained in it after having removed the stopwords and non-important symbols, 
#transforming to lowercase, tokenizing and stemming.
def build_terms(line): 
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("spanish"))
    line = line.lower()  #Convert to lowercase
    line = line.split()  # Tokenize the text to get a list of terms
    line = [x for x in line if x not in stop_words]  # eliminate the stopwords
    line = [x for x in line if x.startswith(("@", "https://", "$", '#')) != True]  # eliminate non-important symbols
    line = [re.sub('[^a-záéíóúäëïöü]+', '', x) for x in line] # since it's in spanish we only have to worry about 'closed' accents
    line = [stemmer.stem(word) for word in line] # perform stemming
    return line


def select_max_rec(df):
    if len(df)<=10:
        return 1
    elif len(df)<=20:
        return 2
    elif len(df)<=30:
        return 3
    return 5


def create_wordcloud(df, idxs):
    top_idx_desc = []
    for idx in idxs:
        top_idx_desc.append(', '.join(build_terms(df['Project Full Description'].iloc[idx])))
    return top_idx_desc


def show_characteristics_page():
    st.title("Sistema de Recomendación de Proyectos de Ciencia Ciudadana Basados en el Currículum de Primaria de Cataluña")

    st.write("""### Este es el dataset de proyectos de Ciencia Ciudadana""")
    st.write("""##### Filtrar por:""")
    filteredCS = projectsCS.copy()
    platform = st.multiselect(':globe_with_meridians: Plataforma', ['Observatorio de la Ciencia Ciudadana en España', 'Oficina de la Ciència Ciutadana'],['Observatorio de la Ciencia Ciudadana en España', 'Oficina de la Ciència Ciutadana'])
    
    # To check all unique project scopes
    scopes = []
    projectsCS.apply(lambda row: separate_scopes(row['Project Scope'], scopes), axis=1)
    scopes.remove('nan')
    
    test = []
    scope = st.multiselect(':books: Ámbitos', scopes, scopes)
    
    if len(platform) != 0:
        if len(platform) == 1:
            filteredCS = projectsCS[projectsCS['Citizen Science Web Name'] == platform[0]]
        else:
            filteredCS = pd.concat([projectsCS[projectsCS['Citizen Science Web Name'] == platform[0]], projectsCS[projectsCS['Citizen Science Web Name'] == platform[1]]])

    scopesdf = []
    filteredCS['Scope2'] = filteredCS['Project Scope']
    for sco in scope:
        filteredCS['Scope2'] = filteredCS.apply(lambda row: scope_in_project(row['Scope2'], sco), axis=1)
        scopesdf.append(filteredCS[filteredCS['Scope2'] == sco])
    
    
    if len(scopesdf) != 0:
        filteredCS = pd.concat(scopesdf)
        filteredCS = filteredCS.drop(['Scope2'], axis=1)
        st.dataframe(filteredCS.sort_index())

    KC = ''
    KC1 = "Desarrollar una actitud responsable a partir de la toma de conciencia de la degradación del medio ambiente basada en el conocimiento de las causas que la provocan, agravan o mejoran, desde una visión sistémica, tanto local como global."
    KC2 = "Identificar los distintos aspectos relacionados con el consumo responsable y de productos de proximidad, valorando sus repercusiones sobre el bien individual y el común, juzgando críticamente las necesidades y los excesos y ejerciendo un control social ante la vulneración de sus derechos como consumidor."
    KC3 = "Desarrollar hábitos de vida saludable a partir de la comprensión del funcionamiento de el organismo y la reflexión crítica sobre los factores internos y externos que inciden, asumiendo la responsabilidad personal en la promoción de la salud pública, incluido el conocimiento de una sexualidad positiva, respetuosa e igualitaria."
    KC4 = "Ejercitar la sensibilidad para detectar situaciones de desigualdad y exclusión desde la comprensión de las causas complejas para desarrollar sentimientos de empatía."
    KC5 = "Desarrollar un compromiso activo con la igualdad de género, la igualdad de trato y la no discriminación, conociendo el recorrido histórico para la consecución de los derechos humanos de todas las personas y colectivos."
    KC6 = "Entender los conflictos como elementos connaturales a la vida en sociedad que deben resolver de forma pacífica y rechazar cualquier expresión de violencia machista, LGTBIfóbica, racista, capacitista o motivada por cualquier otro tipo de situación personal o socioeconómica."
    KC7 = "Analizar de forma crítica y aprovechar las oportunidades de todo tipo que ofrece la sociedad actual, en particular las de la cultura digital, evaluando sus beneficios y riesgos y hacer un uso ético y responsable que contribuya a la mejora de la calidad de vida personal y colectiva."
    KC8 = "Aceptar la incertidumbre como una oportunidad para articular respuestas más creativas, aprendiendo a gestionar la ansiedad que puede comportar."
    KC9 = "Cooperar y convivir en sociedades abiertas y cambiantes, valorar la diversidad personal y cultural como fuente de riqueza y fomentando el interés por otras lenguas y culturas."
    KC10 = "Sentirse parte de un proyecto colectivo, tanto a nivel local como global, desarrollando empatía y generosidad."
    KC11 = "Desarrollar las habilidades que le permitan seguir aprendiendo a lo largo de la vida, desde la confianza en el conocimiento como motor de desarrollo y la valoración crítica de los riesgos y beneficios de este conocimiento."
    listKC = [KC1, KC2, KC3, KC4, KC5, KC6, KC7, KC8, KC9, KC10, KC11]
    shortKC = ['medio ambiente', 'consumo de productos locales', 'vida saludable', 'desigualtad, exclusión y empatía', 'igualdad de género', 'conflictos en la sociedad', 'cultura digital', 'creatividad', 'lenguas y culturas', 'colectivo', 'seguir aprendiendo']

    if len(platform) != 0 and len(scope) != 0:
        pred = st.selectbox("### En qué quieres basarte para encontrar proyectos similares?", ("Competencias Clave", "Otras"))
        if pred == "Otras":
            KC = st.text_input('Introduce las palabras clave para buscar proyectos similares:')
        else:
            selectedKC = st.selectbox("Escoge una competencia clave:", (listKC))
            selectedKC = listKC.index(selectedKC)
            KC = shortKC[selectedKC]

        if KC != '':
            projectsCS_clean = filteredCS.copy()

            # STEP 1: Preprocess the data
            projectsCS_clean['Project Full Description'].apply(build_terms)
            KC = build_terms(KC)

            # STEP 2: Create text embeddings
            vectorizer = TfidfVectorizer()
            text_embeddings = vectorizer.fit_transform(projectsCS_clean['Project Full Description'])
            input_embedding = vectorizer.transform(KC)

            # STEP 3: Calculate similarity (cosine similarity)
            similarities = cosine_similarity(input_embedding, text_embeddings)

            # Step 4: Output CS project recommendations
            num_recommendations = select_max_rec(filteredCS)  # Number of recommended projects to display
            top_indices = similarities.argsort()[0][-num_recommendations:][::-1]  # Sort and get the top indices

            st.write("##### Proyectos recomendados:")
            st.dataframe(filteredCS.iloc[top_indices])

            wc = create_wordcloud(projectsCS_clean, top_indices)
            wordcloud = WordCloud(max_words=150, background_color="white").generate(', '.join(wc))
            fig, ax = plt.subplots(1, 1, figsize = (12, 8))
            ax.imshow(wordcloud, interpolation = 'bilinear')
            plt.axis('off')
            st.pyplot(fig)
