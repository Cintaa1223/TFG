import streamlit as st

def show_about_project():
    st.title("Sistema de Recomendación de Proyectos de Ciencia Ciudadana Basados en el Currículum de Primaria de Cataluña")
    st.write("""Este proyecto es un trabajo de final de grado realizado por Cinta Arnau Arasa, con la ayuda de Patricia Santos y Miriam Calvera como tutoras.""")
    st.write("""Este proyecto trata de la creación de un sistema de recomendación de proyectos de Ciencia Ciudadana. Los proyectos que conforman la base de datos han sido extraídos de las plataformas de Ciencia Ciudadana "Observatorio de la Ciencia Ciudadana en España" y "Oficina de la Ciència Ciutadana". """)
    st.write("""El objetivo del proyecto es poder recomendar un conjunto de proyectos de Ciencia Ciudadan que puedan trabajarse en las clases de primaria según las necesidades del profesor. La idea principal es poder encontrar y recomendar los proyectos que pueden ayudar a alcanzar las competencias clave de primaria según el currículum de primaria de Cataluña. Opcionalmente, los profesores tienen la posibilidad de encontrar otro tipo de proyectos según sus necesidades, simplemente introduciendo las palabras clave en el buscador. Adicionalmente, se puede realizar un filtrado según los ámbitos de los proyectos que se quieran tener en cuenta para el proceso de recomendación.""")