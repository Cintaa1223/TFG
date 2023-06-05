import streamlit as st
from recommender_system import show_characteristics_page
from key_competences import show_key_competences
from about_project import show_about_project
import base64
from pathlib import Path
from PIL import Image
import os


image = Image.open(Path(__file__).parents[0]/'UPF.png')
st.write("WEBAPP by CINTA ARNAU ARASA")
st.sidebar.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels='RGB', output_format='auto')
page = st.sidebar.selectbox("Escoge una sección a visitar:", ("Proyectos de Ciencia Ciudadana", "Competencias Clave", "Sobre el proyecto"))
show_background = st.sidebar.checkbox('Mostrar fondo de pantalla', value = True)
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    opacity: 0.9;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

if show_background:
    wallpaper = (Path(__file__).parents[0]/'wallpaper.png')
    set_background(wallpaper)

if page == "Proyectos de Ciencia Ciudadana":
    show_characteristics_page()
elif page == "Competencias Clave":
    show_key_competences()
else:
    show_about_project()
