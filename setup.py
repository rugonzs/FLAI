import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent

VERSION = '1.0.10' #Muy importante, deberéis ir cambiando la versión de vuestra librería según incluyáis nuevas funcionalidades
PACKAGE_NAME = 'FLAI_CAUSAL' #Debe coincidir con el nombre de la carpeta 
AUTHOR = 'Rubén González Sendino' #Modificar con vuestros datos
AUTHOR_EMAIL = 'rubo.g@icloud.com' #Modificar con vuestros datos
URL = 'https://github.com/rugonzs/FLAI' #Modificar con vuestros datos

LICENSE = 'Apache-2.0 license' #Tipo de licencia
DESCRIPTION = 'Library to creat causal model and mitigate the bias.' #Descripción corta
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding='utf-8') #Referencia al documento README con una descripción más elaborada
LONG_DESC_TYPE = "text/markdown"
PROJECT_URLS = {
    "Documentation": "https://rugonzs.github.io/FLAI/",
    "Source Code": 'https://github.com/rugonzs/FLAI',
}

#Paquetes necesarios para que funcione la libreía. Se instalarán a la vez si no lo tuvieras ya instalado
INSTALL_REQUIRES = [
      'bnlearn==0.7.8','networkx==2.8.8','matplotlib==3.6.2','pgmpy==0.1.20','numpy==1.23.4', 'pandas==1.5.1','scikit-learn==1.0.2','scipy==1.9.3','statsmodels==0.13.5'
      ]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    project_urls=PROJECT_URLS,
    install_requires=INSTALL_REQUIRES,
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True
)
