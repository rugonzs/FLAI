import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent

VERSION = '3.0.1' #Muy importante, deberéis ir cambiando la versión de vuestra librería según incluyáis nuevas funcionalidades
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
      'scikit-learn==1.5.2','bnlearn==0.10.2','networkx==3.4.2','matplotlib==3.9.2','pgmpy==0.1.26','numpy==1.26.4', 'pandas==2.2.3','scipy==1.11.4','statsmodels==0.14.4'
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
