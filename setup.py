import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent

VERSION = '1.0.0' #Muy importante, deberéis ir cambiando la versión de vuestra librería según incluyáis nuevas funcionalidades
PACKAGE_NAME = 'FLAI_CAUSAL' #Debe coincidir con el nombre de la carpeta 
AUTHOR = 'Rubén González Sendino' #Modificar con vuestros datos
AUTHOR_EMAIL = 'rubo.g@icloud.com' #Modificar con vuestros datos
URL = 'https://github.com/rugonzs' #Modificar con vuestros datos

LICENSE = 'Apache-2.0 license' #Tipo de licencia
DESCRIPTION = 'Library to creat causal model and mitigate the bias.' #Descripción corta
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding='utf-8') #Referencia al documento README con una descripción más elaborada
LONG_DESC_TYPE = "text/markdown"


#Paquetes necesarios para que funcione la libreía. Se instalarán a la vez si no lo tuvieras ya instalado
INSTALL_REQUIRES = [
      'bnlearn','networkx','matplotlib','pgmpy','numpy', 'itertools','math','pandas','sklearn'
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
    install_requires=INSTALL_REQUIRES,
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True
)
