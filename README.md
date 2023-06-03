# STAT
RETO CODEFEST AD ASTRA 2023, Librería OpenSource para la identificación con un modelo para análisis textual, y otro para el análisis de Video



# Video

## *Primeros Pasos*

 + Si se desea ejecutar el código de video, es necesario contar con un video o carpeta de videos para ser procesado; estos videos deven estar cargados dentro del entorno para poder ser analisados.
 
Para inicir se debe instalar las siguientes librerias: remotezi (Para ayuda en cuanto a cargar la base de datos), TensorFlow 2.10.0 (Contine modelos de Machine Learning), ultralytics (Detección de objetos en videos e imagenes), opencv-python-headless y opencv-python (Correcta carga de los videos), tensorflow gradio y tensorflow_hub (Disponibilizar Modelos pre entrenados de ML), Pillow (Manipulacion de los frames de cada video), matplotlib (visualizacion los frames), gradio (Para apoyar los modelos ML pre entrenados) y torch y easyocr(Extraccion de las coordenas encontradas dentro de cada video)

 + Importar las siguientes librerias: tqdm, random, pathlib, itertools, collections, cv2, einops, numpy,remotezip, seaborn, matplotlib.pyplot, tensorflow, keras, layers de keras, Image de PIL, pytesseract, img2pdf y os

 + Se carga el video a analizar de la siguiente manera mediante cv2.VideoCapture
 + Se divide el video en fotogramas para analizar frame por frame por medio de cap.read.
 + Se ajusta el modelo pre entrenado de YOLO para captar información de los videos
 + Luego de los frame extraidos se recorta las coordenadas que se encuntran en el video para posteriormente extraer dicho texto de las imagenes por medio de easyocr.Reader y reader.readtext; metodos que solo funcionan si el archivo extract.py se encuentra dentro del proyecto.
 + Se estructuró los resultados tenidno en cuenta lo que el modelo haya encontrado dentro del videos y las corenas extraidas de los frames y se envia el un cvs conjuntamente con 3 videos (El priemero para hacer la segmentación, seguyndo la localizacion del avion y tercero para observar los objetos que capto)
 + 

