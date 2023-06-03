# preprocesamiento
def limpieza(df,columna):
    """Esta función es la encargada de realizar todo el preprocesamiento de tlos datos textuales
    para las etiquetas, categoriza de forma automatica,
    para la columna texto, realiza un preprocesamiento del texto, tokenizando y removiendo stopwords,
    para la columna fecha extrae el año de la noticia"""
    #conversión a minusculas
    df[columna] =  df[columna].str.lower()
    
    #eliminación de acentos
    df[columna] = df[columna].str.replace("á","a")
    df[columna] = df[columna].str.replace("é","e")
    df[columna] = df[columna].str.replace("í","i")
    df[columna] = df[columna].str.replace("ó","o")
    df[columna] = df[columna].str.replace("ú","u")
    df[columna] = df[columna].str.replace(".","")
    df[columna] = df[columna] = df[columna].str.strip()
    
    # etiquetado para completar clases
    
    
    
    if columna == "ETIQUETA":
        df[columna] = np.where(df[columna].str.contains("perida de bosque|chiribiquete|ganaderia|agotamiento de los suelos en la amazonia|incendios|perdida de bosque|tala ilega|sequia|reforestacion|fauna y flora|fenomeno el niño|cambio climatico|derrame de petrolio|ganaderia y deforestacion|caza del defin rosado|ganaderia extensiva y deforestacion|carreteras"),"deforestacion",
                                np.where(df[columna].str.contains("narcotrafico|mineria|conflicto armado|cultivos ilicitos|cultivos ilicitos y deforestacion|tala y quema|recursos ilicitos"),"mineria ilegal",
                                         np.where(df[columna].str.contains("indigenas|vias de transporte|contaminacion|contaminacion por mercurio|aumento de represas hidroelectricas|plan contra la proteccion inmediata de la amazonia|escasez|extraccion de recursos naturales"),"contaminacion",df[columna])))
        
        return df[columna]
     
    # transformación de la fecha
    if columna == "FECHA ":
        df[columna] = df[columna].astype("str").apply(lambda x: x[-4:])
        
        return df[columna]
    
    # transformación de la columna texto
    
    
    if columna == "TEXTO":
        df[columna] = df[columna].apply(lambda x: ' '.join(re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', x).split())) 
        df[columna] = df[columna].apply(lambda x: ' '.join(re.sub('^|deforestacion|amazonas|amazonia|colombia|colombianos|mineria|ilegal'," ",x)\
                                          .split()))
        df[columna] = df[columna].apply(lambda x: nltk.word_tokenize(x))
        df[columna] = df[columna].apply(lambda x: [word for word in x if word not in stopwords.words('spanish')])
        return df[columna]
    
    
    return df[columna]
  
  
  
  def modelo_ner(df,I):
    """df hace referencia a la columna a la cual se le desea aplicar NER"""
    tagger = SequenceTagger.load("flair/ner-spanish-large")
    entities = []
    
    #for i in range(0, df.shape[0]):
        
    oracion = Sentence(df.iloc[0,:]["TEXTO"])
    tagger.predict(oracion)
    for entity in oracion.get_spans("ner"):
        entities.append({
                "text": entity.text,
                "type": entity.tag,
                "score": entity.score})
    entities = pd.DataFrame(entities)
    entities = entities.assign(Texto = df.iloc[0,:]["TEXTO"])
    json=[]
    
    datos_1 = ''
    for i in range(0, len(list(entities[entities["type"]=="ORG"]["text"]))-1):
        datos_1 = datos_1+list(entities[entities["type"]=="ORG"]["text"])[i]
        
    datos_2 = ''
    for i in range(0, len(list(entities[entities["type"]=="LOC"]["text"]))-1):
        datos_2 = datos_2+list(entities[entities["type"]=="LOC"]["text"])[i]
        
    datos_3 = ''
    for i in range(0, len(list(entities[entities["type"]=="PER"]["text"]))-1):
        datos_3 = datos_3+list(entities[entities["type"]=="PER"]["text"])[i]
        
    datos_5 = ''
    for i in range(0, len(list(entities[entities["type"]=="MISC"]["text"]))-1):
        datos_5 = datos_5+list(entities[entities["type"]=="MISC"]["text"])[i]
    
    
    
    json.append({'texto':df["TEXTO"][I],
             'org':datos_1,
             'loc':datos_2,
             'per':datos_3,
             'date':limpieza(df,"FECHA ")[I],
             'misc':datos_5,
             'impact':limpieza(df,"ETIQUETA")[I]
            })
    
    
    return json
                
    
    
    
    
def ner_from_file(text_path):
    noticias = pd.read_excel(text_path)
    noticias = noticias.dropna()
    noticias.index = range(0,noticias.shape[0])
    return modelo_ner(noticias,0)
 


ner_from_file("NOTICIAS DE LA AMAZONIA_CODEFEST.xlsx")


noticias = pd.read_excel("NOTICIAS DE LA AMAZONIA_CODEFEST.xlsx")
noticias = noticias.dropna()
noticias.index = range(0,noticias.shape[0])
noticias


noticias["ETIQUETA"] = limpieza(noticias, "ETIQUETA")
noticias["TEXTO"] = limpieza(noticias, "TEXTO")
noticias["FECHA"] = limpieza(noticias, "FECHA ")


def visualizador_categorias(df,columna, categoria):
    """Esta función debe ingresar los datos preprocesados, con ayuda de la función limpieza, 
    para cada columna correspondiente
    df: Conjunto de datos
    columna: columna de etiquetas
    categoria: deforestacion, mineria ilegal o contaminacion"""
    
    all_words = nltk.FreqDist([w for tokenlist in noticias[df[columna]==categoria].loc[:,"TEXTO"]
.values for w in tokenlist])
    all_words.plot(20,title = categoria)
    print(f"En la categoria {categoria}, existen {df[df[columna]==categoria][columna].count()} noticias, que representan el {(df[df[columna]==categoria][columna].count())/df.shape[0]} del conjunto de datos")
    
    
visualizador_categorias(noticias,"ETIQUETA","deforestacion")

analyzer = create_analyzer(task="sentiment", lang="es")
hate_speech_analyzer = create_analyzer(task="hate_speech", lang="es")

# Definir una función auxiliar para obtener el sentimiento
def get_sentiment(text):
    prediction = analyzer.predict(text)
    sentiment = prediction.output
    return sentiment


# Aplicar la función a la columna 'texto' utilizando el método 'apply' y almacenar los resultados en una nueva columna 'sentimiento'
noticias["TEXTO"].apply(get_sentiment)
noticias
