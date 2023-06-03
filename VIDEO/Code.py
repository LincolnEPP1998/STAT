# Descargar los archivos necesarios
url_weights = "https://pjreddie.com/media/files/yolov3.weights" # Carga modelos YOLO
urllib.request.urlretrieve(url_weights, "yolov3.weights")

url_cfg = "https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg" # Carga modelos YOLO
urllib.request.urlretrieve(url_cfg, "yolov3.cfg")

url_names = "https://github.com/pjreddie/darknet/raw/master/data/coco.names" # Etiquetas para YOLO
urllib.request.urlretrieve(url_names, "coco.names")

# Cargar el modelo YOLO y las clases
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
output_layers = net.getUnconnectedOutLayersNames()


# Cargar el modelo YOLO y las clases
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
output_layers = net.getUnconnectedOutLayersNames()

# Cargar las clases de objetos
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
video_path = "Videos_UEB/VideoCodefest_001-11min.mpg"  # Ruta al archivo de video que deseas utilizar
cap = cv2.VideoCapture(video_path)
height, width = cap.read()[1].shape[0:2]

#Caracteristicas del video
video = cv2.VideoWriter('db0.wmv',cv2.VideoWriter_fourcc(*'mp4v'),2,(width,height))
video1 = cv2.VideoWriter('db1.wmv',cv2.VideoWriter_fourcc(*'mp4v'),2,(350,80))
video2 = cv2.VideoWriter('db2.wmv',cv2.VideoWriter_fourcc(*'mp4v'),2,(339,179))

contador = 1

Resultado = []

# Loop para procesar cada fotograma del video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    #frame = frame[98:609, 115:1100]

    # Procesar el fotograma
    height, width, channels = frame.shape

    Img1 = frame[30:110, 930:1280]
    Img2 = frame[1:180, 1:340] 
    
    #print(Img1.shape)
    #print(Img2.shape)
        
    # Detectar objetos en el fotograma
    blob = cv2.dnn.blobFromImage(frame, 0.0015,
                                 (416, 416), (0,0,0), True, crop=False)
    
    
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Procesar las detecciones
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.65: # Umbral de confianza
                # Obtener las coordenadas del objeto detectado
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calcular las coordenadas de la caja delimitadora
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Guardar la información de la detección
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Aplicar la supresión no máxima para evitar detecciones duplicadas
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)

    # Dibujar las cajas delimitadoras y etiquetas en el fotograma
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 5), font, 0.6, color, 2)
            print(contador)
            reader = easyocr.Reader(['en'])
            result = reader.readtext(Img1, detail = 0)    
            datos_1 = ''
            for i in range(0, len(result)-1):
                datos_1 = datos_1+result[i]
            reader = easyocr.Reader(['en'])
            result = reader.readtext(Img2, detail = 0)    
            datos_2 = ''
            for i in range(0, len(result)-1):
                datos_2 = datos_2+result[i]                
            
            Resultado.append({
                "ID": contador,
                "OBJECT_TYPE": label,
                "TIME": datos_2,
                "COORDINATES_TEXT":datos_1})
            
            video.write(frame)
            video1.write(Img1)
            video2.write(Img2)
            contador=contador+1
            
            
video.release()
video1.release()
video2.release()

Resultado
            
