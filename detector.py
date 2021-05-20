import cv2 as cv
import numpy as np
import wget
from os import mkdir, path
from os.path import join, abspath, dirname, exists

file_path = abspath(__file__)
file_parent_dir = dirname(file_path)
config_dir = join(file_parent_dir, 'config')
inputs_dir = join(file_parent_dir, 'inputs')
yolo_weights_path = join(config_dir, 'yolov3.weights')
yolo_names_path = join(config_dir, 'coco.names')
yolo_config_path = join(config_dir, 'yolov3.cfg')
input_image = join(inputs_dir, 'kemy.jpg')

net = cv.dnn.readNet(yolo_weights_path, yolo_config_path)
#To load all objects that have to be detected
classes=[]
with open(yolo_names_path,"r") as file_object:
    lines = file_object.readlines()

for line in lines:
    classes.append(line.strip("\n"))

#Defining layer names
layer_names = net.getLayerNames()
output_layers = []
for i in net.getUnconnectedOutLayers():
    output_layers.append(layer_names[i[0]-1])

img = cv.imread(input_image)
height, width, channels = img.shape

#Extracting features to detect objects
blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#We need to pass the img_blob to the algorithm
net.setInput(blob)
outs = net.forward(output_layers)

#Displaying information on the screen
class_ids = []
confidences = []
boxes = []
centros = [] #Apoyo para medir las distancias entre personas

for output in outs:
    for detection in output:
        #Detecting confidence in 3 steps
        scores = detection[5:]                #1
        class_id = np.argmax(scores)          #2
        confidence = scores[class_id]        #3

        if confidence > 0.5: #Means if the object is detected
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            #Drawing a rectangle
            x = int(center_x-w/2) # top left value
            y = int(center_y-h/2) # top left value

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            #cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            centros.append([center_x, center_y, x, y, w, h]) #Apoyo para medir las distancias entre personas
            

#Removing Double Boxes
indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4) #0.3, 0.4
w_prom = 0
w_contador = 0
centros_f = []
n_personas = 0 #Contador de personas

for i in range(len(boxes)):
    if i in indexes: #original if i in indexes[0]:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]  # name of the objects
       
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(img, label, (x, y), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2) #indica que es una persona

        centros_f.append(centros[i])
        w_prom = w_prom + w #Se obtiene el promedio del tamaño de las cajas
        w_contador += 1
        n_personas += 1 #Contador de personas

if n_personas>0:
    w_prom = w_prom/w_contador #Obtiene el promedio del ancho de las cajas para usarlo como medida de distancia relativa

#Determina si la persona está a una distancia segura de la otra
#Ubica los centros de las personas y mide sus distancias según w_prom
distancia = ""
validador = 0
color = (0,0,0)
for i in range(0,n_personas):
    centroi = centros_f[i]
    Validador = 0
    distancia="Distancia segura"
    color = (0,255,0)
    for centroj in centros_f:
        centro_x, centro_y, x, y, w, h = centroi
        centroA_x, centroA_y, x2, y2, w2, h2 = centroj
        if centroi != centroj:
            if abs(centro_x-centroA_x) > w_prom*1.7 or abs(centro_y-centroA_y) > w_prom*1.7: #Determina la distancia segura promedio de la hitbox por 1,7 hitboxes en distancia
                if validador == 0:
                    distancia = "Distancia segura"
                    color = (0,255,0)
            else:
                validador = 1
                distancia = "Distancia no segura"
                color = (0,0,255)
    if n_personas == 1:
        distancia = "Distancia segura"
        color = (0,255,0)
    cv.putText(img, distancia, (x, y-10), cv.FONT_HERSHEY_PLAIN, 1, color, 2)


#Escribe si el área está sobrepopulada según la cantidad de personas en la imagen
#El límite de personas puede ser adecuado según el lugar
if n_personas <= 25: #25
    seguridad = "Nivel de capacidad: Area libre"
    cv.putText(img, seguridad, (0, 10), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
if n_personas > 25 and n_personas <= 50: #25 a 50
    seguridad = "Nivel de capacidad: Area congestionada, tome precauciones"
    cv.putText(img, seguridad, (0, 10), cv.FONT_HERSHEY_PLAIN, 1, (100, 255, 255), 2)
if n_personas > 50: # 50
    seguridad = "Nivel de capacidad: Exceso de personas"
    cv.putText(img, seguridad, (0, 10), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

title = f"Found <persons={n_personas}>"
cv.imshow(title, img)
cv.waitKey(0)
cv.destroyAllWindows()