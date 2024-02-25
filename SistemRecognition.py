# libraries
import cv2
import face_recognition as fr
import numpy as np
import mediapipe as mp
import os
from tkinter import *
from PIL import Image, ImageTk
import imutils
import math

from fontTools.config import Config


#log Bioetric Fuction
def Log_Biometric():
    global pantalla2, conteo, parpadeo, img_info, step, cap, lblVideo, RegUser

    # Check cap
    if cap is not None:
        ret, frame = cap.read()

        # Resize
        frame = imutils.resize(frame, width=1280)

        # RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameSave = frameRGB.copy()

        # Frame Show
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret == True:
            # Inference Face Mesh
            res = Facemesh.process(frameRGB)

            # Result List
            px = []
            py = []
            lista = []
            if res.multi_face_landmarks:
                # Extract Face Mesh
                for rostros in res.multi_face_landmarks:
                    # Draw
                    mpDraw.draw_landmarks(frame, rostros, FacemeshObject.FACEMESH_CONTOURS, ConfigDraw, ConfigDraw)

                    # Extract KeyPoint
                    for id, puntos in enumerate(rostros.landmark):

                        # Info img
                        al, an, c = frame.shape
                        x, y = int(puntos.x * an), int(puntos.y * al)
                        px.append(x)
                        py.append(y)
                        lista.append([id, x, y])

                        # 468 KeyPoints
                        if len(lista) == 468:
                            # ojo derecho
                            x1, y1 = lista[145][1:]
                            x2, y2 = lista[159][1:]
                            cx, cy =(x1 + x2) //2, (y1 + y2) // 2
                            longitud1 = math.hypot(x2-x1, y2-y1)

                            # ojo izquiero
                            x3, y3 = lista[374][1:]
                            y4, x4 = lista[386][1:]
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            longitud2 = math.hypot(x4 - x3, y4 - y3)

                            # Pariental Derecho
                            x5, y5 = lista[139][1:]
                            # Parietal Izquierdo
                            x6, y6 = lista[368][1:]

                            # Ceja Derecha
                            x7, y7 = lista[70][1:]
                            # Ceja Izquierda
                            x8, y8 = lista[300][1:]

                            # Face Detect
                            faces = detector.process(frameRGB)

                            if faces.detections is not None:
                                for face in faces.detections:

                                    #Bbox: "ID, BBOX, SCORE"
                                    score = face.score
                                    score = score[0]
                                    bbox = face.location_data.relative_bounding_box

                                    # Threshold
                                    if score > confThreshold:
                                        # Pixels
                                        xi, yi, anc, alt = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                                        xi, yi, anc, alt = int(xi * an), int(yi * al), int(anc * an), int(alt * al)

                                        # Offset X
                                        offsetan = (offsetx / 100) * anc
                                        xi = int(xi - int(offsetan/2))
                                        anc = int(anc + offsetan)
                                        xf = xi * anc

                                        # Offset Y
                                        offsetal = (offsety / 100) * alt
                                        yi = int(yi - offsetal)
                                        alt = int(alt + offsetal)
                                        yf = yi + alt

                                        # Error
                                        if xi < 0: xi = 0
                                        if yi < 0: yi = 0
                                        if anc < 0: anc = 0
                                        if alt < 0: alt = 0

                                        # Steps
                                        if step == 0:
                                            # Draw
                                            cv2.rectangle(frame, (xi, yi, anc, alt), (255, 255, 255), 2)

                                            # IMG Step0
                                            als0, ans0, c = img_step0.shape
                                            frame[50:50 + als0, 50:50 + ans0] = img_step0
                                            # IMG Step1
                                            als1, ans1, c = img_step1.shape
                                            frame[50:50 + als1, 1030:1030 + ans1] = img_step1
                                            # IMG Step2
                                            als2, ans2, c = img_step2.shape
                                            frame[270:270 + als2, 1030:1030 + ans2] = img_step2


                                            # Face Center
                                            if x7 > x5 and x8 < x6:
                                                # IMG Check
                                                alch, anch, c = img_check.shape
                                                frame[165:165 + alch, 1105:1105 + anch] = img_check

                                                # Cont Parpadeo
                                                if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:
                                                    conteo = conteo + 1
                                                    parpadeo = True

                                                elif longitud1 > 10 and longitud2 > 5 and parpadeo == True:
                                                    parpadeo = False

                                                    cv2.putText(frame, f'Parpadeos: {int(conteo)}', (1070, 375), cv2.FONT_HERSHEY_COMPLEX,0.5, (255, 255, 255),
                                                                1)


                                                    # Cond
                                                    if conteo >= 3:
                                                        # Img Check
                                                        alch, anch, c = img_check.shape
                                                        frame[385:385 + alch, 1105:1105 + anch] = img_check

                                                        # Open Eyes
                                                        if longitud1 > 14 and longitud2 > 14:
                                                            # Cut
                                                            cut = frameSave[yi:yf, xi:xf]

                                                            # Save Face
                                                            cv2.imwrite(f"{OutFolderPathFace}/{RegUser}.png", cut)

                                                            # Step 1
                                                            step = 1

                                            else:
                                                conteo = 0

                                        if step == 1:
                                            # Draw
                                            cv2.rectangle(frame, (xi, yi, anc, alt), (0, 255, 0), 2)
                                            # IMG Check Liveness
                                            alli, anli, c = img_liche.shape
                                            frame[50:50 + alli, 50:50 + anli] = img_liche




                            # Circle
                            cv2.circle(frame, (x7, y8), 2, (255,0,0), cv2.FILLED)
                            cv2.circle(frame, (x8, y8), 2, (255, 0, 0), cv2.FILLED)





        # Conv Video
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)

        # Show Video
        lblVideo.configure(image=img)
        lblVideo.image = img
        lblVideo.after(10, Log_Biometric)

    else:
        cap.relase()



# Function  Sign
def Sign():
    print("adios")

# Function Log
def Log():
    global RegName, RegUser, RegPass, InputNameReg, InputUserReg, InputPassReg, cap, lblVideo, pantalla2
    # Extract: Name - User - PassWord
    RegName, RegUser, RegPass = InputNameReg.get(), InputUserReg.get(), InputPassReg.get()

    # Incompleted Form
    if len(RegName) == 0 or len(RegUser) == 0 or len(RegPass) == 0:
        # Print Error
            print(" FORMULARIO INCOMPLETO ")
        # Completed Form
    else:
        # Check User
        UserList = os.listdir(PathUserCheck)

        # Name Users
        UserName = []

        # Check User List
        for lis in UserList:
            # Extract User
            User = lis
            User = User.split('.')
            # Save User
            UserName.append(User[0])

        # Check User
        if RegUser in UserName:
            # Registrado
           print("Usuario registrado anteriormente")

        else:
            # No Registred
            info.append(RegName)
            info.append(RegUser)
            info.append(RegPass)

    # Export Info
    f = open(f"{OutFolderPathUser}/{RegUser}.txt", "w")
    f.write(RegName + ',')
    f.write(RegUser + ',')
    f.write(RegPass)
    f.close()

    # Clean
    InputNameReg.delete (0, END)
    InputUserReg.delete (0, END)
    InputPassReg.delete (0, END)

    # New Screen
    pantalla2 = Toplevel(pantalla)
    pantalla2.title("Login Biometric")
    pantalla2.geometry("1280x720")

    # Label Video
    lblVideo = Label(pantalla2)
    lblVideo.place(x=0, y=0)

    # VideoCapture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)
    Log_Biometric()

# path
OutFolderPathUser = 'C:/Users/52771/Desktop/SecureFaceNet/SecureFaceNet/DataBase/Users'
PathUserCheck = 'C:/Users/52771/Desktop/SecureFaceNet/SecureFaceNet/DataBase/Users/'
OutFolderPathFace = 'C:/Users/52771/Desktop/SecureFaceNet/SecureFaceNet/DataBase/Faces'

# Read img
img_info = cv2.imread("C:/Users/52771/Desktop/SecureFaceNet/SecureFaceNet/SetUp/Step1.png")
img_check = cv2.imread("C:/Users/52771/Desktop/SecureFaceNet/SecureFaceNet/SetUp/check.png")
img_step0 = cv2.imread("C:/Users/52771/Desktop/SecureFaceNet/SecureFaceNet/SetUp/mensaje-2.png")
img_step1 = cv2.imread("C:/Users/52771/Desktop/SecureFaceNet/SecureFaceNet/SetUp/Step1.png")
img_step2 = cv2.imread("C:/Users/52771/Desktop/SecureFaceNet/SecureFaceNet/SetUp/Step2.png")
img_liche = cv2.imread("C:/Users/52771/Desktop/SecureFaceNet/SecureFaceNet/SetUp/mensaje-4.png.")


# Variables
parpadeo = False
conteo = 0
muestra = 0
step = 0

# offset
offsety = 40
offsetx = 20

# Threshold
confThreshold = 0.5

# Tool Draw
mpDraw = mp.solutions.drawing_utils
ConfigDraw = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# Object  Face  Mesh
FacemeshObject = mp.solutions.face_mesh
Facemesh = FacemeshObject.FaceMesh(max_num_faces=1)

# Object Face Detect
FaceObject = mp.solutions.face_detection
detector = FaceObject.FaceDetection(min_detection_confidence=0.5, model_selection=1)

# Info list
info = []

# Ventana principal
pantalla = Tk()
pantalla.title("SecureFaceNet")
pantalla.geometry("1280x720")

# Fondo
imagenF = PhotoImage(file="C:/Users/52771/Desktop/SecureFaceNet/SecureFaceNet/SetUp/reconoce.png")
background = Label(image = imagenF, text = "Inicio")
background.place(x=0, y=0)

# input text
# Name
InputNameReg = Entry(pantalla)
InputNameReg.place(x=110, y=320)

# User
InputUserReg = Entry(pantalla)
InputUserReg.place(x=110, y=430)

# Pass
InputPassReg = Entry(pantalla)
InputPassReg.place(x=110, y=540)

# input Text Sing Up
inputUserLog = Entry(pantalla)
inputUserLog.place(x=750, y=380)

inputPassLog = Entry(pantalla)
inputPassLog.place(x=750, y=500)

# Buttons
# log
imagenBR = PhotoImage(file="C:/Users/52771/Desktop/SecureFaceNet/SecureFaceNet/SetUp/BtLogin.png")
BtReg = Button(pantalla, text = "Registro", image=imagenBR, height="40", width="200", command=Log)
BtReg.place(x=300, y=580)



# log
imagenBL = PhotoImage(file="C:/Users/52771/Desktop/SecureFaceNet/SecureFaceNet/SetUp/BtSign.png")
BtSign = Button(pantalla, text="Registro", image=imagenBL, height="40", width="200", command=Sign)
BtSign.place(x=900, y=580)

pantalla.mainloop()