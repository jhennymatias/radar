#-*- coding:utf-8 -*-
import cv2
import numpy as np
import time
from datetime import datetime
from time import sleep

detec = []
subtracao = cv2.createBackgroundSubtractorMOG2()

##centro do objeto
def pega_centro(x, y, w, h):
	x1 = int(w / 2)
	y1 = int(h / 2)
	cx = x + x1
	cy = y + y1
	return cx,cy

##valor definido por tentativas
def resizeimage(frame):
	height, width, layers = frame.shape
	new_h = 270
	new_w = 430
	frame = cv2.resize(frame, (new_w, new_h))
	return frame

##escolhidos os limites da estrada
def getPerspectiveTransformation1(frame):
	rows, cols, ch = frame.shape
	pts1 = np.float32([[74, 267],  [327, 267],[177, 82], [246, 82]])
	pts2 = np.float32([[0, 0], [530, 0], [0, 400], [530, 400]])
	M = cv2.getPerspectiveTransform(pts1, pts2)
	dst = cv2.warpPerspective(frame, M, (530, 650))
	return dst

##para gravar vídeo
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 25.0, (1060, 400))

##para ler videos
cap1 = cv2.VideoCapture('passarela.mp4')


while(1):
	ret1, frame1 = cap1.read()
	if ret1:
		
		##redimenciona videos		
		frame1 = resizeimage(frame1)
		cv2.imshow('frame1', frame1)
		
		##faz transformada de perspectiva		
		result = getPerspectiveTransformation1(frame1)

		##desenha as barras vermelhas que serão utilizadas para calcular a velocidade
		cv2.line(result, (0, 200), (600, 200), (0,0,255), 3) 
		cv2.line(result, (0, 500), (600, 500), (0,0,255), 3) 


		## redução de ruidos
		grey = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
		blur = cv2.GaussianBlur(grey,(3,3),5)
		
		## background subtraction 
		img_sub = subtracao.apply(blur)

		## para deixar os objetos em apenas um bloco
		dilat = cv2.dilate(img_sub,np.ones((5,5)))
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
		dilatada = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
		dilatada = cv2.morphologyEx (dilatada, cv2. MORPH_CLOSE , kernel)
		
		##encontra os contornos
		contorno,hera = cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		
		## analisa contornos encontrados
		for(i,c) in enumerate(contorno):
			(x,y,w,h) = cv2.boundingRect(c)
			
			validar_contorno = (w <= 400) and (h <= 400)and (w >= 100) and (h >= 100)
			
			if not validar_contorno:
				continue
			
			##desenha um retangulo em volta dos objetos
			cv2.rectangle(result,(x,y),(x+w,y+h),(0,255,0),2)        
			centro = pega_centro(x, y, w, h)
			detec.append(centro)
			cv2.circle(result, centro, 4, (0, 0,255), -1)
			
			for (x,y) in detec:
				if y < (500+8) and y>(500-8):

					## caso o objeto esteja em cima da linha pega tempo
					a = datetime.now()
					cv2.line(result, (0, 500), (600, 500), (0,255,0), 3)  
					detec.remove((x,y))

				if y < (300+8) and y>(300-8):
					
					## caso o objeto esteja em cima da linha pega tempo
					b = datetime.now()
					cv2.line(result, (0, 200), (600, 200), (0,255,0), 3)  
					detec.remove((x,y))

					## calcula tempo entre linhas
					duration = b - a
					duration_in_s = duration.total_seconds() 
					
					##calcula pixel/segundo
					velocidade = 300 / duration_in_s
					velocidade = round(velocidade, 2)
					
					##caso a velocidade esteja acima salva imagem do carro
					if velocidade > 200:
						cv2.imwrite('acima.jpg', frame1)	
						cv2.putText(result, str(velocidade)+ " pixel/s", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),5)
						sleep(1)
		

		##mostra resultado e salva uma imagem            
		cv2.imshow('Resultado', result)
		
		##cria video
		out.write(result)

		##sai caso aperta 'esc'
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
	else:
		break

cap1.release()
out.release()
cv2.destroyAllWindows()

