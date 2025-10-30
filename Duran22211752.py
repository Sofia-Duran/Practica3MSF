"""
Práctica: Sistema musculoesquelético 

Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México

Nombre del alumno: Sofía Cristina Durán Muñoz 
Número de control: 22211752
Correo institucional: l22211752@tectijuana.edu.mx

Asignatura: Modelado de Sistemas Fisiológicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""
# Instalar librerias en consola
#!pip install control
#!pip install slycot
import control as ctrl

# Librerías para cálculo numérico y generación de gráficas
import numpy as np
import math as m
import matplotlib.pyplot as plt
import control as ctrl
from scipy import signal
import pandas as pd

# Datos de la simulación
x0,t0,tend,dt,w,h = 0,0,10,1E-3,10,5
n = round((tend - t0)/dt) + 1
t = np.linspace(t0,tend,n)
u = np.zeros(n); u[round(1/dt):round(2/dt)] = 1

def musculo(R,Cs,Cp):
    num = [Cs*R,1-0.25]
    den = [R*(Cs+Cp),1]
    sys = ctrl.tf(num,den)
    return sys

#Función de transferencia: Control
R,Cs,Cp = 100,10E-6,100E-6
syscontrol = musculo(R,Cs,Cp)
print(f'Función de transferencia del control: {syscontrol}')

#Función de transferencia: Caso
R,Cs,Cp = 10E3,10E-6,100E-6
syscaso = musculo(R,Cs,Cp)
print(f'Función de transferencia del caso: {syscaso}')

#Función de transferencia: Tratamiento
R,Cs,Cp = 10E3,10E-6,100E-6
systratamiento = musculo(R,Cs,Cp)
print(f'Función de transferencia del tratamiento: {systratamiento}')

#Respuestas en lazo abierto
_,Fs1 = ctrl.forced_response(syscontrol,t,u,x0)
_,Fs2 = ctrl.forced_response(syscaso,t,u,x0)

fg1 = plt.figure()
plt.plot(t,u,'-',linewidth=1,color=np.array([85,107,47])/255,label='F(t)')
plt.plot(t,Fs1,'-',linewidth=1,color=np.array([154,63,63])/255,label='Fs1(t): Control')
plt.plot(t,Fs2,'-',linewidth=1,color=np.array([27,60,83])/255,label='Fs2(t): Caso')
plt.grid(False)
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.2,1.2); plt.yticks(np.arange(-0.2,1.4,0.2))
plt.ylabel('F(t) [V]')
plt.xlabel('t [s]')
plt.legend(bbox_to_anchor=(0.5,-0.2),loc='center',ncol=3)
plt.show()
fg1.set_size_inches(w,h)
fg1.tight_layout()
fg1.savefig('sistema musculoesquelético lazo abierto python.png',dpi=600,bbox_inches='tight')
fg1.savefig('sistema musculoesquelético lazo abierto python.pdf')

#Controlador PI
kP = 0.0320097289384142
kI = 28350.2753909505
Cr = 1E-6
Re = 1/(kI*Cr)
Rr = kP*Re
numPI = [Rr*Cr,1]
denPI = [Re*Cr,0]
PI = ctrl.tf(numPI,denPI)
print(f'Función de transferencia del controlador PI: {PI}')
X = ctrl.series(PI,syscaso)
sysPI = ctrl.feedback(X,1,sign=-1)
print(f'Función de transferencia del controlador PI lazo cerrado: {sysPI}')
systratamiento = ctrl.series(syscontrol,sysPI)

##PI = controlador(0.0320097289384142,28350.2753909505,syshipo)

##Respuestas en lazo cerrado
_,Fs3 = ctrl.forced_response(systratamiento,t,u,x0)
fg2 = plt.figure()
plt.plot(t,u,'-',linewidth=1,color=np.array([85,107,47])/255,label='F(t)')
plt.plot(t,Fs1,'-',linewidth=1,color=np.array([154,63,63])/255,label='Fs1(t): Control')
plt.plot(t,Fs2,'-',linewidth=1,color=np.array([27,60,83])/255,label='Fs2(t): Caso')
plt.plot(t,Fs3,':',linewidth=3,color=np.array([152,161,188])/255,label='Fs3(t): Tratamiento')
plt.grid(False)
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.2,1.2); plt.yticks(np.arange(-0.2,1.4,0.2))
plt.ylabel('F(t) [V]')
plt.xlabel('t [s]')
plt.legend(bbox_to_anchor=(0.5,-0.2),loc='center',ncol=4)
plt.show()
fg2.set_size_inches(w,h)
fg2.tight_layout()
fg2.savefig('sistema musculoesquelético lazo cerrado python.png',dpi=600,bbox_inches='tight')
fg2.savefig('sistema musculoesquelético lazo cerrado python.pdf')