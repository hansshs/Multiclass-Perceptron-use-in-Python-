# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:39:58 2022
@author: Hans Herbert Schulz
"""
#%% Bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import time as tm
#%% Funções ativação
degrau     = lambda z: 1 if z>=0 else 0
sigmoid    = lambda z: 1/(1+np.exp(-z))
chosen_one = sigmoid

np.random.seed(69420)
#%% Neurônio
class neuronio:
    def __init__(self, pesos, func_phi = chosen_one):
        self.pesos    = np.zeros(pesos) # = np.zeros(pesos+1)
        self.func_phi = func_phi
        self.threshol = 0.5
        
    def __repr__(self): 
        resp = [f'w{i} = {w}\n' for i,w in enumerate(self.pesos)]
        return ''.join(resp)
    
    def campo_ind(self, x):
        x              = np.r_[1,x] #Adicionar casa do bias
        campo_induzido = np.dot(x,self.pesos)
        v              = np.sum(campo_induzido)
        return v                   #Retorna Campo Induzido
    
    def chamar(self, x):
        v              = self.campo_ind(x)
        return self.func_phi(v)    #Saída Yi para entrada Xi do Neurônio
    


#%% Camadas
class layer:
    def __init__(self, nmb_neuronios, input_nmb, act_func = chosen_one):
        self.nmb_neuronios = nmb_neuronios
        self.layer = [neuronio(input_nmb +1, act_func) for i in range(nmb_neuronios)]
                        #[neuronio]
        self.trash = 0.8
        
    def chamar(self, x):
        ans = []
        for i in range(self.nmb_neuronios):
            ans.append(self.layer[i].chamar(x))
        return np.array(ans)
    
    def chamar_abs(self, x):
        resp = self.chamar(x)
        return [1 if i > self.trash else 0 for i in resp]
#%% Treinamento
def treinar(neuronio, dados, resposta,epoca = 100e3, eta = 0.1,
            t_plot = 1000, tol = 1e-3):
    tam                    = len(dados)
    indexes                = np.array(range((tam)))
    erro_total             = 0
    
    erro_total_pt          =[]      
    
    erro_print             = []
    iter_num               = 0
    
    for i in range(int(epoca)):
        np.random.shuffle(indexes)                     #Randomizar Inputs
        iter_num = iter_num + 1
        for j in (indexes):
           Y_r             = neuronio.chamar(dados[j]) #Resposta encontrada
           erro            = resposta[j] - Y_r
           for k in range(neuronio.nmb_neuronios):
               neuronio.layer[k].pesos += eta*(erro[k])*np.r_[1,dados[j]]
           erro_total     += sum(erro**2)/2
        if (erro_total <= tol):                        #Condição Convergência
            print("Parabéns! O Neurônio Convergiu\n") 
            print(f'Iterações até Convergência: {iter_num}\n')
            break
        
        erro_total        /= tam                       #Corrigir o Valor do erro para melhor compreensão
        erro_total_pt.append(erro_total)	
        if (i % t_plot == 0):
            erro_print.append(erro_total)
            print(erro_total)
        
    return erro_total_pt, iter_num
#%% Teste Laranja
dados    = np.loadtxt('input.txt')
resposta = []

with open('output.txt') as arq :
    for linha in arq:
        linha = linha.replace('\n','')
        if (linha == "maca"):
            resposta.append([0,1])
        else:
            resposta.append([1,0])
resposta = np.array(resposta)

cerebro   = layer(2,2)
cerebro_1 = layer(2,2)
cerebro_01 = layer(2,2)

start_time_05 = tm.time()

erro_01       = treinar(cerebro,  dados, resposta, 100e3, 0.01)
 
erro_05       = treinar(cerebro,  dados, resposta, 100e3, 0.05)
print(f'\nTempo Validação = {tm.time() - start_time_05} segundos\n')

start_time_1 = tm.time()
erro_1  = treinar(cerebro_1,dados, resposta, 100e3, 0.5)
print(f'\nTempo Validação = {tm.time() - start_time_1}segundos\n')
#%% Print dados Laranja
def reta(percep, x ,dados,i=1):
    w0, w1, w2 = percep.layer[i].pesos
    return (-w1*x - w0)/w2
x_ax = np.linspace(90, 130)
for i, val in enumerate(resposta):
    if all(val == np.array([0,1])):
        plt.plot(dados[i,0], dados[i,1],'ro')
    else:
        plt.plot(dados[i,0], dados[i,1],'o', color = 'orange')

plt.plot(x_ax, reta(cerebro, x_ax,dados))
plt.xlabel('Massa [g]')
plt.ylabel('pH')
plt.title('Reta de Classificação de Frutas')
plt.legend(['Maçã', 'Laranja'])
plt.style.use('seaborn-bright')
plt.show()

err_05 = range(erro_05[1]-1); err_1 = range(erro_1[1]-1); err_01 = range(erro_01[1]-1) 
plt.plot(err_05, erro_05[0], color = 'blue', alpha =0.5)
plt.plot(err_1, erro_1[0], color = 'magenta', alpha =0.5)
plt.plot(err_01, erro_01[0], color = 'yellow', alpha =0.5)

plt.xlabel(r'Iterações')
plt.ylabel('Erro Quadrático Médio')
plt.title(r'Erro para diferentes $\eta$ com função sigmoide')
plt.legend([r'$\eta = 0.05$', r'$\eta = 0.1$', r'$\eta = 0.01$'])
plt.show()
#%% Criação Base de Dados Aritificial Citrus
'''
Deixou-se esse trecho comentado pois não foi criado seed

massa_citrus_tang = np.array(np.round(np.random.uniform(111.3, 125.5,25),2)).T
massa_citrus_maca = np.array(np.round(np.random.uniform(90.1, 105.5,25),2)).T

massa_citrus      = np.concatenate((massa_citrus_tang, massa_citrus_maca))

ph_citrus_tang    = np.array(np.round(np.random.uniform(2.5, 5.5,25),2)).T
ph_citrus_maca    = np.array(np.round(np.random.uniform(3.5, 7.9,25),2)).T
ph_citrus         = np.concatenate((ph_citrus_tang, ph_citrus_maca))
    
dataset           = np.column_stack((massa_citrus,ph_citrus)) 
np.savetxt("citrus1.txt", dataset,fmt = '%.2f', delimiter = ',')
'''
#%% Teste Citrus
citrus_data   = np.array(pd.read_csv('citrus.txt'))
citrus_input  = np.zeros([len(citrus_data),2])
citrus_output = []

for i in range(len(citrus_data)):
    if citrus_data[i,2] == "laranja":
        citrus_output.append([1,0])
    else:
        citrus_output.append([0,1])
    for j in range(2):
        citrus_input[i,j] = citrus_data[i,j]
citrus_output = np.array(citrus_output)

pomar_1 = layer(2,2)
pomar_2 = layer(2,2)
pomar_3 = layer(2,2)

erro_pomar_1 = treinar(pomar_1,  citrus_input, citrus_output, 100e3, 0.05)
erro_pomar_2 = treinar(pomar_2,  citrus_input, citrus_output, 100e3, 0.1)
erro_pomar_3 = treinar(pomar_3,  citrus_input, citrus_output, 100e3, 0.01)
#%% Plots Citrus
plt.plot(citrus_input[:24,0], citrus_input[:24,1],'o', color = 'orange')
plt.plot(citrus_input[24:-1,0], citrus_input[24:-1,1],'o', color = 'red')
plt.plot(x_ax, reta(pomar_1, x_ax,citrus_input))
plt.title('Separação do Perceptron para 2 Classes')
plt.xlabel('Massa [g]')
plt.ylabel('pH')
plt.legend(['Laranja', 'Maçã'])
plt.show()

#Plot Erro Citrus
err_pomar_1 = range(erro_pomar_1[1]-1); err_pomar_2 = range(erro_pomar_2[1]-1);err_pomar_3 = range(erro_pomar_3[1]-1)

plt.plot(err_pomar_1, erro_pomar_1[0], color = 'cyan', alpha = 0.8)
plt.plot(err_pomar_2, erro_pomar_2[0], color = 'magenta', alpha = 0.6)
plt.plot(err_pomar_3, erro_pomar_3[0], color = 'yellow', alpha = 0.5)
plt.title(r'Gráfico de Erro para diferentas $\eta$ e função degrau')
plt.xlabel(r'Iterações')
plt.ylabel('Erro Quadrático Médio')
plt.legend([r'$\eta = 0.05$', r'$\eta = 0.1$', r'$\eta = 0.01$'])
plt.show()
print(f'Nº Iterações para diferentes valores de etas: \neta = 0.01: {erro_pomar_3[1]}\n\neta = 0.05: {erro_pomar_1[1]}\n\neta = 0.1: {erro_pomar_2[1]}')
#%% Teste Iris
dados_2    = np.array(pd.read_csv('iris.data'))
resposta_2 = []
input_iris = np.zeros((len(dados_2),4))

for i in range (len(dados_2)):
    if dados_2[i,4] == "Iris-setosa":
        resposta_2.append([1,0,0])
    if dados_2[i,4] == "Iris-versicolor":
        resposta_2.append([0,1,0])    
    if dados_2[i,4] == "Iris-virginica":
        resposta_2.append([0,0,1])
    for j in range(4):
        input_iris[i,j] = dados_2[i,j]
    
resposta_2      = np.array(resposta_2)
flower_picker   = layer(3,4)
start_time_iris = tm.time() #flower_picker.chamar(input_iris[0])
iris_try        = treinar(flower_picker, input_iris, resposta_2, 10e3,0.2)
finish_time_iris = tm.time()
#%%Visualizar Dados Iris
for i in range (len(input_iris)):
    np.set_printoptions(precision=3)
    print((flower_picker.chamar(input_iris[i])))
    
print(f'\nTempo Validação = {(finish_time_iris - start_time_iris)/60} minutos\n')
#%% Bootstrap
def bootstrap_val(test_times, input_dataset, output_dataset, classes, atributos, iterations = 15e3):
    indexes         =  [i for i,_ in enumerate(input_dataset)]
    indexes_test    = sk.utils.resample(indexes)
    indexes_train   = []
    erro_vetorizado  = []
    
    for k in range(test_times):
        print(f'\nBootstrap Iteration #{k+1}\n')
        for i,_ in enumerate(input_dataset):
            if not i in indexes_test:
                indexes_train.append(i)    
        
        input_test_train = np.array([input_dataset[i] for i in indexes_train])
        out_test_train   = np.array([output_dataset[i] for i in indexes_train])
        
        input_test_case  = np.array([input_dataset[i] for i in indexes_test])
        out_test_case    = np.array([output_dataset[i] for i in indexes_test])
        
        neuronio_test    = layer(classes,atributos)  #Testando para Iris (3 classes, 4 atributos)
        treinar(neuronio_test, input_test_train, out_test_train, iterations)
        
        erro_acumulado   = 0
        for i in range(len(input_test_case)):
            a = neuronio_test.chamar_abs(input_test_case[i])
            if not np.all(a == out_test_case[i]):
                erro_acumulado+=1
                print(f'erro bootstrap = {erro_acumulado/len(out_test_case)}')
            erro_vetorizado.append(erro_acumulado/len(out_test_case))
    return np.std(erro_vetorizado), np.average(erro_vetorizado)
#%% Bootstrap na Base Citrus
testtime = tm.time()
testando_bootstrap = bootstrap_val(10, citrus_input, citrus_output,2,2,100e3)
print(f'test_time = {(tm.time() - testtime)/60} minutos')
#%%Bootstrap na Base Iris
testtime2 = tm.time()
bootstrap_iris = bootstrap_val(3, input_iris, resposta_2, 3, 4, 10e3) 
print(f'test_time = {(tm.time() - testtime2)/60} minutos')

    