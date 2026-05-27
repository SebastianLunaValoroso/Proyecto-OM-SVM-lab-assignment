#Codigo de Python
from yogi import Yogi
from amplpy import AMPL #type:ignore
import re
import numpy as np
import pandas as pd #type:ignore

#np y pd anaden mucho overhead


def isfloat(s: str) -> bool:
    """Devuelve True si s es un float"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def generator_preprocess(file:str,dim_x:int, dim_y:int=-1)->tuple[list[list[float]],list[float]]:
    """Elimina los '*' y separa los puntos en una matrix x (atributos de los puntos) y un vector y (clases de los puntos) de file.

    Devuelve:
    - una matriz x de dimensiones m x dim_x
    - un vector y de dimensiones 1 x m
    
    Prec: file debe haber sido generado con gensvmdat

    :param: dim_y: corresponde al numero de elementos de y

    """
    dim_known:bool = dim_y > 0 #es True si se especifica dim_y
    if dim_known:
        x:list[list[float]] = [[0. for _ in range(dim_x)] for _ in range(dim_y)]
        y:list[float] = [0. for _ in range(dim_y)]
        i:int = 0
    else:
        x = []
        y = []
    with open(file) as file: #type:ignore
        reader:Yogi = Yogi(file) #type:ignore
        elem = reader.scan(float)
        while elem is not None:
            if dim_known:
                x[i][0] = elem
                for j in range(1,dim_x):
                    x[i][j] = reader.read(float)
                elem_y:str = reader.read(str)
                if isfloat(elem_y):
                    y[i] = float(elem_y)
                else:
                    y[i] = float(elem_y[:len(elem_y) -1])
                i+=1
            else:
                x.append([reader.read(float) if j >0 else elem for j in range(dim_x)])
                elem_y = reader.read(str)
                if isfloat(elem_y):
                    y.append(float(elem_y))
                else:
                    y.append(float(elem_y[:len(elem_y) -1]))
            elem = reader.scan(float)
                
    return (x,y)



def post_process_time_raw(time_raw:str)->float:
    """Utiliza una regular expression para encontrar el tiempo y lo devuelve. Devuelve -1. en caso de error"""
    sol = re.search(r"Solver time = ([0-9.]+)s", time_raw)
    if sol is not None and isfloat(sol.group(1)):
        return float(sol.group(1))
    return -1.



def post_process_primal_solve(w_raw:list[tuple[int,float]],gamma:float,s_raw:list[tuple[int,float]],primal_obj:float,time_raw:str)->tuple[list[float],float,list[float],float,float]:
    """Devuelve (w,gamma,s,primal_obj,time) en un formato mas adecuado"""
    w:list[float] = [num for i, num in w_raw]
    s:list[float] = [num for i, num in s_raw]
    time:float = post_process_time_raw(time_raw)
    #eventualmente podriamos crear una regular expression para conseguir el valor de la func objetivo de cplex, pero aqui nos quedamos con primal_obj
    return (w,gamma,s,primal_obj,time)



def post_process_dual_solve(lamb_raw:list[tuple[int,float]],primal_obj:float,time_raw:str)->tuple[list[float],float,float]:
    """Devuelve (lamb,primal_obj,time) en un formato mas adecuado"""
    lamb:list[float] = [num for i, num in lamb_raw]
    time:float = post_process_time_raw(time_raw)
    return (lamb,primal_obj,time)



def primal_solve(x:list[list[float]], y:list[float], modelo:str, nu:float)->tuple[list[float],float,list[float],float,float]: #modificar lo de None
    """Resuelve el Primal SVM con los datos de entrada y devuelve la solucion
    
    Solver: CPLEX

    Prec: len(x) > 0 y len(y) > 0

    :param: modelo: Tiene que ser un archivo .mod
    """
    m:int = len(y)
    n:int = len(x[0])

    ampl = AMPL()
    ampl.read(modelo)

    # Asignacion de los datos
    ampl.param["n"] = n
    ampl.param["m"] = m
    ampl.param["nu"] = nu

    ampl.param["y"] = {i: y[i-1] for i in range(1,m+1)}
    ampl.param["x"] = {
        (i,j): x[i-1][j-1]
        for i in range(1, m+1)
        for j in range(1, n+1)

    }

    #Solve
    ampl.option["solver"] = "cplex"
    ampl.option["cplex_options"] = "timing=1" #si usamos cplex
    ampl.solve()
    

    #Recuperacion de la solucion
    time_raw:str = ampl.getOutput("solve;") #para capturar la salida de solve, como el tiempo
    primal_obj:float = ampl.get_objective("primal_svm").value() #es diferente de lo que meustra cplex por pantalla
    w = ampl.get_variable("w").to_list()
    gamma = ampl.get_variable("gamma").value()
    s = ampl.get_variable("s").to_list()

    #Post Processing
    return post_process_primal_solve(w,gamma,s,primal_obj,time_raw)



def dual_solve(k:np.matrix,y:list[float], modelo:str, nu:float)->tuple[list[float],float,float]:
    m:int = len(y)

    ampl = AMPL()
    ampl.read(modelo)

    # Asignacion de los datos
    ampl.param["m"] = m
    ampl.param["nu"] = nu

    ampl.param["y"] = {i: y[i-1] for i in range(1,m+1)}

    k_df = pd.DataFrame(
        k,
        columns= [i for i in range(1,m+1)],
        index= [i for i in range(1,m+1)]
    )
    ampl.get_parameter("K").set_values(k_df)

    #Solve
    ampl.option["solver"] = "cplex"
    ampl.option["cplex_options"] = "timing=1" #si usamos cplex
    ampl.solve()

    #Recuperacion de la solucion
    time_raw:str = ampl.getOutput("solve;") #para capturar la salida de solve, como el tiempo
    primal_obj:float = ampl.get_objective("dual_svm").value() #es diferente de lo que meustra cplex por pantalla
    lamb = ampl.get_variable("lambda").to_list()

    #Post Processing
    return post_process_dual_solve(lamb,primal_obj,time_raw)




