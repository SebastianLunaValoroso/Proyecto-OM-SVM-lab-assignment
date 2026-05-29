#Codigo de Python
from yogi import Yogi
from amplpy import AMPL #type:ignore
import re
import numpy as np
import pandas as pd #type:ignore
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional, Union


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
    w:list[float] = [num for _, num in w_raw]
    s:list[float] = [num for _, num in s_raw]
    time:float = post_process_time_raw(time_raw)
    #eventualmente podriamos crear una regular expression para conseguir el valor de la func objetivo de cplex, pero aqui nos quedamos con primal_obj
    return (w,gamma,s,primal_obj,time)



def post_process_dual_solve(lamb_raw:list[tuple[int,float]],primal_obj:float,time_raw:str)->tuple[list[float],float,float]:
    """Devuelve (lamb,primal_obj,time) en un formato mas adecuado"""
    lamb:list[float] = [num for _, num in lamb_raw]
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



def dual_solve(k: np.matrix,y:list[float], modelo:str, nu:float)->tuple[list[float],float,float]:
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

def process_wdbc(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Lee el archivo wdbc.data del repositorio, elimina el ID, 
    mapea las clases M -> 1.0 y B -> -1.0 y devuelve matrices de NumPy.
    """
    X_raw: list[list[float]] = []
    y_raw: list[float] = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            
            # Segunda columna es la etiqueta
            label = 1.0 if parts[1] == 'M' else -1.0
            y_raw.append(label)
            
            # De la tercera columna en adelante son las características
            features = [float(val) for val in parts[2:]]
            X_raw.append(features)
            
    return np.array(X_raw), np.array(y_raw)


def split_dataset(X: np.ndarray, y: np.ndarray, test_size: float = 0.3, seed: int = 77214914) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide el conjunto de datos en entrenamiento y prueba utilizando una semilla fija.
    Acepta tanto listas de Python como arrays de NumPy.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def compute_gaussian_kernel(X1: np.ndarray, X2: np.ndarray, sigma_sq: Optional[float] = None) -> np.ndarray:
    """
    Calcula la matriz de Kernel Gaussiano entre dos conjuntos de puntos X1 y X2.
    Si sigma_sq es None, calcula el valor por defecto basado en la varianza de X1.
    """
    if sigma_sq is None:
        # Definición basada en la recomendación del profesor (heurística de sklearn)
        sigma_sq = (X1.shape[1] * X1.var()) / 2.0

    # Distancia euclídea al cuadrado eficiente: ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2*x1*x2^T
    norm_X1 = np.sum(X1**2, axis=1).reshape(-1, 1)
    norm_X2 = np.sum(X2**2, axis=1).reshape(1, -1)
    dists_sq = norm_X1 + norm_X2 - 2 * np.dot(X1, X2.T)
    
    return np.exp(-dists_sq / (2 * sigma_sq))

def calculate_accuracy_linear(X_test: np.ndarray, y_test: np.ndarray, w: Union[list[tuple[int, float]], list[float], np.ndarray], gamma: float) -> float:
    """
    Calcula la accuracy para modelos lineales (Primal o Dual lineal) usando w y gamma.
    """
    if isinstance(w, list) and len(w) > 0 and isinstance(w[0], tuple):
        w_dict = dict(w)
        w_arr: np.ndarray = np.array([w_dict[i] for i in sorted(w_dict.keys())], dtype=np.float64)
    else:
        w_arr = np.array(w, dtype=np.float64)
        
    # Ecuación del hiperplano: f(x) = X_test * w + gamma
    scores: np.ndarray = np.dot(X_test, w_arr) + gamma
    predictions: np.ndarray = np.sign(scores)
    predictions[predictions == 0] = 1.0  # Frontera por convenio a clase positiva
    
    accuracy: float = float(np.mean(predictions == y_test) * 100)
    return accuracy

def calculate_accuracy_rbf(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, 
                           lambda_val: Union[list[tuple[int, float]], list[float], np.ndarray], nu: float, sigma_sq: Optional[float] = None) -> float:
    """
    Calcula la accuracy para el modelo Dual con Kernel RBF utilizando el truco del kernel.
    Primero recupera gamma de los vectores de soporte y luego predice.
    """
    if isinstance(lambda_val, list) and len(lambda_val) > 0 and isinstance(lambda_val[0], tuple):
        lambda_dict = dict(lambda_val)
        lambda_arr: np.ndarray = np.array([lambda_dict[i] for i in sorted(lambda_dict.keys())], dtype=np.float64)
    else:
        lambda_arr = np.array(lambda_val, dtype=np.float64)
    
    # 1. Reconstrucción de la matriz de Kernel interna de Train para extraer gamma
    K_train: np.ndarray = compute_gaussian_kernel(X_train, X_train, sigma_sq)
    
    # 2. Identificar Vectores de Soporte puros (0 < lambda < nu) para despejar gamma
    tol: float = 1e-4
    sv_idx: np.ndarray = np.where((lambda_arr > tol) & (lambda_arr < nu - tol))[0]
    if len(sv_idx) == 0:
        sv_idx = np.where(lambda_arr > tol)[0]  # Fallback tolerante si están al límite
        
    gammas: List[float] = []
    for i in sv_idx:
        margin_term: float = float(np.sum(lambda_arr * y_train * K_train[:, i]))
        gammas.append((1.0 / y_train[i]) - margin_term)
    gamma_rec: float = float(np.mean(gammas))
    
    # 3. Kernel entre los puntos nuevos de Test (filas) y los conocidos de Train (columnas)
    K_test_train: np.ndarray = compute_gaussian_kernel(X_test, X_train, sigma_sq)
    
    # Clasificación no lineal: sign( sum(lambda_j * y_j * K(x_test, x_train)) + gamma )
    scores: np.ndarray = np.dot(K_test_train, lambda_arr * y_train) + gamma_rec
    predictions: np.ndarray = np.sign(scores)
    predictions[predictions == 0] = 1.0
    
    accuracy: float = float(np.mean(predictions == y_test) * 100)
    return accuracy

def recover_linear_hyperplane(X_train: np.ndarray, y_train: np.ndarray, lambda_val: list, nu: float) -> tuple[np.ndarray, float]:
    """
    Recupera el vector de pesos 'w' y el sesgo 'gamma' a partir de la solucion dual lineal.
    """
    # Si la estructura viene de ampl.to_list(), puede ser una lista de tuplas (id, valor)
    if len(lambda_val) > 0 and isinstance(lambda_val[0], tuple):
        lambda_dict = dict(lambda_val)
        lamb = np.array([lambda_dict[i] for i in sorted(lambda_dict.keys())], dtype=np.float64)
    else:
        lamb = np.array(lambda_val, dtype=np.float64)
        
    # w = sum(lambda_i * y_i * x_i)
    w = np.sum((lamb * y_train)[:, np.newaxis] * X_train, axis=0)
    
    # Buscar un Vector de Soporte puro (0 < lambda < nu) para calcular gamma
    tol = 1e-4
    sv_idx = np.where((lamb > tol) & (lamb < nu - tol))[0]
    if len(sv_idx) == 0:
        sv_idx = np.where(lamb > tol)[0] # Fallback tolerante
        
    gammas = []
    for idx in sv_idx:
        gamma_k = (1.0 / y_train[idx]) - np.dot(w, X_train[idx])
        gammas.append(gamma_k)
        
    gamma_final = float(np.mean(gammas)) if len(gammas) > 0 else 0.0
    return w, gamma_final
