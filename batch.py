#Codigo para realizar las ejecuciones

import pandas as pd
import numpy as np
import tools as tl
from typing import Any
from sklearn.datasets import make_moons #type:ignore

def generate_non_linear_data(num_samples:int,seed:int)->tuple[np.ndarray,np.ndarray]:
    """Genera los datos no separables linealmente con make_moons() adaptado a clases 1 y -1"""
    X,y = make_moons(n_samples=num_samples, noise=0.15, random_state=seed)
    y = np.where(y == 0,-1,1)
    return (X,y)



def compare_hyperplanes(w1:np.ndarray|list[float],gamma1:float,w2:np.ndarray|list[float],gamma2:float,tol:float= 1e-4)->tuple[bool,bool]:
    """Devuelve (w1 == w2), (gamma1 == gamma2) con aproximacion de tol"""
    if isinstance(w1, list):
        w1 = np.array(w1)
    if isinstance(w2, list):
        w2 = np.array(w2)
    same_size_w = len(w1) == len(w2)

    if same_size_w:
        w1_eq_w2 = bool(np.all(np.abs(w1 - w2) <= tol))
    else:
        w1_eq_w2 = same_size_w

    g1_eq_g2 = bool(np.abs(gamma1 - gamma2) <= tol)

    return w1_eq_w2,g1_eq_g2



def experiment_iteration(X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray,seed:int,dataset:str,nu:float,num_exec:int,general_data: list[dict[str,Any]],hyperplane_validation: list[dict[str,Any]],tol:float= 1e-4,test_size:float=0.3,sigma_sq:float|None=None)->None:
    """Realiza una iteracion de un solve Primal, Dual sin Kernel, Dual con kernel y alamacena los resultados en general_data y hyperplane_validation"""

    #Primal
    w_prim,gamma_prim,s,primal_obj,time_prim, bar_opt_primal = tl.primal_solve(X_train.tolist(),y_train.tolist(),"primal_svm.mod",nu)
    acc_train_primal = tl.calculate_accuracy_from_hyperplane(w_prim,gamma_prim,X_train,y_train)
    acc_test_primal = tl.calculate_accuracy_from_hyperplane(w_prim,gamma_prim,X_test,y_test)
    general_data.append({
        "Dataset": dataset, "Nu": nu, "Semilla": seed, "Modelo": "Primal",
        "Funcion_Objetivo": primal_obj, "Tiempo": time_prim,"Bar_iter":bar_opt_primal,
        "Accuracy_Train": acc_train_primal, "Accuracy_Test":acc_test_primal, "Num_exec":num_exec
    })

    #Dual Sin Kernel
    k_lineal = np.matrix(X_train) * np.transpose(np.matrix(X_train))
    lamb_dual, dual_obj,time_dual , bar_opt_dual= tl.dual_solve(k_lineal,y_train.tolist(),"dual_svm.mod",nu)
    w_dual, gamma_dual = tl.recover_linear_hyperplane(X_train,y_train,lamb_dual,nu,tol)
    acc_train_dual = tl.calculate_accuracy_from_hyperplane(w_dual,gamma_dual,X_train,y_train)
    acc_test_dual = tl.calculate_accuracy_from_hyperplane(w_dual,gamma_dual,X_test,y_test)
    w_eq,g_eq =compare_hyperplanes(w_prim,gamma_prim,w_dual,gamma_dual)
    general_data.append({
        "Dataset": dataset, "Nu": nu, "Semilla": seed, "Modelo": "DualSinKernel",
        "Funcion_Objetivo": dual_obj, "Tiempo": time_dual,"Bar_iter":bar_opt_dual,
        "Accuracy_Train": acc_train_dual, "Accuracy_Test":acc_test_dual, "Num_exec":num_exec
    })
    hyperplane_validation.append({
        "w_eq":w_eq, "g_eq":g_eq,"Dataset": dataset, "Nu": nu, "Num_exec":num_exec
    })

    #Dual con Kernel
    k_rbf = tl.compute_gaussian_kernel(X1=X_train,sigma_sq=sigma_sq)
    lamb_rbf, rbf_obj,time_rbf , bar_opt_rbf= tl.dual_solve(k_rbf,y_train.tolist(),"dual_svm.mod",nu)
    acc_train_rbf = tl.calculate_accuracy_dual_rbf(k_rbf,lamb_rbf,X_train,y_train,X_train,y_train,nu,sigma_sq,tol)
    acc_test_rbf = tl.calculate_accuracy_dual_rbf(k_rbf,lamb_rbf,X_train,y_train,X_test,y_test,nu,sigma_sq,tol)
    general_data.append({
        "Dataset": dataset, "Nu": nu, "Semilla": seed, "Modelo": "DualRBF",
        "Funcion_Objetivo": rbf_obj, "Tiempo": time_rbf,"Bar_iter":bar_opt_rbf,
        "Accuracy_Train": acc_train_rbf, "Accuracy_Test":acc_test_rbf, "Num_exec":num_exec
    })



def experiments(valores_nu:list[float],datasets:list[str],seed:int,num_exec:int,general_data: list[dict[str,Any]],hyperplane_validation: list[dict[str,Any]],tol:float= 1e-4,test_size:float=0.3,sigma_sq:float|None=None)->None:
    """Ejecuta un experimento por cada nu y por cada dataset"""
    for dataset in datasets:
            if dataset.isdigit():
                X,y = generate_non_linear_data(int(dataset),seed)
                name_dataset:str = f"Moons_{dataset}"
            elif dataset == "wdbc.data":
                X,y = tl.process_wdbc(dataset)
                name_dataset = "WDBC"
            else:
                X,y = tl.generator_preprocess(dataset,4) #type:ignore
                #en este caso es una list[float], pero split ya lo convertira a numpy
                name_dataset = f"P{dataset[1:len(dataset)-4]}"
            X_train, X_test, y_train, y_test = tl.split_dataset(X,y,test_size,seed)
            
            for nu in valores_nu:
                experiment_iteration(X_train, X_test, y_train, y_test,seed,name_dataset,nu,num_exec,general_data,hyperplane_validation,tol,test_size,sigma_sq)

                
        






def store_data(general_data: list[dict[str,Any]],hyperplane_validation: list[dict[str,Any]])->None:
    """Almacena los resultados en reporte_svm_general_data.csv y reporte_svm_hyperplane_validation.csv"""

    general_data_df = pd.DataFrame(general_data)
    hyperplane_validation_df = pd.DataFrame(hyperplane_validation)
    print("Almacenando resultados en .csv ...")
    general_data_df.to_csv("reporte_svm_general_data.csv", index=False)
    hyperplane_validation_df.to_csv("reporte_svm_hyperplane_validation.csv",index=False)
    print("Almacenamiento de resultados completado con éxito en  reporte_svm_general_data.csv y reporte_svm_hyperplane_validation.csv")
    print("Fin de batch.py")

def main()->None:
    valores_nu:list[float] = [0.01,0.1, 0., 1.0, 10.0, 100.0]
    datasets:list[str] = ["points_100.dat","points_1000.dat","points_2000.dat","wdbc.data","100","1000","2000"] #los ultimos casos son para swiss_roll, "points_5000.dat","5000"
    semilla:int = 77214914
    num_iters:int = 3 #numero de repeticion de los experimentos
    general_data: list[dict[str,Any]] = []
    hyperplane_validation: list[dict[str,Any]] = []
    for i in range(num_iters):
        print(f"==============Iteración {i+1}/{num_iters}==============")
        experiments(valores_nu,datasets,semilla,i+1,general_data,hyperplane_validation)
    store_data(general_data,hyperplane_validation)

main()