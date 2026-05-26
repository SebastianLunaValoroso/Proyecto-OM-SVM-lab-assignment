#Codigo de Python
from yogi import Yogi

def isfloat(s: str) -> bool:
    """Devuelve True si s es un float"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def generator_preprocess(file:str,dim_x:int, dim_y:int=-1)->tuple[list[list[float]],list[float]]:
    """Elimina los '*' y prepara los datos de file para la creacion de un .dat para los modelos svm de AMPL.

    Devuelve:
    - una matriz x de dimensiones m x dim_x
    - un vector y de dimensiones dim_i x 1
    
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





def main()->None:
    x,y = generator_preprocess('points_1000.dat',4)
    dim_y = len(y)
    if dim_y > 0:
        dim_x = len(x[0])
        for i in range(dim_y):
            for j in range(dim_x):
                print(x[i][j],end=" ")
            print(y[i])

main()