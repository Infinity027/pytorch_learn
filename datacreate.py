import numpy as np
import random as rd
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='out.csv', help="saving name of the dataframe")
    parser.add_argument("--size", type=int, default=1000, help="number of data")
    parser.add_argument("--complex", type=int, default=5, help="Complexity of dataset")
    parser.add_argument("--var", type=int, default=1, help="number of variable")
    parser.add_argument("--min_value", type=int, default=0, help="minimum value of variables")
    parser.add_argument("--max_value", type=int, default=5, help="maximum value of variables")
    parser.add_argument("--noise_prob", type=int, default=0, help="noise probablity in dataset")
    parser.add_argument("--normalize", type=str, default="std", help="normalization method - 'std','minmax','none'")
    args = parser.parse_args()
    if args.name.split('.')[1]!="csv":
        args.name += '.csv'
    return args

def minmax_normalize(data):
    return (data-data.min())/(data.max()-data.min())

def std_normalize(data):
    return (data-data.mean())/data.std()

class RegressionData():
    def __init__(self, var=1, min_value=-1, max_value=1, size=100, noise_prob=0, normalize="std"):
        self.data_size = size
        self.x = np.random.random([var,size])*(max_value-min_value) + min_value
        self.y = None
        self.tri_mode = ['cos','sin','tan']
        self.modes = ['pow','tri','log','linear']   #available function
        self.noise_prob = noise_prob                # noise probablity for to create data depend of every complex loop
        self.normalize = normalize                      #'std' or 'minmax' or 'none'

    #mode - 'add', 'assign'
    def getData(self, complex=1):
        for i in range(complex):
            if self.x.ndim > 1:
                num_var = np.random.randint(1,self.x.shape[0]+1)
            else:
                num_var = 1

            if i==0:
                self.y = self.createData(self.x,num_var)
            elif np.random.choice(["add","assign"]) == "add":
                self.y += self.createData(self.x,num_var)
            else:
                self.y = self.createData(self.y)
            if np.random.choice([True,False],p=[self.noise_prob,1-self.noise_prob]):
                self.y += np.random.normal(0,self.y.std()/10,self.data_size)

            if self.normalize=="std":
                self.y = std_normalize(self.y)
            elif self.normalize=="minmax":
                self.y = minmax_normalize(self.y)
            else:
                pass                   
                                  
        return self.x, self.y
    
    def createData(self, data, num_var=1):
        result = np.ones(self.x.shape[1])
        for i in range(num_var):
            modes = self.modes[:]
            if data.ndim>1:
                var_ind = np.random.randint(data.shape[0])
                x1 = data[var_ind]
            else:
                x1 = data
            #one posibility that previous result can be new independent variable
            if i!=0 and np.random.choice([True,False]):
                x1 = result 
            #remove log function if negative number present in data
            if np.any(x1<0):
                modes.remove('log')
            mode = rd.choice(modes)

            if mode=='pow':
                if np.any(x1<0):
                    power = np.random.choice([1,2,3])
                else:
                    power = np.random.rand()*3
                result *= self.pow(x1,power)
            elif mode=='log':
                result *= self.log(x1)
            elif mode=='linear':
                if np.random.choice([True,False]):
                    result *= (np.random.rand()*x1)
                else:
                    result += (np.random.rand()*x1)
            else:
                tri_mode = rd.choice(self.tri_mode)
                result *= self.tri(x1,tri_mode)

        return result
    
    def pow(self, data, power = 1):
        result = np.power(data,power)
        if np.any(np.isnan(result)):
            print("Warning: NaN encountered in power calculation")
            result = np.nan_to_num(result)
        return result
    
    def log(self, data):
        return np.log(np.clip(data, a_min=1e-6, a_max=None))
    
    def tri(self, data, mode='sin'):
        if mode=='sin':
            return np.sin(data)
        elif mode=='cos':
            return np.cos(data)
        else:
            return np.tan(np.mod(data, np.pi - 1e-6))

if __name__ == "__main__":
    args = parse_args()
    print("Total independent variable:",args.var)
    print("complexity:",args.complex)
    print(f"Range of variables:[{args.min_value}-{args.max_value}]")
    print("Noise Probablity:",args.noise_prob)
    print("Normalize method:",args.normalize)
    reg = RegressionData(args.var,args.min_value,args.max_value,args.size,args.noise_prob,args.normalize)
    X, y = reg.getData(args.complex)
    columns= []
    for i in range(args.var):
        columns.append('X'+str(i+1))
    df = pd.DataFrame(X.T, columns=columns)
    df['target'] = y
    df.to_csv(args.name, index=False)
    print("Data save in:",args.name)