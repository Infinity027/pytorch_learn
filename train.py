import torch 
import torch.optim as optim
from torch import nn
from dataloader import CustomDataset
from torch.utils.data import DataLoader
from model import RegressionModel
import argparse
import os
import pandas as pd
from plot import draw_figure

#define number of layer and neuron present in every layer
LAYERS = [4,2,1]
MODEL_SAVE_NAME = "model1.pth"
FIGURE_NAME = "plot1.png"

class EarlyStopping: #Earlystopping to prevent overfitting
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_state = model.state_dict()
        elif val_loss >= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = val_loss
            self.best_model_state = model.state_dict() #save the best model weights
            self.counter = 0
        return False

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='out.csv', help="path of the dataframe")
    parser.add_argument("--training_size", type=float, default=0.5, help="training percentage")
    parser.add_argument("--eval_size", type=int, default=0.3, help="evalution percentage")
    parser.add_argument("--test_size", type=int, default=0.2, help="test percentage")
    parser.add_argument("--resume", type=str, help="path of trained model")
    args = parser.parse_args()
    if not os.path.exists(args.data):
        print("Inavlid data path -- try again--")
        args.data = 'out.csv'

    if args.resume: 
        if not os.path.exists(args.resume):
            print("Trained Model not found!!!")
            args.resume = None 
    
    if args.training_size+args.eval_size+args.test_size!=1:
        print("Inavalid Data distribution!! --set as below--")
        args.training_size = 0.5
        args.eval_size = 0.3
        args.test_size = 0.2
        print("Training percentage:",args.training_size*100)
        print("evalution percentage:",args.eval_size*100)
        print("test percentage:",args.test_size*100)
    return args

def train_model(model, traindata, evaldata, optimizer, criterion, num_epochs=100, earlystopping=None):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in traindata:
            predictions = model(batch_X).squeeze(1) 
            loss = criterion(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss/len(traindata))
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}",end='|')
        if evaldata!=None:
            val_loss = test_model(model, evaldata, criterion)
            val_losses.append(val_loss)
            print(f"Evalution Loss: {val_losses[-1]:.4f}")
        if earlystopping!=None and earlystopping(val_loss, model):
            print(f"EarlyStopping called model stopped at {epoch+1} epoch")
            earlystopping.load_best_model(model)
            break
    return model, train_losses, val_losses

def test_model(model, evaldata, criterion):
    epoch_loss = 0.0
    with torch.inference_mode():
        for batch_X, batch_y in evaldata:
            predictions = model(batch_X).squeeze(1) 
            loss = criterion(predictions, batch_y)
            epoch_loss += loss.item()
    return epoch_loss/len(evaldata)

if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.data)
    input_size = df.shape[1]-1
    traindata = df[:int((args.training_size)*len(df))]
    #data divide into training, validation and test
    if args.test_size!=None:
        evaldata = df[int((args.training_size)*len(df)):int((args.training_size+args.eval_size)*len(df))]
        testdata = df[int((args.training_size+args.eval_size)*len(df)):]
    else:
        evaldata = df[(args.training_size)*len(df):]
    print("Training data size:",len(traindata))
    print("Test data size:",len(testdata))
    print("Evalution data size:",len(evaldata))

    traindata = CustomDataset(traindata)
    evaldata = CustomDataset(evaldata)
    testdata = CustomDataset(testdata)

    traindata = DataLoader(traindata, batch_size=64, shuffle=True)
    evaldata = DataLoader(evaldata, batch_size=32)
    testdata = DataLoader(testdata, batch_size=32)
    #model define
    layers = [4,2,1]
    model = RegressionModel(layers,input_size)
    #if already trained model present then load the weight of previouse trained model
    if args.resume:
        model.load_state_dict(torch.load(args.resume, weights_only=True))
    #loss function define
    loss_fn = nn.MSELoss()
    #optimizer define
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    #Define Earlystopping
    es = EarlyStopping(patience=5, delta=0.01)
    model, train_losses, val_losses = train_model(model, traindata, evaldata, optimizer, loss_fn, 100, es)
    draw_figure(train_losses, val_losses, save_name = FIGURE_NAME)
    print("model save as:", MODEL_SAVE_NAME)
    torch.save(obj=model, f=MODEL_SAVE_NAME)
