import matplotlib.pyplot as plt
import plotly.express as px
import argparse
import os
import pandas as pd

            
def draw_figure(train_losses, val_losses, save_name="plot.png"):
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(len(train_losses)),train_losses, 'b', label="Training Loss")
    plt.plot(range(len(val_losses)),val_losses, 'r', label="Validation Loss")
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig(save_name, dpi=fig.dpi)
    fig.clear()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="out.csv", help="Path to csv file")
    parser.add_argument("--output_path", type=str, default="plot_image", help="save path of plot images")
    
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    return args

def scatter2D(x,y):
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(x,y)
    plt.set_xlabel('X1')
    plt.set_ylabel('X2')
    plt.show()

def scatter3D(x,y,z,elev_angel=0,azim_angel=-90,roll_angel=0):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='o')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    ax.view_init(elev=elev_angel, azim=azim_angel, roll=roll_angel)
    plt.show()

def interactive_Scatter3D(x,y,z):
# Create 3D scatter
    fig = px.scatter_3d(x=x,y=y,z=z)
    fig.update_traces(marker=dict(size=2,line=dict(width=0.01)))
    fig.show()

if __name__=='__main__':
    args = parse_args()
    df = pd.read_csv(args.data)
    if df.shape[1]>4:
        print("dimension is more than 3 not possible to draw any graph")
    elif df.shape[1]==3:
        df.sort_values(by=['X1'], inplace=True)