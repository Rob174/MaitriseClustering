import numpy as np
import pandas as pd
import plotly.express as px

def show_clustering(points_coords: np.ndarray,points_assign: np.ndarray,clust_coords: np.ndarray, title: str, x0: int = 0, x1: int = 1):
    """
    # Inputs:
        points_coords (np.ndarray) : [num_points, num_coordinates] 
        points_assign (np.ndarray) : [num_points, ] with in each cell the centroid index associated with the point at the same position
        clust_coords (np.ndarray) : [num_clusters, num_coordinates]
    """
    df = pd.DataFrame(np.concatenate((points_coords,clust_coords),axis=0),columns=[f"x_{i}" for i in range(points_coords.shape[1])])
    df["cluster"] = np.concatenate((points_assign,np.array([i for i in range(len(clust_coords))])),axis=0)
    df["iscentroid"] = np.concatenate((np.zeros(len(points_coords)),np.ones(len(clust_coords))),axis=0)+1
    fig = px.scatter(df, x=f"x_{x0}", y=f"x_{x1}", symbol="cluster", size="iscentroid",title=title)
    fig.update_layout(
        xaxis_title=f'$x_{x0}$',
        yaxis_title=f'$x_{x1}$'
    )
    fig.show()
    