import numpy as np
import pandas as pd
import plotly.express as px
from typing import *
from abc import ABC, abstractmethod


class AbstractVisualizer(ABC):
    @abstractmethod
    def array_to_dataframe(self, points_coords: np.ndarray, points_assign: np.ndarray, clust_coords: np.ndarray):
        raise NotImplemented

    @abstractmethod
    def show(self, *args, **kwargs):
        raise NotImplemented


class VisualizeCluster(AbstractVisualizer):
    def __init__(self, x0: int, x1: int) -> None:
        """
        # Inputs
            x0 (int) : index of the first coordinate to visualize
            x1 (int) : index of the second coordinate to visualize
        """
        super().__init__()
        self.x0 = x0
        self.x1 = x1

    def array_to_dataframe(self, points_coords: np.ndarray, points_assign: np.ndarray, clust_coords: np.ndarray, *args, **kwargs):
        """
        # Inputs:
            points_coords (np.ndarray) : [num_points, num_coordinates] 
            points_assign (np.ndarray) : [num_points, ] with in each cell the centroid index associated with the point at the same position
            clust_coords (np.ndarray) : [num_clusters, num_coordinates]
        """
        df = pd.DataFrame(np.concatenate((points_coords, clust_coords), axis=0), columns=[
            f"x_{i}" for i in range(points_coords.shape[1])])
        df["cluster"] = np.concatenate((points_assign, np.array(
            [i for i in range(len(clust_coords))])), axis=0)
        df["iscentroid"] = np.concatenate(
            (np.zeros(len(points_coords)), np.ones(len(clust_coords))), axis=0)+1
        return df

    def show(self, points_coords: np.ndarray, points_assign: np.ndarray, clust_coords: np.ndarray, title: str):
        """
        # Inputs:
            points_coords (np.ndarray) : [num_points, num_coordinates] 
            points_assign (np.ndarray) : [num_points, ] with in each cell the centroid index associated with the point at the same position
            clust_coords (np.ndarray) : [num_clusters, num_coordinates]
        """
        df = self.array_to_dataframe(
            points_coords, points_assign, clust_coords)
        fig = px.scatter(df, x=f"x_{self.x0}", y=f"x_{self.x1}",
                         symbol="cluster", size="iscentroid", title=title, color="cluster")
        fig.update_layout(
            xaxis_title=f'$x_{self.x0}$',
            yaxis_title=f'$x_{self.x1}$'
        )
        fig.show()


class VisualizeClusterList(VisualizeCluster):
    def array_to_dataframe(self, points_coords: np.ndarray, LClusters: List[Dict[str, np.ndarray]], *args, **kwargs):
        """
        # Inputs:
            points_coords (np.ndarray) : [num_points, num_coordinates] 
            points_assign (np.ndarray) : [num_points, ] with in each cell the centroid index associated with the point at the same position
            clust_coords (np.ndarray) : [num_clusters, num_coordinates]
        """
        df_list = [super(VisualizeClusterList, self).array_to_dataframe(
            points_coords, **clust) for clust in LClusters]  # type: ignore
        for i in range(len(df_list)):
            df_list[i] = df_list[i].assign(step=i)
        df = pd.concat(df_list)
        return df

    def show(self, points_coords: np.ndarray, Lclusters: List[Dict[str, np.ndarray]]):
        df = self.array_to_dataframe(points_coords, Lclusters)
        fig = px.scatter(df, x=f"x_{self.x0}", y=f"x_{self.x1}",
                         symbol="cluster", size="iscentroid", animation_frame="step", color="cluster"
                         )
        # Thanks to https://community.plotly.com/t/dynamic-animation-title-annotation/38747/6
        for button in fig.layout.updatemenus[0].buttons:
            button['args'][1]['frame']['redraw'] = True
        for k in range(len(fig.frames)):
            fig.frames[k]['layout'].update(title_text=Lclusters[k]['title'])
        fig.update_layout(
            xaxis_title=f'$x_{self.x0}$',
            yaxis_title=f'$x_{self.x1}$'
        )
        fig.show()


class VisualizationCallback:
    def __init__(self, visualizer: AbstractVisualizer) -> None:
        self.points_coords: Optional[np.ndarray] = None
        self.clusters = []
        self.visualizer: AbstractVisualizer = visualizer

    def register_points(self, points_coords: np.ndarray):
        self.points_coords = points_coords

    def register_cluster(self, title: str, points_assign: np.ndarray, clust_coords: np.ndarray):
        self.clusters.append(
            {"title": title, "points_assign": points_assign,
                "clust_coords": clust_coords}
        )

    def show(self, id_clust: int = None):
        assert self.points_coords is not None, "No points registered. Call register_points before show"
        if id_clust is not None:
            self.visualizer.show(
                self.points_coords, **self.clusters[id_clust])
        else:
            self.visualizer.show(self.points_coords, self.clusters)
