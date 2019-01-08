"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology
"""
import logging
from abc import ABC, abstractmethod
from typing import List

from camera import Camera
from person import Person

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Visualizer(ABC):
    def __init__(self, people: List[Person]):
        self._people = people

    @abstractmethod
    def render(self, people: List[Person]=None) -> None:
        """
        Render current scene with tracked paths of people.
        :param people: [optional] updates people to render (replaces previously rendered people)
        """
        pass


class Plotter3D(Visualizer):
    def __init__(self, people: List[Person], cameras: List[Camera]):
        super().__init__(people)
        logger.debug('Using Plotter3D as Visualizer.')
        self._cameras = cameras

        # plot configuration
        self.plot_size_x = 600
        self.plot_size_y = 600
        self.plot_size_z = 150
        self.distance_plane_size_xy = 100
        self.ray_extender = 100
        self.intersection_line_extender = 100

        # plot
        from mpl_toolkits.mplot3d import Axes3D  # required for `ax = fig.add_subplot(111, projection='3d')`
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')  # requires `from mpl_toolkits.mplot3d import Axes3D`
        self._render_cameras()
        self._render_axis_labels()
        self.ax.legend()

        plt.ion()
        plt.show()

    def _render_axis_labels(self):
        self.ax.set_xlabel('x [cm]')
        self.ax.set_ylabel('y [cm]')
        self.ax.set_zlabel('z [cm]')

    def render(self, people: List[Person]=None):
        if people:
            self._people = people

        plt.cla()
        self._render_cameras()
        self._render_people_paths()
        self._render_axis_labels()
        self.ax.legend()
        plt.draw()
        plt.pause(0.001)  # NOTE: https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib

    def _render_people_paths(self):
        for person in self._people:
            person_xs = []
            person_ys = []
            person_zs = []
            for time_frame in person.time_frames:
                person_xs.append(time_frame.coordinates_3d[0])
                person_ys.append(time_frame.coordinates_3d[1])
                person_zs.append(time_frame.coordinates_3d[2])
            self.ax.plot(
                person_xs,
                person_ys,
                person_zs,
                'x:',
                label=person.name
            )
        logger.debug('Rendered {} people.'.format(len(self._people)))

    def _render_cameras(self):
        for camera in self._cameras:
            # camera point
            self.ax.scatter(
                [camera.position[0]],
                [camera.position[1]],
                [camera.position[2]],
                label=camera.name
            )
            # camera ray
            self.ax.plot(
                [camera.position[0], camera.position[0] + camera.orientation[0] * self.ray_extender],
                [camera.position[1], camera.position[1] + camera.orientation[1] * self.ray_extender],
                [camera.position[2], camera.position[2] + camera.orientation[2] * self.ray_extender],
                label=camera.name + ' ray'
            )
