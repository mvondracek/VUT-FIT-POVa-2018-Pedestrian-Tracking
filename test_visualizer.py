"""
Pedestrian Tracking
2018
POVa - Computer Vision
FIT - Faculty of Information Technology
BUT - Brno University of Technology
"""
import time
import unittest
from unittest import TestCase

from camera import Camera
from config import FOCAL_LENGTH_CAMERA_M, FOCAL_LENGTH_CAMERA_F
from person import Person, PersonTimeFrame
from visualizer import Plotter3D


class TestPlotter3D(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.camera_m = Camera(
            name='m (front camera)',
            focal_length=FOCAL_LENGTH_CAMERA_M,
            position=(0, 0, 147),
            orientation=(0, 1, 0)
        )
        self.camera_f = Camera(
            name='f (side camera)',
            focal_length=FOCAL_LENGTH_CAMERA_F,
            position=(-200, 0, 147),
            orientation=(1, 1, 0)
        )

    @unittest.skip("This is not an automated test. Visual check would be required.")
    def test_render(self):
        people = []
        time_frame = PersonTimeFrame([])
        time_frame.coordinates_3d = (0, 300, 147)
        alice = Person(time_frame, name='Alice')
        people.append(alice)
        time_frame = PersonTimeFrame([])
        time_frame.coordinates_3d = (0, 600, 147)
        bob = Person(time_frame, name='Bob')
        people.append(bob)

        visualizer = Plotter3D(people, [self.camera_m, self.camera_f])
        visualizer.render()
        print('render')

        time.sleep(0.5)
        time_frame = PersonTimeFrame([])
        time_frame.coordinates_3d = (50, 300, 147)
        alice.time_frames.append(time_frame)
        time_frame = PersonTimeFrame([])
        time_frame.coordinates_3d = (0, 400, 147)
        bob.time_frames.append(time_frame)

        visualizer.render()
        print('render')

        time.sleep(0.5)
        time_frame = PersonTimeFrame([])
        time_frame.coordinates_3d = (50, 350, 147)
        alice.time_frames.append(time_frame)
        time_frame = PersonTimeFrame([])
        time_frame.coordinates_3d = (0, 200, 147)
        bob.time_frames.append(time_frame)

        visualizer.render()
        print('render')

        time.sleep(0.5)
        time_frame = PersonTimeFrame([])
        time_frame.coordinates_3d = (-50, 350, 147)
        alice.time_frames.append(time_frame)

        time_frame = PersonTimeFrame([])
        time_frame.coordinates_3d = (-200, 200, 147)
        bob.time_frames.append(time_frame)

        visualizer.render()
        print('render')

        print('closing in 2 s')
        time.sleep(2)
