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

import cv2

from person import PersonTimeFrame, Person
from utils import utils

logger = logging.getLogger(__name__)


class PersonTracker(ABC):
    @property
    @abstractmethod
    def people(self) -> List[Person]:
        """
        :return: all tracked people including those not actively visible right now
        """
        pass

    @abstractmethod
    def track(self, frame: PersonTimeFrame) -> Person:
        """
        Track person from provided time frame.
        Assigns given time frame to already tracked person or creates new unique person for it.
        """
        pass


class NullTracker(PersonTracker):
    """
    Does **not** do any real person tracking, only passes data through.
    Creates person for each passed time frame.
    """
    _people: List[Person]

    def __init__(self):
        logger.debug('Using NullTracker as PersonTracker.')
        self._people = []

    @property
    def people(self) -> List[Person]:
        return self._people

    def track(self, frame: PersonTimeFrame) -> Person:
        person = Person(frame)
        logger.debug("New person! Person={}, 3D={}".format(person.name, person.time_frames[-1].coordinates_3d))
        self._people.append(person)
        logger.debug('Tracked people total: {}'.format(len(self._people)))
        return person


class HistogramTracker(PersonTracker):
    _people: List[Person]

    def __init__(self):
        logger.debug('Using HistogramTracker as PersonTracker.')
        self._people = []

    @property
    def people(self) -> List[Person]:
        return self._people

    def track(self, frame: PersonTimeFrame) -> Person:
        """
        If consecutive time frames have similar histogram they can belong to the same person.

        Multiple hist-cmp methods are used for matching. Frames are matched to the same person only if all methods
        agree on the best match.
        """
        if len(self._people) == 0:
            person = Person(frame)
            self._people.append(person)
            logger.debug("New! Person={}, 3D={}".format(person.name, person.time_frames[-1].coordinates_3d))
        else:
            view = frame.views[0]
            torso = view.get_torso_subimage()
            # cv2.imshow('torso view={}'.format(id(view)), torso)  # used for debugging
            # cv2.waitKey(1)  # used for debugging
            histogram = utils.calculate_flat_histogram(torso)

            intersect_histcmps = []
            hellinger_histcmps = []
            for person in self._people:
                view = person.time_frames[-1].views[0]
                torso = view.get_torso_subimage()
                # cv2.imshow('torso person.name={}, view={}'.format(person.name, id(view)), torso)  # used for debugging
                # cv2.waitKey(1)  # used for debugging

                front_histogram = utils.calculate_flat_histogram(torso)
                intersect_histcmps.append(cv2.compareHist(front_histogram, histogram, cv2.HISTCMP_INTERSECT))
                hellinger_histcmps.append(cv2.compareHist(front_histogram, histogram, cv2.HISTCMP_HELLINGER))

            # INTERSECT method: HIGHER value ~ HIGHER similarity
            best_intersect_match = max(x for x in intersect_histcmps if x is not None)
            i_intersect_match = intersect_histcmps.index(best_intersect_match)
            # HELLINGER method: LOWER value ~ HIGHER similarity
            best_hellinger_match = min(x for x in hellinger_histcmps if x is not None)
            i_hellinger_match = hellinger_histcmps.index(best_hellinger_match)

            logger.debug("best_intersect_match={}".format(best_intersect_match))
            logger.debug("best_hellinger_match={}".format(best_hellinger_match))
            if i_intersect_match == i_hellinger_match and best_hellinger_match < 0.3:  # TODO threshold
                person = self._people[i_intersect_match]
                person.time_frames.append(frame)
                # Do not append person to `self._people` here. It is already tracked.
                logger.debug("Tracked. Person={}, 3D={}".format(person.name, person.time_frames[-1].coordinates_3d))
            else:
                person = Person(frame)
                self._people.append(person)
                logger.debug("New! Person={}, 3D={}".format(person.name, person.time_frames[-1].coordinates_3d))

        logger.debug('Tracked people total: {}'.format(len(self._people)))
        return person


class PositionAndHistogramTracker(PersonTracker):
    @property
    def people(self) -> List[Person]:
        raise NotImplementedError  # TODO implement

    def track(self, frame: PersonTimeFrame) -> Person:
        # NOTE: If consecutive time frames have similar position and histogram they can belong to the same person.
        raise NotImplementedError  # TODO implement
