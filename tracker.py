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

from person import PersonTimeFrame, Person

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
        logger.debug("New person! Time={}, Person={}, 3D={}"
                     .format(person.time_frames[-1].time, person.name, person.time_frames[-1].coordinates_3d))
        self._people.append(person)
        logger.debug('Tracked people total: {}'.format(len(self._people)))
        return person


class PositionAndHistogramTracker(PersonTracker):
    @property
    def people(self) -> List[Person]:
        raise NotImplementedError  # TODO implement

    def track(self, frame: PersonTimeFrame) -> Person:
        # NOTE: If consecutive time frames have similar position and histogram they can belong to the same person.
        raise NotImplementedError  # TODO implement
