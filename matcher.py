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

from person import PersonTimeFrame, PersonView

logger = logging.getLogger(__name__)


class PersonMatcher(ABC):
    @abstractmethod
    def match(self, front_views: List[PersonView], side_views: List[PersonView]) -> List[PersonTimeFrame]:
        """
        Match together views (`PersonView`) of the same people as time frame (`PersonTimeFrame`).
        :param front_views: views of detected people from front camera (in undefined order)
        :param side_views: views of detected people from side camera (in undefined order)
        :return: time frame for each person detected in two views (1 front and 1 side)
        """
        pass


class NullMatcher(PersonMatcher):
    """
    Does **not** do any real person matching, only passes data through.
    Simply puts together views, which are on the same position in input lists. If input lists are not the same size,
    remaining views are discarded.
    """
    def __init__(self):
        logger.debug('Using NullMatcher as PersonMatcher.')

    def match(self, front_views: List[PersonView], side_views: List[PersonView]) -> List[PersonTimeFrame]:
        views_min = min(len(front_views), len(side_views))
        views_max = max(len(front_views), len(side_views))
        matched = []
        for i in range(views_min):
            matched.append(PersonTimeFrame([front_views[i], side_views[i]]))
        if views_min != views_max:
            logger.warning("NullMatcher discarded {0} remaining views (`PersonView`)".format((views_max - views_min)))
        return matched


class HistogramMatcher(PersonMatcher):
    def match(self, front_views: List[PersonView], side_views: List[PersonView]) -> List[PersonTimeFrame]:
        raise NotImplementedError  # TODO Implement histogram matching. See `openpose.main.person_synchronization`
