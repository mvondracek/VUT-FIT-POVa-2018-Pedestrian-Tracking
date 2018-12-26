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

from person import PersonTimeFrame, PersonView
from utils import utils

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
    def __init__(self):
        self.front_img_orig = None
        self.side_img_orig = None

    # FIXME: this won't be needed if "whole person image" is implemented and passed from detector->person_view->match
    def set_original_images(self, front_image, side_image):
        """ Each step set original images, so it is possibel to extract a torso area from them. """
        self.front_img_orig = front_image
        self.side_img_orig = side_image

    # TODO optimalization: differ clear image and image with people -> whole person box -> could be better for histograms
    def match(self, front_views: List[PersonView], side_views: List[PersonView]) -> List[PersonTimeFrame]:
        front_histograms = []
        side_histograms = []
        for view in front_views:
            torso = self.get_torso_subimage(view, self.front_img_orig)
            front_histograms.append(utils.calculate_flat_histogram(torso))

        for view in side_views:
            torso = self.get_torso_subimage(view, self.side_img_orig)
            side_histograms.append(utils.calculate_flat_histogram(torso))

        """
        # DEBUG CHECK
        cv2.namedWindow('tst', cv2.WINDOW_NORMAL)
        for view in front_views:
            cv2.imshow('tst', view.person_image)
            cv2.waitKey()
        for view in side_views:
            cv2.imshow('tst', view.person_image)
            cv2.waitKey()
        """

        # FIXME add a threshold? Currently the values are too diverse and any threshold is useless.
        # Multiple hist-cmp methods are used for matching. Views are matched only if all methods agree on the best match.
        results = []
        for index, front_hist in enumerate(front_histograms):
            intersect_histcmps = []
            hellinger_histcmps = []
            for side_hist in side_histograms:
                if side_hist is None:  # already matched histograms are removed to not match twice
                    intersect_histcmps.append(None)
                    hellinger_histcmps.append(None)
                else:
                    intersect_histcmps.append(cv2.compareHist(front_hist, side_hist, cv2.HISTCMP_INTERSECT))
                    hellinger_histcmps.append(cv2.compareHist(front_hist, side_hist, cv2.HISTCMP_HELLINGER))

            best_intersect_match = max(x for x in intersect_histcmps if x is not None)  # INTERSECT method: HIGHER value ~ HIGHER similarity
            i_intersect_match = intersect_histcmps.index(best_intersect_match)
            best_hellinger_match = min(x for x in hellinger_histcmps if x is not None)  # HELLINGER method: LOWER value ~ HIGHER similarity
            i_hellinger_match = hellinger_histcmps.index(best_hellinger_match)
            if i_intersect_match == i_hellinger_match:
                results.append(PersonTimeFrame([front_views[index], side_views[i_intersect_match]]))
                side_histograms[i_intersect_match] = None  # already matched a front_view, don't compare it any further
                # TODO if one person is left in both screens, it is matched even if totally different - threshold could fix it, but threshold is impossible now

        return results

    @staticmethod
    def get_torso_subimage(view, image):
        """
        Extract a subimage of just a torso for given person view. Should be better for histograms since contains less surroundings than the whole person box.
        :param view: PersonView to find the torso coordinates for
        :param image: original image, the torso is extracted from this image
        :return: subimage containing only the torso
        """
        image_width = image.shape[1]
        body_height = int(utils.euclidean_distance(view.pose_top_coordinate, view.pose_bottom_coordinate))
        half_body_width = int(body_height / 6)  # an average body's width from side is about one third of the height (half of the height from front)

        pose_top_left = (min(view.pose_top_coordinate[0], view.pose_bottom_coordinate[0]), min(view.pose_top_coordinate[1], view.pose_bottom_coordinate[1]))
        pose_bottom_right = (max(view.pose_top_coordinate[0], view.pose_bottom_coordinate[0]), max(view.pose_top_coordinate[1], view.pose_bottom_coordinate[1]))

        roi_top_left = (max(0, pose_top_left[0] - half_body_width), pose_top_left[1])
        roi_bottom_right = (min(image_width, pose_bottom_right[0] + half_body_width), pose_bottom_right[1])

        return image[roi_top_left[1]:roi_bottom_right[1]+1, roi_top_left[0]:roi_bottom_right[0]+1]


class PositionBasedHistogramMatcher(PersonMatcher):
    """
    # FIXME Edit the following note... This is just note to describe an idea
    Matching based only based on histograms may be not enough. Imagine cam1 detected p1 and p2, but cam2 detected p2 and p3.
    In this case p2 would match p2 correctly, but p1 may match p3, because that is the most matching histogram for p1.
    To prevent that, location verification could be done based on intersection of line of sight of cam1 and cam2. If we draw
    line from cam1 to p1, and line from cam2 to p1, they intersects in a pointX. Then we draw circle with a center in the pointX
    and a diameter 50 cm. Then look at the best matchin person for p1 from cam1, that is p3 from cam2, and check if p3 is estimated
    to be inside of the circle (based on distance to cam2). If p3 is inside, then probably p1 == p2, if is outside, then p1 =/= p3.
    """

    def match(self, front_views: List[PersonView], side_views: List[PersonView]):
        raise NotImplementedError  # TODO histogram matching with verification based on person location
