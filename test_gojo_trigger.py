"""Gesture geometry and noisy tracking sequences, independent of a real webcam."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import mediapipe as mp
import numpy as np

from gojo_effect import crossed_finger_score, GojoGestureGate, GojoPoseRecognizer


def hand_points(kind='v', angle=0, mirrored=False, width=640, height=360):
    # An ordinary V, with ring/pinky folded towards the palm.
    points = np.array([
        (0, 0), (-.3, .25), (-.65, .45), (-.8, .65), (-1, .75),
        (-.45, .8), (-.55, 1.4), (-.62, 1.75), (-.7, 2.05),
        (0, 1), (.08, 1.6), (.25, 1.95), (.4, 2.25),
        (.38, .85), (.45, 1.3), (.45, .9), (.35, .65),
        (.7, .65), (.85, 1), (.85, .7), (.65, .4),
    ], np.float32)
    if kind == 'crossed':
        points[7:9] = ((-.3, 1.8), (-.1, 2.1))
        points[11:13] = ((-.4, 1.8), (-.7, 2.05))
    elif kind == 'relaxed':
        points[10:13] = ((.15, 1.5), (-.2, 1.6), (-.4, 1.65))
        points[14:17] = ((.45, 1.6), (.48, 1.5), (.45, 1.4))
    elif kind == 'open':
        points[14:17] = ((.5, 1.4), (.6, 1.8), (.7, 2.1))
        points[18:21] = ((.9, 1.1), (1.1, 1.4), (1.2, 1.7))
    elif kind in ('fist', 'pointing'):
        points[10:13] = ((.1, 1.4), (.2, 1.0), (.1, .65))
        if kind == 'fist':
            points[6:9] = ((-.5, 1.3), (-.5, .9), (-.4, .6))
    if mirrored:
        points[:, 0] *= -1
    radians = np.radians(angle)
    rotation = np.array(((np.cos(radians), -np.sin(radians)),
                         (np.sin(radians), np.cos(radians))))
    points = points @ rotation.T
    pixels = points * (60, -60) + (width*.5, height*.75)
    return [SimpleNamespace(x=float(x/width), y=float(y/height)) for x, y in pixels]


class GojoTriggerTests(unittest.TestCase):
    def test_v_crossed_and_relaxed_fingers_work_on_either_rotated_hand(self):
        for kind in ('v', 'crossed', 'relaxed'):
            for mirrored in (False, True):
                for angle in (-65, 0, 65):
                    with self.subTest(kind=kind, mirrored=mirrored, angle=angle):
                        self.assertGreaterEqual(crossed_finger_score(
                            hand_points(kind, angle, mirrored), 640, 360), .75)

    def test_open_palm_fist_and_single_index_do_not_trigger(self):
        for kind in ('open', 'fist', 'pointing'):
            self.assertEqual(crossed_finger_score(hand_points(kind), 640, 360), 0)
        self.assertEqual(crossed_finger_score([], 640, 360), 0)
        invalid = hand_points()
        invalid[8].x = float('nan')
        self.assertEqual(crossed_finger_score(invalid, 640, 360), 0)
        tiny = [SimpleNamespace(x=.5, y=.5) for _ in range(21)]
        self.assertEqual(crossed_finger_score(tiny, 640, 360), 0)

    def test_recognizer_accepts_actual_v_landmarks_and_rejects_open_hand(self):
        detector = Mock()
        recognizer = GojoPoseRecognizer(detector)
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.zeros((360, 640, 3), np.uint8))
        detector.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[hand_points('v', width=640, height=640)])
        self.assertTrue(recognizer.recognize_pose(image, 100))
        self.assertTrue(all(0 <= value <= 1 for value in recognizer.anchor))
        detector.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[hand_points('open', width=640, height=640)])
        self.assertFalse(recognizer.recognize_pose(image, 200))

    def test_short_tracking_miss_preserves_hold(self):
        gate = GojoGestureGate()
        self.assertFalse(gate.update(True, 0))
        self.assertFalse(gate.update(True, .16))
        self.assertFalse(gate.update(False, .28))
        self.assertTrue(gate.update(True, .48))

    def test_long_tracking_loss_requires_a_fresh_hold(self):
        gate = GojoGestureGate()
        gate.update(True, 0)
        gate.update(True, .1)
        gate.update(False, .5)
        self.assertFalse(gate.update(True, .6))
        self.assertFalse(gate.update(True, .8))
        self.assertTrue(gate.update(True, 1.01))

    def test_hold_fires_once_and_needs_release_before_next_activation(self):
        gate = GojoGestureGate()
        self.assertFalse(gate.update(True, 0))
        self.assertFalse(gate.update(True, .2))
        self.assertTrue(gate.update(True, .41))
        self.assertFalse(gate.update(True, .6))
        self.assertFalse(gate.update(True, .8))
        gate.update(False, .9)
        gate.update(False, 1.31)
        self.assertFalse(gate.update(True, 1.4))
        self.assertFalse(gate.update(True, 1.6))
        self.assertTrue(gate.update(True, 1.81))


if __name__ == '__main__':
    unittest.main()
