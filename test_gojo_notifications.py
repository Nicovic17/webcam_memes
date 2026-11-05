"""Verify actual domain composition, timing and foreground protection."""
import unittest
from unittest.mock import patch

import numpy as np

from gojo_effect import GojoDomain, ENTRY_SECONDS, EXIT_SECONDS
from gojo_notifications import GojoNotifications, MESSAGES


class NotificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layer = GojoNotifications(640, 360)

    def test_variety_entry_rush_and_sparse_continuous_background(self):
        self.assertGreaterEqual(len(set(text for _, text, _, _ in MESSAGES)), 10)
        self.assertEqual(self.layer.flights(0), ())
        entry = self.layer.flights(2.6)
        self.assertGreaterEqual(len(entry), 8)
        self.assertGreater(len(set(f.card for f in entry)), 5)
        self.assertGreater(max(f.width for f in entry), min(f.width for f in entry)*2)
        for age in (6, 12, 30, 5000):
            flights = self.layer.flights(age)
            self.assertGreater(len(flights), 0)
            self.assertLessEqual(len(flights), 3)
        self.assertNotEqual(self.layer.flights(6), self.layer.flights(6.2))

    def test_cards_animate_without_drawing_over_person(self):
        frame = np.full((360, 640, 3), 20, np.uint8)
        mask = np.zeros((360, 640), np.float32)
        mask[:, 260:380] = 1
        first = self.layer.render(frame.copy(), 2.2, mask)
        later = self.layer.render(frame.copy(), 2.5, mask)
        self.assertFalse(np.array_equal(first, later))
        self.assertFalse(np.array_equal(first, frame))
        np.testing.assert_array_equal(first[:, 260:380], frame[:, 260:380])
        np.testing.assert_array_equal(self.layer.render(frame.copy(), 0, mask), frame)
        np.testing.assert_array_equal(self.layer.render(frame.copy(), 3,
                                      np.ones_like(mask)), frame)

    def test_domain_opens_with_cards_then_closes_and_replays_cleanly(self):
        width, height = 320, 180
        domain = GojoDomain(width, height)
        frame = np.full((height, width, 3), 70, np.uint8)
        mask = np.zeros((height, width), np.float32)
        domain.enter(10)
        for age in (2.2, ENTRY_SECONDS+3):
            shown = domain.render(frame, mask, 10+age)
            with patch.object(domain.notifications, 'render', side_effect=lambda scene, *args, **kwargs: scene):
                baseline = domain.render(frame, mask, 10+age)
            self.assertFalse(np.array_equal(shown, baseline))
        domain.collapse(18)
        np.testing.assert_array_equal(domain.render(frame, mask, 18+EXIT_SECONDS), frame)
        domain.enter(30)
        replay = domain.render(frame, mask, 32.2)
        domain.reset()
        domain.enter(40)
        np.testing.assert_array_equal(domain.render(frame, mask, 42.2), replay)
        domain.reset()
        np.testing.assert_array_equal(domain.render(frame, mask, 43), frame)


if __name__ == '__main__':
    unittest.main()
