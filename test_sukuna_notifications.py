"""Exercise notifications through Sukuna's real background/subject/skull stack."""
import unittest
from unittest.mock import patch

import numpy as np

from sukuna_effect import SukunaDomain, ENTRY_SECONDS, EXIT_SECONDS


class SukunaNotificationTests(unittest.TestCase):
    def test_opening_and_active_cards_stay_behind_person_and_skulls(self):
        width, height = 640, 360
        domain = SukunaDomain(width, height)
        frame = np.full((height, width, 3), 70, np.uint8)
        mask = np.zeros((height, width), np.float32)
        mask[:, 270:370] = 1
        domain.enter(10)
        for age in (2.2, ENTRY_SECONDS+3):
            shown = domain.render(frame, mask, 10+age)
            with patch.object(domain.notifications, 'render', side_effect=lambda scene, *args, **kwargs: scene):
                baseline = domain.render(frame, mask, 10+age)
            self.assertFalse(np.array_equal(shown, baseline))
            np.testing.assert_array_equal(shown[:, 270:370], baseline[:, 270:370])
            if age > ENTRY_SECONDS:
                # Fully opaque skull pixels must occlude messages as well as
                # the shrine, even when a card crosses the bottom of the frame.
                black = domain.skulls.render(np.zeros_like(frame), age, 1)
                white = domain.skulls.render(np.full_like(frame, 255), age, 1)
                opaque = (black == white).all(axis=2)
                self.assertGreater(opaque.sum(), 100)
                np.testing.assert_array_equal(shown[opaque], baseline[opaque])
        # Every notification is hidden if the person fills the whole frame.
        full_mask = np.ones_like(mask)
        shown = domain.render(frame, full_mask, 18)
        with patch.object(domain.notifications, 'render', side_effect=lambda scene, *args, **kwargs: scene):
            np.testing.assert_array_equal(domain.render(frame, full_mask, 18), shown)

    def test_cancel_collapse_reset_and_reopen_remove_or_restart_messages(self):
        domain = SukunaDomain(320, 180)
        frame = np.full((180, 320, 3), 50, np.uint8)
        mask = np.zeros((180, 320), np.float32)
        domain.enter(10)
        first = domain.render(frame, mask, 12.2)
        domain.collapse(12.3)  # Cancel during entry.
        np.testing.assert_array_equal(domain.render(frame, mask, 12.4), frame)
        domain.enter(20)
        np.testing.assert_array_equal(domain.render(frame, mask, 22.2), first)
        domain.render(frame, mask, 20+ENTRY_SECONDS)
        domain.collapse(26)
        np.testing.assert_array_equal(domain.render(frame, mask, 26+EXIT_SECONDS), frame)
        domain.enter(30)
        domain.render(frame, mask, 32.2)
        domain.reset()
        np.testing.assert_array_equal(domain.render(frame, mask, 33), frame)


if __name__ == '__main__':
    unittest.main()
