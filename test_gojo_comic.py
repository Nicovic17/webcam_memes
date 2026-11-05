import unittest
from unittest.mock import patch

import numpy as np

from gojo_comic import COMIC_SECONDS,GojoComicOverlay
from gojo_effect import GojoDomain


class ComicTests(unittest.TestCase):
    def test_appears_only_during_opening_and_fades_out(self):
        overlay=GojoComicOverlay(640,360)
        frame=np.full((360,640,3),50,np.uint8)
        np.testing.assert_array_equal(overlay.render(frame.copy(),None,0),frame)
        full=overlay.render(frame.copy(),None,1)
        fading=overlay.render(frame.copy(),None,COMIC_SECONDS-.2)
        self.assertGreater(np.abs(full.astype(float)-frame).sum(),
                           np.abs(fading.astype(float)-frame).sum())
        self.assertFalse(np.array_equal(full,frame))
        np.testing.assert_array_equal(overlay.render(frame.copy(),None,COMIC_SECONDS),frame)
        np.testing.assert_array_equal(full[:,:320],frame[:,:320])

    def test_follows_head_smoothly_on_right_and_preserves_anchor_during_misses(self):
        overlay=GojoComicOverlay(640,360)
        frame=np.zeros((360,640,3),np.uint8)
        mask=np.zeros((360,640),np.float32)
        shown=overlay.render(frame.copy(),mask,1,face_box=(.4,.25,.6,.55))
        self.assertFalse(shown[:,:320].any())
        self.assertTrue(shown[:,320:].any())
        start=overlay.position.copy()
        self.assertGreaterEqual(start[0],320)
        self.assertLess(abs(start[1]+overlay.sprite.shape[0]*.52-144),1)
        overlay.render(frame.copy(),mask,2,face_box=(.5,.4,.7,.7))
        self.assertGreater(overlay.position[0],start[0])
        self.assertGreater(overlay.position[1],start[1])
        retained=overlay.position.copy()
        overlay.render(frame.copy(),mask,2.1)
        np.testing.assert_array_equal(overlay.position,retained)
        overlay.reset()
        self.assertIsNone(overlay.position)

    def test_head_at_right_edge_keeps_balloon_visible_to_its_left_on_right_half(self):
        overlay=GojoComicOverlay(960,540)
        frame=np.zeros((540,960,3),np.uint8)
        overlay.render(frame,None,1,face_box=(.85,.3,.98,.6))
        x,y=overlay.position
        h,w=overlay.sprite.shape[:2]
        self.assertGreaterEqual(x,480)
        self.assertLess(x+w,.85*960)
        self.assertGreaterEqual(y,0)
        self.assertLessEqual(y+h,540)

    def test_actual_domain_restarts_balloon_and_reset_removes_it(self):
        frame=np.full((180,320,3),75,np.uint8)
        mask=np.zeros((180,320),np.float32)
        domain=GojoDomain(320,180)
        domain.enter(10)
        shown=domain.render(frame,mask,11)
        with patch.object(domain.comic,'render',side_effect=lambda scene,mask,age,**kwargs:scene):
            without=domain.render(frame,mask,11)
        self.assertGreater(np.abs(shown.astype(float)-without).sum(),100)
        domain.reset()
        np.testing.assert_array_equal(domain.render(frame,mask,12),frame)
        domain.enter(20)
        np.testing.assert_array_equal(domain.render(frame,mask,21),shown)
        domain.collapse(21)
        np.testing.assert_array_equal(domain.render(frame,mask,21),frame)


if __name__=='__main__':
    unittest.main()
