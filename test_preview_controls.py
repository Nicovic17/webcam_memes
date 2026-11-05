import unittest

import cv2
import numpy as np

from preview_controls import MirrorControl
import test_webcam_domains as runtime_tests


class MirrorTests(unittest.TestCase):
    def test_click_toggles_only_inside_button_and_tracks_generation(self):
        control=MirrorControl(960,540)
        x1,y1,x2,y2=control.bounds
        control.on_mouse(cv2.EVENT_LBUTTONUP,1,1,0,None)
        self.assertTrue(control.mirrored)
        control.on_mouse(cv2.EVENT_LBUTTONUP,(x1+x2)//2,(y1+y2)//2,0,None)
        self.assertFalse(control.mirrored)
        self.assertEqual(control.generation,1)
        control.toggle()
        self.assertTrue(control.mirrored)
        self.assertEqual(control.generation,2)

    def test_button_is_preview_only_and_does_not_mutate_camera_frame(self):
        control=MirrorControl(960,540)
        frame=np.full((540,960,3),100,np.uint8)
        preview=control.draw(frame)
        np.testing.assert_array_equal(frame,np.full_like(frame,100))
        self.assertFalse(np.array_equal(preview,frame))
        x1,y1,x2,y2=control.bounds
        outside=np.ones(frame.shape[:2],bool)
        outside[y1:y2,x1:x2]=False
        np.testing.assert_array_equal(preview[outside],frame[outside])

    def test_keyboard_m_changes_domain_output_orientation(self):
        frame=np.full((90,160,3),30,np.uint8)
        frame[:,80:]=200
        for mode in ('gojo','sukuna'):
            case=runtime_tests.RuntimeTests()
            virtual,_,_,_=case.run_video(mode,keys=[ord('m'),0,ord('q')],source_frame=frame)
            self.assertEqual(virtual.send.call_count,3)
            np.testing.assert_array_equal(virtual.send.call_args_list[0].args[0],cv2.flip(frame,1))
            np.testing.assert_array_equal(virtual.send.call_args_list[1].args[0],frame)
