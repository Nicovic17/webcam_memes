from queue import Queue
from threading import Event
import unittest
from unittest.mock import Mock,patch

import cv2
import numpy as np

from video_pipeline import LatestFrameCapture,tracking_image,LatestAnalysis,FramePacer
from sukuna_skulls import ComicSkullForeground
import run3


class QueueCamera:
    def __init__(self):
        self.frames=Queue()
        self.released=False
        self.read_backlog=Event()

    def read(self):
        item=self.frames.get()
        self.read_backlog.set()
        return item

    def release(self):
        self.released=True
        self.frames.put((False,None))


class VideoPipelineTests(unittest.TestCase):
    def test_camera_opener_uses_latest_frame_capture_for_all_modes(self):
        camera=Mock()
        first=np.zeros((720,1280,3),np.uint8)
        camera.isOpened.return_value=True
        camera.read.return_value=(True,first)
        camera.get.return_value=cv2.VideoWriter_fourcc(*'MJPG')
        wrapped=Mock()
        with patch.object(run3.cv2,'VideoCapture',return_value=camera), \
             patch.object(run3,'LatestFrameCapture',return_value=wrapped) as factory, \
             patch('builtins.print'):
            capture,pending=run3.open_camera()
        self.assertIs(capture,wrapped)
        self.assertIsNone(pending)
        factory.assert_called_once_with(camera,first)
        camera.set.assert_any_call(cv2.CAP_PROP_FPS,run3.VIRTUAL_CAMERA_FPS)
        camera.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE,1)
        self.assertEqual(camera.set.call_args_list[-1].args,
                         (cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG')))

    def test_pacer_counts_camera_and_render_time_instead_of_sleeping_after_send(self):
        pacer=FramePacer(30)
        with patch('video_pipeline.time.perf_counter',return_value=.060), \
             patch('video_pipeline.time.sleep') as sleep:
            pacer.wait(0)
            sleep.assert_not_called()
        with patch('video_pipeline.time.perf_counter',return_value=.010), \
             patch('video_pipeline.time.sleep') as sleep:
            pacer.wait(0)
            self.assertAlmostEqual(sleep.call_args.args[0],1/30-.010)

    def test_slow_inference_never_blocks_polling_and_skips_pending_frames(self):
        entered,release=Event(),Event()
        processed=[]
        cleanup=Mock()
        def slow(value):
            processed.append(value)
            if value==1:
                entered.set()
                release.wait(2)
            return value
        worker=LatestAnalysis(slow,cleanup)
        try:
            worker.submit(1)
            self.assertTrue(entered.wait(1))
            for _ in range(10):
                self.assertIsNone(worker.latest())  # Rendering can continue now.
            worker.submit(2)
            worker.submit(3)
            release.set()
            with worker._condition:
                self.assertTrue(worker._condition.wait_for(
                    lambda:worker._result is not None and worker._result.data==3,timeout=2))
            self.assertEqual(processed,[1,3])
        finally:
            release.set()
            worker.close()
        cleanup.assert_called_once()

    def test_worker_surfaces_inference_failure_and_closes_models(self):
        cleanup=Mock()
        worker=LatestAnalysis(Mock(side_effect=ValueError('model failed')),cleanup)
        worker.submit(None)
        worker._thread.join(timeout=1)
        with self.assertRaisesRegex(RuntimeError,'model failed'):
            worker.latest()
        worker.close()
        cleanup.assert_called_once()

    def test_discards_backlog_and_returns_newest_frame(self):
        camera=QueueCamera()
        first=np.zeros((2,2,3),np.uint8)
        capture=LatestFrameCapture(camera,first)
        try:
            np.testing.assert_array_equal(capture.read()[1],first)
            for value in (1,2,3):
                camera.frames.put((True,np.full_like(first,value)))
            with capture._condition:
                self.assertTrue(capture._condition.wait_for(lambda:capture._sequence>=4,timeout=2))
            # A slow consumer gets frame 3, rather than stale frames 1 and 2.
            self.assertEqual(int(capture.read()[1][0,0,0]),3)
        finally:
            camera.frames.put((False,None))
            capture.release()
        self.assertTrue(camera.released)
        self.assertFalse(capture._thread.is_alive())
        capture.release()  # Closing twice must be safe.

    def test_read_failure_wakes_consumer_and_cleanup_closes_camera(self):
        camera=Mock()
        camera.read.side_effect=RuntimeError('camera disconnected')
        capture=LatestFrameCapture(camera,np.zeros((2,2,3),np.uint8))
        capture._thread.join(timeout=1)
        self.assertTrue(capture.read()[0])
        self.assertEqual(capture.read(),(False,None))
        capture.release()
        camera.release.assert_called_once()

    def test_tracking_preserves_aspect_ratio_and_output_frame(self):
        frame=np.full((540,960,3),(1,2,3),np.uint8)
        image=tracking_image(frame)
        self.assertEqual((image.width,image.height),(640,360))
        np.testing.assert_array_equal(image.numpy_view()[0,0],(3,2,1))
        self.assertEqual(frame.shape,(540,960,3))
        small=tracking_image(frame[:180,:320])
        self.assertEqual((small.width,small.height),(320,180))

    def test_skull_blending_handles_fractional_emergence_at_full_resolution(self):
        skulls=ComicSkullForeground(960,540)
        frame=np.full((540,960,3),50,np.uint8)
        for age in (2.2,2.7,3.1,5):
            shown=skulls.render(frame.copy(),age,np.float64(.67))
            self.assertEqual(shown.shape,frame.shape)
            self.assertEqual(shown.dtype,np.uint8)


if __name__=='__main__':unittest.main()
