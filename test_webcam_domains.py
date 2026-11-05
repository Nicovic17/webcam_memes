"""Headless integration checks; no real camera, sound playback or OBS required."""
from contextlib import ExitStack
from itertools import count
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import cv2
import numpy as np

import run3
import webcam_domains as app
from video_pipeline import AnalysisResult
from preview_controls import MirrorControl


class ImmediateAnalysis:
    """Deterministic inference adapter for runtime controls/rendering tests.

    Real non-blocking scheduling is tested separately with a stalled worker.
    """
    def __init__(self,process,cleanup=None):
        self.process,self.cleanup=process,cleanup
        self.result=None
        self.sequence=0
    def submit(self,data):
        self.sequence+=1
        self.result=AnalysisResult(self.sequence,self.process(data),0)
    def latest(self):return self.result
    def close(self):
        if self.cleanup:self.cleanup()


class BlockedAnalysis(ImmediateAnalysis):
    """No inference completes while the simulated video keeps running."""
    def submit(self,data):pass


class SelectionTests(unittest.TestCase):
    def test_keyboard_and_cancel(self):
        for key,expected in (('1','meme'),('2','gojo'),('3','sukuna'),('q',None),('\x1b',None)):
            with ExitStack() as stack:
                for name in ('namedWindow','setMouseCallback','imshow','destroyWindow'):
                    stack.enter_context(patch.object(app.cv2,name))
                stack.enter_context(patch.object(app.cv2,'waitKey',return_value=ord(key)))
                stack.enter_context(patch.object(app.cv2,'getWindowProperty',return_value=1))
                self.assertEqual(app.choose_effect(),expected)

    def test_click_selects_each_effect(self):
        for index,(mode,_,_) in enumerate(app.EFFECTS):
            callback=[None]
            with ExitStack() as stack:
                stack.enter_context(patch.object(app.cv2,'setMouseCallback',side_effect=lambda title,fn:callback.__setitem__(0,fn)))
                for name in ('namedWindow','imshow','destroyWindow'):
                    stack.enter_context(patch.object(app.cv2,name))
                stack.enter_context(patch.object(app.cv2,'waitKey',side_effect=lambda ms:(callback[0](cv2.EVENT_LBUTTONUP,200,140+index*108,0,None) or 255)))
                stack.enter_context(patch.object(app.cv2,'getWindowProperty',return_value=1))
                self.assertEqual(app.choose_effect(),mode)

    def test_dispatch_preserves_meme_and_only_starts_selected_domain(self):
        for mode in ('meme','gojo','sukuna',None):
            with patch.object(app,'choose_effect',return_value=mode), \
                 patch.object(run3,'run_meme_webcam') as meme, \
                 patch.object(app,'run_domain_webcam') as domain, \
                 patch.object(run3,'open_camera') as camera:
                run3.main()
                self.assertEqual(meme.call_count,int(mode=='meme'))
                self.assertEqual(domain.call_count,int(mode in ('gojo','sukuna')))
                camera.assert_not_called()
                if mode in ('gojo','sukuna'):
                    self.assertEqual(domain.call_args.args,(mode,camera))


class ResourceTests(unittest.TestCase):
    def test_task_failure_closes_previously_created_models(self):
        segmenter,hands=Mock(),Mock()
        with patch.object(app.vision.ImageSegmenter,'create_from_options',return_value=segmenter), \
             patch.object(app.vision.HandLandmarker,'create_from_options',side_effect=(hands,RuntimeError('fallback failed'))):
            with self.assertRaisesRegex(RuntimeError,'fallback failed'):
                with ExitStack() as resources:
                    app.create_domain_tasks('sukuna',resources)
        segmenter.close.assert_called_once()
        hands.close.assert_called_once()

    def test_all_task_models_close_after_success(self):
        for mode in ('gojo','sukuna'):
            segmenter,hands,fallback=Mock(),Mock(),Mock()
            with patch.object(app.vision.ImageSegmenter,'create_from_options',return_value=segmenter), \
                 patch.object(app.vision.HandLandmarker,'create_from_options',side_effect=(hands,fallback)):
                with ExitStack() as resources:
                    app.create_domain_tasks(mode,resources)
            segmenter.close.assert_called_once()
            hands.close.assert_called_once()
            self.assertEqual(fallback.close.call_count,int(mode=='sukuna'))

    def test_person_mask_resizes_and_handles_missing_detection(self):
        segmenter=Mock()
        segmenter.segment_for_video.return_value=SimpleNamespace(confidence_masks=[])
        empty=app.person_mask(segmenter,None,1,160,90)
        self.assertEqual(empty.shape,(90,160))
        self.assertEqual(empty.dtype,np.float32)
        self.assertFalse(empty.any())
        segmenter.segment_for_video.return_value=SimpleNamespace(confidence_masks=[SimpleNamespace(numpy_view=lambda:np.ones((8,8),np.float32))])
        np.testing.assert_array_equal(app.person_mask(segmenter,None,2,160,90),np.ones((90,160),np.float32))


class RuntimeTests(unittest.TestCase):
    def run_video(self,mode,pose=False,keys=None,virtual_error=None,send_error=None,gesture=False,
                  worker_type=ImmediateAnalysis,source_frame=None):
        import vanish_sound
        capture,virtual,sound,segmenter,recognizer=Mock(),Mock(),Mock(),Mock(),Mock()
        face_tracker=Mock()
        face_tracker.detect_for_video.return_value=SimpleNamespace(face_landmarks=[[
            SimpleNamespace(x=.4,y=.25),SimpleNamespace(x=.6,y=.55)]])
        frame=(np.full((90,160,3),100,np.uint8) if source_frame is None else source_frame)
        capture.read.return_value=(True,frame.copy())
        recognizer.recognize_pose.return_value=pose
        recognizer.anchor=(.5,.4)
        recognizer.last_hint='Mostra le mani.'
        virtual.device='OBS Virtual Camera'
        outputs=[]
        if keys is None:
            keys=[0]*11+[ord('q')] if gesture else list(map(ord,('v','v','r',' ','q')))
        clock=iter((0,.1,.2,.3,.4,.5,.6,.7,.8,.9,2,5.6,6)) if gesture else count(0,.1)
        def tasks(mode,resources):
            resources.callback(segmenter.close)
            resources.callback(recognizer.close)
            return segmenter,recognizer
        with ExitStack() as stack:
            stack.enter_context(patch('builtins.print'))
            stack.enter_context(patch.object(app,'create_domain_tasks',side_effect=tasks))
            stack.enter_context(patch.object(app,'LatestAnalysis',worker_type))
            stack.enter_context(patch.object(app,'FramePacer'))
            stack.enter_context(patch.object(app,'create_gojo_face_tracker',return_value=face_tracker))
            factory=stack.enter_context(patch.object(vanish_sound,'VanishSound',return_value=sound))
            stack.enter_context(patch.object(app.pyvirtualcam,'Camera',side_effect=virtual_error,return_value=virtual))
            stack.enter_context(patch.object(app,'person_mask',return_value=np.zeros((90,160),np.float32)))
            stack.enter_context(patch.object(app.time,'monotonic',side_effect=lambda:next(clock)))
            for name in ('namedWindow','resizeWindow','destroyAllWindows','setMouseCallback'):
                stack.enter_context(patch.object(app.cv2,name))
            stack.enter_context(patch.object(app.cv2,'getWindowProperty',return_value=1))
            stack.enter_context(patch.object(app.cv2,'waitKey',side_effect=keys))
            stack.enter_context(patch.object(app.cv2,'imshow',side_effect=lambda title,img:outputs.append(img.copy())))
            if send_error:
                virtual.send.side_effect=send_error
            camera_opener=Mock(return_value=(capture,frame.copy()))
            if virtual_error or send_error:
                with self.assertRaises(RuntimeError):
                    app.run_domain_webcam(mode,camera_opener,width=160,height=90)
            else:
                self.assertEqual(app.run_domain_webcam(mode,camera_opener,width=160,height=90),0)
        camera_opener.assert_called_once_with(0)
        factory.assert_called_once_with(vanish_sound.GOJO_SOUND_PATH if mode=='gojo' else vanish_sound.SUKUNA_SOUND_PATH)
        sound.close.assert_called_once()
        segmenter.close.assert_called_once()
        recognizer.close.assert_called_once()
        capture.release.assert_called_once()
        self.assertEqual(face_tracker.close.call_count,int(mode=='gojo'))
        if not virtual_error:
            virtual.close.assert_called_once()
        return virtual,sound,outputs,frame

    def test_manual_open_plays_once_and_close_reset_stop_audio(self):
        for mode in ('gojo','sukuna'):
            _,sound,_,_=self.run_video(mode)
            sound.play.assert_called_once()
            self.assertEqual(sound.stop.call_count,2)

    def test_video_keeps_sending_frames_when_inference_has_no_result(self):
        for mode in ('gojo','sukuna'):
            virtual,sound,outputs,_=self.run_video(mode,worker_type=BlockedAnalysis)
            self.assertEqual(virtual.send.call_count,5)
            self.assertEqual(len(outputs),5)

    def test_meme_video_keeps_running_without_waiting_for_landmarkers(self):
        frame=np.full((90,160,3),30,np.uint8)
        frame[:,80:]=200
        capture,virtual,face,hands=Mock(),Mock(),Mock(),Mock()
        capture.read.return_value=(True,frame.copy())
        with ExitStack() as stack:
            stack.enter_context(patch('builtins.print'))
            stack.enter_context(patch.object(run3,'LatestAnalysis',BlockedAnalysis))
            stack.enter_context(patch.object(run3,'FramePacer'))
            stack.enter_context(patch.object(run3,'load_meme',return_value=frame))
            stack.enter_context(patch.object(run3,'open_camera',return_value=(capture,None)))
            stack.enter_context(patch.object(run3.mp_vision.FaceLandmarker,'create_from_options',return_value=face))
            stack.enter_context(patch.object(run3.mp_vision.HandLandmarker,'create_from_options',return_value=hands))
            stack.enter_context(patch.object(run3.pyvirtualcam,'Camera',return_value=virtual))
            for name in ('namedWindow','resizeWindow','destroyAllWindows','imshow','setMouseCallback'):
                stack.enter_context(patch.object(run3.cv2,name))
            stack.enter_context(patch.object(run3.cv2,'waitKey',side_effect=(ord('m'),0,0,0,ord('q'))))
            run3.run_meme_webcam()
        self.assertEqual(virtual.send.call_count,5)
        self.assertEqual(virtual.send.call_args_list[0].args[0][100,5,0],200)
        self.assertEqual(virtual.send.call_args_list[1].args[0][100,5,0],30)
        face.detect_for_video.assert_not_called()
        hands.detect_for_video.assert_not_called()
        face.close.assert_called_once()
        hands.close.assert_called_once()
        capture.release.assert_called_once()
        virtual.close.assert_called_once()

    def test_real_renderers_receive_gesture_and_send_same_animated_frames(self):
        for mode in ('gojo','sukuna'):
            virtual,sound,outputs,frame=self.run_video(mode,pose=True,gesture=True)
            sound.play.assert_called_once()
            self.assertEqual(virtual.send.call_count,len(outputs))
            x1,y1,x2,y2=MirrorControl(160,90).bounds
            outside=np.ones((90,160),bool)
            outside[y1:y2,x1:x2]=False
            for sent,shown in zip(virtual.send.call_args_list,outputs):
                np.testing.assert_array_equal(sent.args[0][outside],shown[outside])
            self.assertFalse(np.array_equal(outputs[-1],frame))

    def test_virtual_camera_open_and_send_failures_release_resources(self):
        self.run_video('sukuna',virtual_error=RuntimeError('OBS busy'))
        self.run_video('gojo',send_error=RuntimeError('send failed'))

    def test_detected_face_is_forwarded_to_gojo_comic_positioning(self):
        from gojo_effect import GojoDomain
        original=GojoDomain.render
        with patch.object(GojoDomain,'render',autospec=True,side_effect=original) as render:
            self.run_video('gojo',pose=True,gesture=True)
        self.assertTrue(any(call.kwargs.get('face_box')==(.4,.25,.6,.55)
                            for call in render.call_args_list))


if __name__=='__main__':
    unittest.main()
