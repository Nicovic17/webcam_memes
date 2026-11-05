"""Effect selection and standalone domain runtime for webcam_memes.

All models, effects and media are local to this project; disappear/local_app
is neither imported nor accessed at runtime.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from video_pipeline import LatestAnalysis,FramePacer,VideoStats,tracking_image
from preview_controls import MirrorControl

BASE_DIR = Path(__file__).resolve().parent
HAND_MODEL = BASE_DIR / "hand_landmarker.task"
FACE_MODEL = BASE_DIR / "face_landmarker.task"
SEGMENTER_MODEL = BASE_DIR / "models" / "selfie_segmenter_landscape.tflite"
EFFECTS = (
    ("meme", "Meme Webcam", "Pose delle mani e meme sul volto"),
    ("gojo", "Gojo Webcam", "Unlimited Void / dominio cosmico"),
    ("sukuna", "Sukuna Webcam", "Malevolent Shrine / dominio e teschi"),
)


def choose_effect():
    """Select before opening any camera, detector or audio device."""
    title = "Webcam Memes - Scegli effetto"
    selected = [None]
    image = np.full((470,720,3),(26,29,34),np.uint8)
    cv2.putText(image,"SCEGLI L'EFFETTO WEBCAM",(55,62),
                cv2.FONT_HERSHEY_SIMPLEX,.87,(240,245,247),2,cv2.LINE_AA)
    for index,(_,label,detail) in enumerate(EFFECTS):
        top = 100 + index*108
        cv2.rectangle(image,(50,top),(670,top+88),(48,54,62),-1)
        cv2.rectangle(image,(50,top),(670,top+88),(111,142,153),2)
        cv2.putText(image,f"{index+1}  {label}",(75,top+35),
                    cv2.FONT_HERSHEY_SIMPLEX,.80,(240,245,247),2,cv2.LINE_AA)
        cv2.putText(image,detail,(75,top+65),
                    cv2.FONT_HERSHEY_SIMPLEX,.52,(179,195,201),1,cv2.LINE_AA)
    cv2.putText(image,"Clicca una voce o premi 1, 2, 3.  Q / ESC: esci.",(50,450),
                cv2.FONT_HERSHEY_SIMPLEX,.50,(179,195,201),1,cv2.LINE_AA)

    def on_mouse(event,x,y,flags,userdata):
        if event == cv2.EVENT_LBUTTONUP and 50 <= x <= 670:
            for index,(mode,_,_) in enumerate(EFFECTS):
                if 100+index*108 <= y <= 188+index*108:
                    selected[0] = mode

    cv2.namedWindow(title,cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title,on_mouse)
    try:
        while selected[0] is None:
            cv2.imshow(title,image)
            key = cv2.waitKey(30)&0xFF
            if ord('1') <= key <= ord('3'):
                selected[0] = EFFECTS[key-ord('1')][0]
            elif key in (ord('q'),27):
                return None
            if cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:
                return None
        return selected[0]
    finally:
        cv2.destroyWindow(title)


def create_domain_tasks(mode,resources):
    """Create local Tasks models, closing earlier resources on any failure."""
    if mode not in ('gojo','sukuna'):
        raise ValueError(f"Dominio non riconosciuto: {mode}")
    for path in (HAND_MODEL,SEGMENTER_MODEL):
        if not path.is_file():
            raise FileNotFoundError(f"Modello mancante in webcam_memes: {path}")
    from gojo_effect import GojoPoseRecognizer
    from sukuna_effect import SukunaPoseRecognizer

    segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,output_category_mask=False,
    ))
    resources.callback(segmenter.close)
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO,num_hands=2,
        min_hand_detection_confidence=.35,min_hand_presence_confidence=.35,
        min_tracking_confidence=.45,
    )
    # Transfer ownership to the recognizer only after construction succeeds.
    with ExitStack() as pending:
        hands = vision.HandLandmarker.create_from_options(options)
        pending.callback(hands.close)
        if mode == 'sukuna':
            fallback = vision.HandLandmarker.create_from_options(
                replace(options,running_mode=vision.RunningMode.IMAGE))
            pending.callback(fallback.close)
            recognizer = SukunaPoseRecognizer(hands,fallback)
        else:
            recognizer = GojoPoseRecognizer(hands)
        resources.callback(recognizer.close)
        pending.pop_all()
    return segmenter,recognizer


def create_gojo_face_tracker():
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,num_faces=1,
        min_face_detection_confidence=.45,min_face_presence_confidence=.45,
        min_tracking_confidence=.45,
    ))


def person_mask(segmenter,image,timestamp_ms,width,height):
    result = segmenter.segment_for_video(image,timestamp_ms)
    if not result.confidence_masks:
        return np.zeros((height,width),np.float32)
    raw = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
    resized = cv2.resize(raw,(width,height),interpolation=cv2.INTER_LINEAR)
    resized = cv2.GaussianBlur(resized,(0,0),sigmaX=2.2,sigmaY=2.2)
    return np.clip((resized-.12)/.68,0,1)


def create_domain_analyzer(mode,width,height):
    """Models execute serially in their own worker, never on the display loop."""
    resources=ExitStack()
    try:
        segmenter,recognizer=create_domain_tasks(mode,resources)
        face_tracker=create_gojo_face_tracker() if mode=='gojo' else None
        if face_tracker is not None:
            resources.callback(face_tracker.close)
        state=dict(pose=False,pose_at=None,anchor=(.5,.45),hint='',mask=None,face_box=None,generation=0)
        last_gesture=last_face=float('-inf')

        def analyze(packet):
            nonlocal last_gesture,last_face
            frame,stamp,now,running,comic_active,generation=packet
            if generation!=state['generation']:
                state.update(pose=False,pose_at=None,mask=None,face_box=None,generation=generation)
                last_gesture=last_face=float('-inf')
            image=tracking_image(frame)
            if now-last_gesture>=(.25 if running else .08):
                state['pose']=recognizer.recognize_pose(image,stamp)
                state['pose_at']=now
                state['anchor']=recognizer.anchor
                state['hint']=getattr(recognizer,'last_hint','')
                last_gesture=now
            if running:
                state['mask']=person_mask(segmenter,image,stamp,width,height)
            if face_tracker is not None and comic_active and now-last_face>=.12:
                faces=face_tracker.detect_for_video(image,stamp).face_landmarks
                if faces:
                    xs=[point.x for point in faces[0]]
                    ys=[point.y for point in faces[0]]
                    state['face_box']=(min(xs),min(ys),max(xs),max(ys))
                last_face=now
            return state.copy()

        return LatestAnalysis(analyze,cleanup=resources.close)
    except Exception:
        resources.close()
        raise


def run_domain_webcam(mode,camera_opener,*,width=960,height=540,fps=30):
    if mode not in ('gojo','sukuna'):
        raise ValueError(f"Dominio non riconosciuto: {mode}")
    from gojo_effect import GojoDomain,GojoGestureGate
    from gojo_comic import COMIC_SECONDS
    from sukuna_effect import SukunaDomain,SukunaGestureGate,SukunaGestureCoach
    from vanish_sound import GOJO_SOUND_PATH,SUKUNA_SOUND_PATH,VanishSound

    title = dict((key,label) for key,label,_ in EFFECTS)[mode]
    with ExitStack() as resources:
        analysis=create_domain_analyzer(mode,width,height)
        resources.callback(analysis.close)
        sound = VanishSound(GOJO_SOUND_PATH if mode=='gojo' else SUKUNA_SOUND_PATH)
        resources.callback(sound.close)
        effect = GojoDomain(width,height) if mode=='gojo' else SukunaDomain(width,height)
        gate = GojoGestureGate() if mode=='gojo' else SukunaGestureGate()
        coach = SukunaGestureCoach() if mode=='sukuna' else None
        cap,pending_frame = camera_opener(0)
        resources.callback(cap.release)
        try:
            virtual = pyvirtualcam.Camera(width=width,height=height,fps=fps,
                                          fmt=pyvirtualcam.PixelFormat.BGR,backend='obs')
        except Exception as error:
            raise RuntimeError("Non riesco ad avviare OBS Virtual Camera. Disattiva la sua "
                               "Virtual Camera in OBS e chiudi altre istanze del programma.") from error
        resources.callback(virtual.close)
        resources.callback(cv2.destroyAllWindows)
        cv2.namedWindow(title,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title,width,height)
        mirror=MirrorControl(width,height)
        cv2.setMouseCallback(title,mirror.on_mouse)
        mirror_generation=mirror.generation
        if mode=='gojo':
            print('[Gojo] Alza indice e medio a V con una mano; anulare e mignolo ripiegati. '
                  'Non serve incrociare le dita. Mantieni per circa 0.4 secondi.')
        else:
            print('[Sukuna] Avvicina le due mani con almeno due dita alzate per mano, '
                  'oppure mettile a preghiera. Non devono toccarsi. Mantieni per circa mezzo secondo.')
        print(f'[{title}] V: anteprima, R: richiudi, SPAZIO: reset, F: fullscreen, Q/ESC: esci.')
        print(f'[WEBCAM VIRTUALE] Attiva: {virtual.device}')
        started = time.monotonic()
        face_box = None
        anchor=(.5,.45)
        last_pose_at=None
        last_timestamp = -1
        mask = None
        failures = 0
        fullscreen = False
        pacer=FramePacer(fps)
        stats=VideoStats(title)
        while True:
            frame_started=time.perf_counter()
            if pending_frame is not None:
                frame,pending_frame = pending_frame,None
                ok = True
            else:
                ok,frame = cap.read()
            if not ok or frame is None or not frame.size:
                failures += 1
                if failures >= 30:
                    raise RuntimeError('La webcam ha smesso di inviare fotogrammi per circa un secondo.')
                time.sleep(.03)
                continue
            failures = 0
            if mirror.generation!=mirror_generation:
                mirror_generation=mirror.generation
                if mask is not None:
                    mask=cv2.flip(mask,1)
                if face_box is not None:
                    x1,y1,x2,y2=face_box
                    face_box=(1-x2,y1,1-x1,y2)
                anchor=(1-anchor[0],anchor[1])
                gate.reset()
                last_pose_at=None
            if mirror.mirrored:
                frame=cv2.flip(frame,1)
            frame = cv2.resize(frame,(width,height))
            now = time.monotonic()
            timestamp_ms = max(last_timestamp+1,int((now-started)*1000))
            last_timestamp = timestamp_ms
            comic_active=mode=='gojo' and effect.running and now-effect.started<COMIC_SECONDS
            analysis.submit((frame.copy(),timestamp_ms,now,effect.running,comic_active,mirror_generation))
            tracked=analysis.latest()
            state=None
            if tracked is not None and tracked.data['generation']==mirror_generation:
                state=tracked.data
                if state['mask'] is not None:
                    mask=state['mask']
                if state['face_box'] is not None:
                    face_box=state['face_box']
                anchor=state['anchor']
            if state is not None and state['pose_at'] is not None and state['pose_at']!=last_pose_at:
                last_pose_at=state['pose_at']
                pose=state['pose']
                triggered = gate.update(pose,last_pose_at)
                if coach is not None and not triggered and not effect.running:
                    hint = coach.update(state['hint'],pose,gate,now)
                    if hint:
                        print(f'[Sukuna / Mani] {hint}')
                if triggered and effect.enter(now,anchor):
                    sound.play()
                    print(f'[{title}] Domain Expansion!')
            display = (effect.render(frame,mask,now,face_box=face_box) if mode=='gojo'
                       else effect.render(frame,mask,now))
            virtual.send(display)
            cv2.imshow(title,mirror.draw(display))
            key = cv2.waitKey(1)&0xFF
            if key in (ord('q'),27) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:
                break
            if key == ord('m'):
                mirror.toggle()
            elif key == ord('v'):
                if effect.enter(time.monotonic(),anchor):
                    sound.play()
            elif key == ord('r'):
                effect.collapse(time.monotonic())
                sound.stop()
            elif key == ord(' '):
                effect.reset()
                gate.reset()
                sound.stop()
                mask = None
                face_box = None
            elif key == ord('f'):
                fullscreen = not fullscreen
                cv2.setWindowProperty(title,cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            stats.tick(tracked)
            pacer.wait(frame_started)
    return 0
