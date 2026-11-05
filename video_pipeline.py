"""Keep capture latency bounded while the effects process the newest frame."""
from threading import Condition,Event,Thread
from dataclasses import dataclass
import time

import cv2
import mediapipe as mp


@dataclass(frozen=True)
class AnalysisResult:
    sequence: int
    data: object
    elapsed_ms: float


class FramePacer:
    """Limit total loop time, including camera waiting and rendering."""
    def __init__(self,fps):
        self.interval=1/fps

    def wait(self,started_at):
        remaining=self.interval-(time.perf_counter()-started_at)
        if remaining>0:
            time.sleep(remaining)


class VideoStats:
    def __init__(self,label):
        self.label=label
        self.started=time.perf_counter()
        self.frames=0

    def tick(self,analysis=None):
        self.frames+=1
        now=time.perf_counter()
        elapsed=now-self.started
        if elapsed>=3:
            detail=f' | analisi {analysis.elapsed_ms:.0f} ms' if analysis is not None else ''
            print(f'[VIDEO] {self.label}: {self.frames/elapsed:.1f} FPS{detail}')
            self.started,self.frames=now,0


class LatestAnalysis:
    """One inference worker, one replaceable pending frame, non-blocking polling.

    cleanup belongs to this thread: models are never closed while an inference
    is still executing, including when application shutdown times out.
    """
    def __init__(self,process,cleanup=None):
        self._process,self._cleanup=process,cleanup
        self._condition=Condition()
        self._pending=None
        self._result=None
        self._error=None
        self._sequence=0
        self._stopping=False
        self._thread=Thread(target=self._run,name='webcam-inference',daemon=True)
        self._thread.start()

    def submit(self,data):
        with self._condition:
            if self._stopping:
                return
            self._sequence+=1
            self._pending=(self._sequence,data)
            self._condition.notify()

    def latest(self):
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f'Analisi webcam interrotta: {self._error}') from self._error
            return self._result

    def _run(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda:self._stopping or self._pending is not None)
                    if self._stopping:
                        return
                    sequence,data=self._pending
                    self._pending=None
                start=time.perf_counter()
                result=self._process(data)
                elapsed=(time.perf_counter()-start)*1000
                with self._condition:
                    self._result=AnalysisResult(sequence,result,elapsed)
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._error=error
                self._condition.notify_all()
        finally:
            if self._cleanup is not None:
                self._cleanup()

    def close(self):
        with self._condition:
            self._stopping=True
            self._pending=None
            self._condition.notify_all()
        self._thread.join(timeout=2)


def configure_video_processing():
    # Many tiny OpenCV operations run alongside MediaPipe's own worker pools.
    # Avoid launching a full machine-sized pool for each of those operations.
    if cv2.getNumThreads()>2:
        cv2.setNumThreads(2)


def tracking_image(frame,max_width=640):
    height,width=frame.shape[:2]
    if width>max_width:
        frame=cv2.resize(frame,(max_width,max(1,round(height*max_width/width))),
                         interpolation=cv2.INTER_AREA)
    return mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))


class LatestFrameCapture:
    """Single owner of VideoCapture.read; overwrite stale frames, never queue them."""
    def __init__(self,capture,first_frame):
        self.capture=capture
        self._condition=Condition()
        self._stop=Event()
        self._frame=first_frame
        self._sequence=1
        self._consumed=0
        self._failed=False
        self._released=False
        self._thread=Thread(target=self._capture,name='webcam-latest-frame',daemon=True)
        self._thread.start()

    def _capture(self):
        failures=0
        try:
            while not self._stop.is_set():
                ok,frame=self.capture.read()
                if self._stop.is_set():
                    break
                if not ok or frame is None or not frame.size:
                    failures+=1
                    if failures>=30:
                        break
                    self._stop.wait(.03)
                    continue
                failures=0
                with self._condition:
                    self._frame=frame
                    self._sequence+=1
                    self._condition.notify_all()
        except Exception:
            # read() reports failure to the main video loop, which owns UI/errors.
            pass
        finally:
            with self._condition:
                self._failed=True
                self._condition.notify_all()

    def read(self):
        deadline=time.monotonic()+1.2
        with self._condition:
            while self._sequence==self._consumed and not self._failed and not self._stop.is_set():
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    return False,None
                self._condition.wait(remaining)
            if self._stop.is_set() or (self._failed and self._sequence==self._consumed):
                return False,None
            self._consumed=self._sequence
            return True,self._frame

    def release(self):
        if self._released:
            return
        self._released=True
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        # Let an ordinary in-flight camera read finish before releasing it.
        self._thread.join(timeout=.5)
        self.capture.release()
        if self._thread.is_alive():
            self._thread.join(timeout=.5)
