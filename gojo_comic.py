"""Brief transparent manga balloon composited over Gojo's opening."""
from pathlib import Path

import cv2
import numpy as np

COMIC_PATH = Path(__file__).resolve().parent / 'assets' / 'unlimitedvoidcomic.png'
COMIC_SECONDS = 3.2


def _ease(value):
    value = float(np.clip(value,0,1))
    return value*value*(3-2*value)


class GojoComicOverlay:
    def __init__(self,width,height,path=COMIC_PATH):
        self.width,self.height = width,height
        source = cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if source is None:
            raise RuntimeError(f'Vignetta Gojo non trovata: {path}')
        if source.ndim != 3 or source.shape[2] != 4 or not source[:,:,3].any():
            raise RuntimeError(f'La vignetta Gojo deve essere un PNG con trasparenza visibile: {path}')
        # Remove empty margins without modifying the supplied file. Premultiply
        # before filtering to retain clean, transparent black ink outlines.
        x,y,w,h = cv2.boundingRect((source[:,:,3]>0).astype(np.uint8))
        source = source[y:y+h,x:x+w]
        alpha = source[:,:,3:4].astype(np.float32)/255
        sprite = np.concatenate((source[:,:,:3].astype(np.float32)*alpha,alpha),axis=2)
        ratio = min(width*.23/w,height*.46/h,1)
        self.sprite = cv2.resize(sprite,(max(1,round(w*ratio)),max(1,round(h*ratio))),
                                 interpolation=cv2.INTER_AREA)
        self.margin_x,self.margin_y = max(1,round(width*.03)),max(1,round(height*.04))
        self.reset()

    def reset(self):
        self.position = None

    def render(self,scene,mask,age,face_box=None):
        if age<=0 or age>=COMIC_SECONDS:
            return scene
        opacity = _ease(age/.18)*_ease((COMIC_SECONDS-age)/.5)
        h,w = self.sprite.shape[:2]
        min_x = self.width*.51
        max_x = self.width-self.margin_x-w
        target = np.array((self.width*.68,self.height*.20),np.float32)
        if face_box is not None:
            fx1,fy1,fx2,fy2 = face_box
            gap = max(4,self.width*.018)
            right = fx2*self.width+gap
            left = fx1*self.width-gap-w
            # Normally sit to the right of the face. If the head is already
            # near the screen edge, use its left side, still in the right half.
            target[0] = left if right>max_x and left>=min_x else right
            target[1] = (fy1+fy2)*self.height/2-h*.52
        elif self.position is not None:
            target = self.position.copy()  # Keep the last anchor during tracking gaps.
        target[0] = np.clip(target[0],min_x,max_x)
        target[1] = np.clip(target[1],self.margin_y,self.height-self.margin_y-h)
        if self.position is None:
            self.position = target
        else:
            self.position += (target-self.position)*.25
        entrance = _ease(age/.24)
        scale = .92+.08*entrance
        sprite = cv2.resize(self.sprite,(max(1,round(w*scale)),max(1,round(h*scale))))
        h,w = sprite.shape[:2]
        x = round(float(self.position[0]))
        y = round(float(self.position[1]))-round((1-entrance)*self.height*.025)
        x2,y2 = min(self.width,x+w),min(self.height,y+h)
        part = sprite[:y2-y,:x2-x]
        roi = scene[y:y2,x:x2]
        roi[:] = np.clip(roi*(1-part[:,:,3:4]*opacity)+part[:,:,:3]*opacity,0,255).astype(np.uint8)
        return scene
