"""Layered, cel-shaded horned skulls drawn in code, in front of the camera subject."""
from __future__ import annotations

import math
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class SkullSprite:
    color: np.ndarray
    alpha: np.ndarray
    inverse: np.ndarray

    @property
    def shape(self):
        return self.color.shape

    @classmethod
    def from_premultiplied(cls,sprite):
        alpha=np.clip(sprite[:,:,3],0,1).astype(np.float32)
        color=np.zeros_like(sprite[:,:,:3])
        np.divide(sprite[:,:,:3],alpha[:,:,None],out=color,where=alpha[:,:,None]>0)
        return cls(np.clip(np.rint(color),0,255).astype(np.uint8),alpha,1-alpha)


def _ease(value):
    value = min(1.0, max(0.0, value))
    return value * value * (3 - 2 * value)


def _skull_sprite(variant, angle):
    """Ink outlines, flat bone/shadow colors and an independently moving jaw."""
    scale = 2
    head = np.zeros((640,640,4),np.uint8)
    jaw = head.copy()
    ink = (15,12,20,255)
    bone = (179+variant*6,196+variant*5,212+variant*4,255)
    light = (207,220,232,255)
    shadow = (92,105,133,255)

    def pts(values):
        return np.round((np.asarray(values,np.float32)+32)*scale).astype(np.int32)

    def shape(canvas,values,color=bone,outline=True):
        polygon=pts(values)
        cv2.fillPoly(canvas,[polygon],color,cv2.LINE_AA)
        if outline:
            cv2.polylines(canvas,[polygon],True,ink,3*scale,cv2.LINE_AA)

    def line(canvas,values,color=ink,thickness=2):
        cv2.polylines(canvas,[pts(values)],False,color,thickness*scale,cv2.LINE_AA)

    # Swept bovine horns: asymmetrical tips keep the pile from looking stamped.
    horn=[(70,112),(45,105),(26,87),(15,64),(12,38),(20,12+variant*4),
          (25,45),(37,65),(55,78),(78,82)]
    shape(head,horn)
    shape(head,[(256-x,y+variant*2) for x,y in horn])
    shape(head,[(18,48),(29,68),(49,86),(68,88),(64,103),(42,94),(26,77)],shadow,False)
    shape(head,[(238,48),(227,68),(207,86),(188,88),(192,103),(214,94),(230,77)],shadow,False)
    line(head,[(44,78),(40,88),(45,96)])
    line(head,[(212,78),(216,88),(211,96)])
    # Broad brow, cheek bones and a long bovine nasal bridge.
    outline=[(77,86),(99,73),(125,79),(146,72),(175,84),(189,105),
             (191,130),(180,151),(165,166),(156,193),(148,215),(136,229),
             (118,229),(105,212),(98,185),(92,166),(76,153),(65,130),(66,107)]
    shape(head,outline)
    shape(head,[(77,93),(70,119),(80,145),(99,160),(104,188),(114,207),
                (112,176),(104,148),(108,116),(99,101)],shadow,False)
    shape(head,[(173,92),(185,115),(179,142),(159,161),(149,195),(139,214),
                (143,177),(148,146),(143,119),(155,101)],shadow,False)
    shape(head,[(108,86),(122,90),(134,85),(145,99),(135,122),(137,158),
                (128,174),(116,158),(120,123)],light,False)
    for eye in ([(77,116),(92,108),(107,113),(114,124),(105,140),(87,143),(76,132)],
                [(179,116),(164,108),(149,113),(142,124),(151,140),(169,143),(180,132)]):
        shape(head,eye,ink,False)
    line(head,[(73,110),(91,101),(110,108)],thickness=3)
    line(head,[(183,110),(165,101),(146,108)],thickness=3)
    shape(head,[(125,170),(131,170),(141,188),(137,207),(129,211),
                (118,205),(115,189)],ink,False)
    line(head,[(128,177),(128,196)],bone)
    line(head,[(96,145),(104,155),(99,162),(107,172)])
    line(head,[(160,145),(152,155),(157,162),(149,172)])
    line(head,[(105,82),(111,96),(106,103),(113,109)])
    line(head,[(145,78),(141,91),(148,98)])
    line(head,[(83,153),(87,165),(94,171)])
    # Open mouth and two rows of separated teeth.
    shape(head,[(108,211),(147,211),(151,234),(106,234)],ink,False)
    for i in range(6):
        x=108+i*6.2
        shape(head,[(x,213),(x+6,212),(x+5,225),(x+1,226)],light)
    shape(jaw,[(98,199),(106,206),(110,230),(119,237),(138,237),
               (148,229),(152,206),(160,198),(156,234),(145,249),
               (112,249),(101,236)],bone)
    shape(jaw,[(103,222),(109,236),(122,243),(142,241),(146,248),
               (113,247),(104,235)],shadow,False)
    for i in range(5):
        x=113+i*6.2
        shape(jaw,[(x,235),(x,226),(x+5,225),(x+6,235)],bone)

    def finish(canvas):
        small=cv2.resize(canvas,(320,320),interpolation=cv2.INTER_AREA).astype(np.float32)
        alpha=small[:,:,3:4]/255
        premult=np.concatenate((small[:,:,:3]*alpha,alpha),axis=2)
        rotation=cv2.getRotationMatrix2D((160,160),angle,1)
        return cv2.warpAffine(premult,rotation,(320,320),flags=cv2.INTER_LINEAR)
    return finish(head),finish(jaw)


class ComicSkullForeground:
    """Two staggered rows rise from below, bob and move their jaws at 12 fps."""
    def __init__(self,width,height):
        self.width,self.height=width,height
        self.tiles=[]
        rng=np.random.default_rng(2661)
        # Back row first; front row deliberately overlaps it and the live torso.
        for row,xs in enumerate((np.linspace(-0.05,1.07,8),np.linspace(-0.02,1.00,7))):
            for i,x in enumerate(xs):
                edge=min(1,abs(x-.5)*2)
                h=height*((.25+rng.uniform(0,.025)) if row==0 else (.31+.10*edge))
                angle=float(rng.uniform(-17,17))
                head,jaw=_skull_sprite(i%3,angle)
                side=max(16,round(h*1.25))
                head=cv2.resize(head,(side,side),interpolation=cv2.INTER_AREA)
                jaw=cv2.resize(jaw,(side,side),interpolation=cv2.INTER_AREA)
                head=SkullSprite.from_premultiplied(head)
                jaw=SkullSprite.from_premultiplied(jaw)
                base=height*((.965+rng.uniform(-.02,.015)) if row==0 else 1.065)
                delay=1.65+(1-edge)*.65+(i%3)*.10+row*.30
                self.tiles.append((head,jaw,x*width,base,h,delay,float(rng.uniform(0,math.tau)),angle))

    @staticmethod
    def _over(scene,sprite,left,top,opacity):
        h,w=sprite.shape[:2]
        x1,y1=max(0,left),max(0,top)
        x2,y2=min(scene.shape[1],left+w),min(scene.shape[0],top+h)
        if x1>=x2 or y1>=y2:
            return
        region=np.s_[y1-top:y2-top,x1-left:x2-left]
        alpha=sprite.alpha[region]
        inverse=sprite.inverse[region]
        if opacity<.999999:
            alpha=alpha*np.float32(opacity)
            inverse=np.float32(1)-alpha
        roi=scene[y1:y2,x1:x2]
        roi[:]=cv2.blendLinear(roi,sprite.color[region],inverse,alpha)

    def render(self,scene,age,visibility=1.0):
        if visibility<=0 or age<=1.65:
            return scene
        tick=math.floor(age*12)/12
        for head,jaw,cx,base,h,delay,phase,angle in self.tiles:
            progress=_ease((age-delay)/1.15)
            opacity=progress*visibility
            if opacity<=0.001:
                continue
            rise=(1-progress*visibility)*h*.85
            bob=math.sin(tick*1.65+phase)*h*.025*progress
            sway=math.sin(tick*.65+phase)*h*.018
            left=round(cx-head.shape[1]/2+sway)
            top=round(base-head.shape[0]*.88+rise+bob)
            self._over(scene,head,left,top,opacity)
            opening=(.5+.5*math.sin(tick*2.2+phase))*h*.04*progress
            self._over(scene,jaw,left+round(math.sin(math.radians(angle))*opening),
                       top+round(math.cos(math.radians(angle))*opening),opacity)
        return scene
