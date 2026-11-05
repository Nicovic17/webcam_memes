"""Local preview controls; never paint buttons into the virtual-camera feed."""
import cv2
import numpy as np


class MirrorControl:
    def __init__(self,width,height):
        self.mirrored=True
        self.generation=0
        scale=min(1.,width/720,height/405)
        w=min(width-8,max(90,round(210*scale)))
        h=min(height-4,max(18,round(38*scale)))
        margin=max(2,round(12*scale))
        x=(width-w)//2
        y=height-margin-h
        self.bounds=(x,y,x+w,y+h)
        self.font=max(.24,.55*scale)

    def toggle(self):
        self.mirrored=not self.mirrored
        self.generation+=1

    def on_mouse(self,event,x,y,flags,userdata):
        x1,y1,x2,y2=self.bounds
        if event==cv2.EVENT_LBUTTONUP and x1<=x<x2 and y1<=y<y2:
            self.toggle()

    def draw(self,frame):
        preview=frame.copy()
        x1,y1,x2,y2=self.bounds
        roi=preview[y1:y2,x1:x2]
        tint=np.full_like(roi,(34,43,48))
        cv2.addWeighted(roi,.18,tint,.82,0,dst=roi)
        color=(125,220,175) if self.mirrored else (190,195,205)
        cv2.rectangle(roi,(0,0),(x2-x1-1,y2-y1-1),color,1,cv2.LINE_AA)
        text=f'Flip: {"ON" if self.mirrored else "OFF"}  [M]'
        (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,self.font,1)
        cv2.putText(roi,text,((x2-x1-tw)//2,(y2-y1+th)//2),
                    cv2.FONT_HERSHEY_SIMPLEX,self.font,(235,242,246),1,cv2.LINE_AA)
        return preview
