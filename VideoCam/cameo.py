import cv2
import filters
from managers import WindowManager, CaptureManager
import depth
"""
Our application is represented by the Cameo class with two methods :
    - run()
    - onKeypress()
ON initializing, a Cameo object created a WindowManager object with onKeypress as a callback, as well as a CaptureManager object using a camera 
(specifically, a cv2.VideoCapture object) and the same WindowManager object. When run is called the application execute a min loop in which frames 
 and events are processed.

"""
class Cameo(object):

    def __init__(self):
        self._windowManager = WindowManager('Cameo',self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)
        self._curveFilter = filters.testFilter()

    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            
            if frame is not None:
                filters.strokeEdges(frame,frame)
                self._curveFilter.apply(frame,frame)
            
            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """Handle a keypress.
        
        space -> Take a screenshot.
        tab -> Start/stop recording a screencast.
        escape -> Quit.
        """
        if keycode == 32: # space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9: # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27: # escape
            self._windowManager.destroyWindow()


class CameoDepth(Cameo):
    def __init__(self):
        self._windowManager = WindowManager('Cameo', self.onKeypress)

    #Because i couldn't use the same api for the webcam as the source code i had to give the index of the camera instead of 
    # creating a variable with the cv2.CAP_OPENNI2_ASUS    
        #device = cv2.CAP_OPENNI2# uncomment for Kinect
        #device = cv2.CAP_OPENNI2_ASUS # uncomment for Xtion or Structure
        #self._captureManager = CaptureManager(cv2.VideoCapture(device), self._windowManager, True)

        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)
        self._curveFilter = filters.FindEdgesFilter()

    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            self._captureManager.channel = cv2.CAP_OPENNI_DISPARITY_MAP
            disparityMap = self._captureManager.frame
            self._captureManager.channel = cv2.CAP_OPENNI_VALID_DEPTH_MASK
            validDepthMask = self._captureManager.frame
            self._captureManager.channel = cv2.CAP_OPENNI_BGR_IMAGE
            frame = self._captureManager.frame
            if frame is None:
                # Failed to capture a BGR frame.
                # Try to capture an infrared frame instead.
                self._captureManager.channel = cv2.CAP_OPENNI_IR_IMAGE
                frame = self._captureManager.frame

            if frame is not None:
                # Make everything except the median layer black.
                mask = depth.createMedianMask(disparityMap, validDepthMask)
                frame[mask == 0] = 0

                if self._captureManager.channel == \
                        cv2.CAP_OPENNI_BGR_IMAGE:
                    # A BGR frame was captured.
                    # Apply filters to it.
                    filters.strokeEdges(frame, frame)
                    self._curveFilter.apply(frame, frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

if __name__ == "__main__":
    #Cameo().run()
    CameoDepth().run()