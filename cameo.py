import cv2
import filters
from managers import WindowManager, CaptureManager

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
        self._curveFilter = filters.FindEdgesFilter()

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

if __name__ == "__main__":
    Cameo().run()