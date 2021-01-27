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
            self._captureManager.writeImage('./res/filterRes/screenshot.png')
        elif keycode == 9: # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('./res/filterRes/screencast.avi')
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

    def onKeypress(self, keycode):
        """Handle a keypress.
        
        space -> Take a screenshot.
        tab -> Start/stop recording a screencast.
        escape -> Quit.
        """
        if keycode == 32: # space
            self._captureManager.writeImage('./res/depthRes/screenshot.png')
        elif keycode == 9: # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('./res/depthRes/screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27: # escape
            self._windowManager.destroyWindow()


# We use the same Cameo class that is being used for filter or depth imaging as our Parent class. 

class CameoFaceDet(Cameo):
    def __init__(self):
        self._windowManager = WindowManager("Cameo", self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)

        # We initialize two CascadeClassifier objects, one for faces and another for eyes 

        self.face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

        # As in most of our interactive scripts, we open a camera feed and start iterating over frames. We continue until the usser presses the keys 

    def run(self):
        
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            # Whenever we successfully capture a frame, we convert it into grayscale as our first step in processing it.
            if frame is not None :
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # We detect faces with the detectMutliScale function of our face detector. We use scaleFactorn, minNeighbors, and minSize arguments (minSize will
                # specify the minimum size of a face, specifically 120,120. No attemps will be made to detect faces smaller than this). Assuming that the user will
                # be sitting close to the camera, it is ssafe to say that the user's face will be larger than 120x120 pixels.
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize = (120,120))

                # We iterate over the rectangles of the detected faces. We draw a blue border around each rectangle in the original color image. Then, whithin the 
                # same rectangular region of the grayscale image, we perform eye detection. 

                for (x, y, w , h) in faces :
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y:y+h, x:x+w]

                    # The eye detector is a bit less accurate than the face detector. Shadows, parts of frame glasses, or other regions of the face can be falsly
                    # detected as an eye. To improve the results i improved the code by adding a maxSize of the region of interrest which is of course smaller than
                    # the face to avoid false positives taht are too large to be eyes.
                    
                    eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.03, 5, minSize = (40, 40), maxSize = (90, 90))
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """Handle a keypress.
        
        space -> Take a screenshot.
        tab -> Start/stop recording a screencast.
        escape -> Quit.
        """
        if keycode == 32: # space
            self._captureManager.writeImage('./res/faceTrackRes/screenshot.png')
        elif keycode == 9: # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('./res/faceTrackRes/screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27: # escape
            self._windowManager.destroyWindow()

if __name__ == "__main__":
    Cameo().run()
    #CameoDepth().run()
    #CameoFaceDet().run()