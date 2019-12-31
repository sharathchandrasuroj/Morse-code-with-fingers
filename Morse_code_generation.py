"@Author: @UdayKiran"


#Importing libraries
import cv2
import numpy as np
import copy
import math
from itertools import groupby
import time
import tensorflow as tf


class CaptureHand():

    def __init__(self):
        pass

    def setParameters(self, parameters):
        # parameters for detecting hand in video
        self.hand_box_x = parameters.get("hand_box_x",0.6)  # start point/total width
        self.hand_box_y = parameters.get("hand_box_y",0.7) # start point/total width
        self.threshold = parameters.get("threshold",40)  #  BINARY threshold
        self.blur_alue = parameters.get("blur_value",41)  # GaussianBlur parameter
        self.background_threshold = parameters.get("background_threshold",50)
        self.learning_rate = parameters.get("learning_rate",0)

        # variables
        self.is_bg_captured = 0   # bool, whether the background captured
        self.trigger_switch = False  # if true, keyborad simulator works

        self.morse_codes = {"12":"A","2111":"B","2121":"C","211":"D","1":"E","1121":"F","221":"G","1111":"H","11":"I", "1222":"J","212":"K", "1211":"L","22":"M","21":"N","222":"O","1221":"P","2212":"Q","121":"R","111":"S","2":"T","112":"U","1112":"V", "122":"W","2112":"X","2122":"Y","2211":"Z","1222":"1","11222":"2","11122":"3","11112":"4","11111":"5","21111":"6","22111":"7","22211":"8","22221":"9","22222":"0"} #Morse codes for Alphabets and numbers

    def printThreshold(self,thr):
        print("! Changed threshold to "+str(thr))

    def remove_back_ground(self, frame, bgModel):
        fgmask = bgModel.apply(frame,learningRate=self.learning_rate)
        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        res = cv2.bitwise_and(frame, frame, mask=fgmask)
        return res

    def load_model(self, path):
        model = tf.keras.models.load_model(path)
        print(model.summary())
        classes = ["0","1","2","3","4","5"] #class labels
        return model, classes

    def capture_video(self, path):
        camera = cv2.VideoCapture(0) #start capturing video
        camera.set(10,200)
        cv2.namedWindow('trackbar')
        cv2.createTrackbar('trh1', 'trackbar', self.threshold, 100, self.printThreshold)
        model, classes = self.load_model(path) #loading pretrained model to predict no of fingers opened
        store_predicted_classes = [] #to store predicted classes
        final_text = '' #adding each character to the final text
        start_new_char = False
        text_image = np.zeros((512, 512, 1), dtype = "uint8")

        while camera.isOpened():
            ret, frame = camera.read()
            threshold = cv2.getTrackbarPos('trh1', 'trackbar')
            frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
            frame = cv2.flip(frame, 1)  # flip the frame horizontally
            cv2.rectangle(frame, (int(self.hand_box_x * frame.shape[1]), 0),
                         (frame.shape[1], int(self.hand_box_y * frame.shape[0])), (255, 0, 0), 2)
            cv2.imshow('original_video', frame)

            #  Main operation
            if self.is_bg_captured == 1:  # this part wont run until background captured
                img = self.remove_back_ground(frame, bgModel)
                img = img[0:int(self.hand_box_y * frame.shape[0]),
                            int(self.hand_box_x * frame.shape[1]):frame.shape[1]]  # clip the ROI
                cv2.imshow('mask', img)
                cv2.waitKey(250)#To reduce the frame numbers to capure the hand perfectly.
                # convert the image into binary image
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (self.blur_alue, self.blur_alue), 0)
                cv2.imshow('blur', blur)
                ret, thresh = cv2.threshold(blur, self.threshold, 255, cv2.THRESH_BINARY)
                cv2.imshow('ori', thresh)
                img2= cv2.resize(thresh,dsize=(150,150), interpolation = cv2.INTER_CUBIC) #resizing image to fin into tensorflow model 150X150
                #Numpy array
                np_image_data = np.asarray(img2)
                #maybe insert float convertion here - see edit remark!
                np_final = np.expand_dims(np_image_data,axis=0)
                input_data = np_final.reshape((np_final.shape[0], np_final.shape[1], np_final.shape[2], 1))
                input_data = tf.cast(input_data, tf.float32)

                class_pred = classes[np.argmax(model.predict(input_data)[0])] #predict the class of the image
                cv2.putText(thresh, str(class_pred), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245,210,65), 2, 1)#print the predicted class on image
                cv2.imshow('predict', thresh)

                #This logic is to predict the morse code
                if class_pred == "4": #To add space to your morse code
                    final_text+=" "
                if class_pred == "3": #To start cpturing the fingures for next morse character
                    start_new_char = True
                if start_new_char and class_pred=="5": #To stop cpturing the fingures for current morse character and predict the morse code for captured fingers
                    char = ""
                    store_predicted_classes = [x[0] for x in groupby(store_predicted_classes)] #Remove consecutive duplicates
                    for i in store_predicted_classes:
                        if i!="0" and i!="3" and i!="4":
                            char+=i
                    predicted_char = self.morse_codes[char] #get morse code
                    final_text+=predicted_char
                    store_predicted_classes=[]
                    start_new_char = False
                if start_new_char and (class_pred=="1" or class_pred=="2" or class_pred=="0"):
                    store_predicted_classes.append(class_pred)
                cv2.putText(text_image, final_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245,210,65), 2, 1)
                cv2.imshow('test', text_image)

            # This code is to capture background and reset backgroud
            k = cv2.waitKey(10)
            if k == 27:  # press ESC to exit
                camera.release()
                cv2.destroyAllWindows()
                break
            elif k == ord('b'):  # press 'b' to capture the background
                bgModel = cv2.createBackgroundSubtractorMOG2(0, self.background_threshold)
                self.is_bg_captured = 1
                print( '!!!Background Captured!!!')
            elif k == ord('r'):  # press 'r' to reset the background
                bgModel = None
                self.trigger_switch = False
                self.is_bg_captured = 0
                print ('!!!Reset BackGround!!!')
            elif k == ord('n'):
                self.trigger_switch = True
                print ('!!!Trigger On!!!')



if __name__=="__main__":
    parameters = {"hand_box_x":0.6,"hand_box_y":0.7,"threshold":40,"blur_value":41,"background_threshold":50,"learning_rate":0} #predefinied parameters(You can try different values)
    model_path = "my_model.h5" #pretrained model path
    obj = CaptureHand()
    obj.setParameters(parameters)
    obj.capture_video(model_path)
