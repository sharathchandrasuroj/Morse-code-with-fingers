## Demo
you can find it [here](https://www.youtube.com/watch?v=n-jaHrTmmo0)

## Environment
- Platform: Python 3
- Librarys: 
	- OpenCV 4
	- TensorFlow

## How to run it?
# Finger detection 
- Keep data in the specified format 
- Run the finger_detection.py file to generate the model
- keep the saved model in the same folder
# Capturing the background and subtraction 
- Run Morse_code_generation.py
- press `'b'` to capture the background model (without hand)
- press `'r'` to reset the background model
- press `'ESC'` to exit

## What it does
It tries to capture the **Morse code** pattern and detect the character based on the observed pattern. When we run the Morse_code_generation.py file, Your camera will be opened and you will see a blue bounding box. when you press **b** it will capture the background without your hand first in the blue bounding box,  and it opens 5 windows (Mask, Blur, Ori, Predicted, Test) 
- **Mask window** will show you the masked image of your hand by removing the background.
- **Blur Window** will show you the blur grey image of your hand by removing the background.
- **Ori Window** will show you the Threshold image of your hand.
- **Predicted Window** will show the predicted number of fingers opened.
- **Test Window** will show the predicted text from morse code. 
 
- If you open **one finger** it is small beep(Dot in Morse code).
- If you open **Two fingers** it is Long beep(Dash in Morse code).
- If you open **Three fingers** it is to start capturing for a new character or reset the current character.
- If you open **Five fingers** it is to Stop capturing and predict the character from morse code pattern.
- If you open **Four fingers** it is to add space between characters.

## How I built it
It is a two-step process. First, I built a [model to detect no of fingers opened] (https://github.com/udaykiranreddykondreddy/Morse-code-with-fingers/blob/master/finger_detection.py) by using TensorFlow 2.0 and a dataset called [Fingers](https://www.kaggle.com/koryakinp/fingers) from kaggle.

The second step is to generate the morse code. Here the first step is to detect the hand in the video and the next step is to take that and predict how many fingers are opened using the pre-trained model which is generated from the first step. 

I used the background subtraction method and HSV segmentation from OpenCV to detect the hand. Then each frame is sent through two filters one is a grey image and another is threshold image. We will be using this threshold image and pass it to the pre-trained model to detect no of fingers opened. 

Based on the no of fingers opened Morse code will be generated.

## Future work
- Improving the current logic
- Generating Morse code with eyelids
----------------------
## References

1. lzane [Fingers-Detection-using-OpenCV-and-Python](https://github.com/lzane/Fingers-Detection-using-OpenCV-and-Python)


