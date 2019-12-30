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
# Morse Code generation 
- After background, capture try to generate morse code
- Open one finger for the small beep
- Open two fingers for the long beep
- Open three fingers to start a new character 
- Open five fingers to stop the current character
- Open four fingers to add space

## Future work
- Improving the current logic
- Generating Morse code with eyelids
----------------------
## References

1. In [Fingers-Detection-using-OpenCV-and-Python](https://github.com/lzane/Fingers-Detection-using-OpenCV-and-Python)


