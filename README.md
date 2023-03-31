The Project has two main parts, the first part is responsible for generating a GIF from a user-provided input word. The second part is responsible for reading and displaying the frames of a video file that corresponds to the generated GIF.
You can run Final.py for giving input word. 
If you want to use images folder just unzip it. There are all extracted images and labeled. So you don't need to run data_extract.ipynb
At the beginning of the code, there is an import statement for a custom module gif_generate that has a create_gif function, which generates a GIF from a provided input word. The user is prompted to enter a word, which is then passed to the create_gif function.

After generating the GIF, the code creates a VideoCapture object to read the video file that corresponds to the generated GIF. The code checks if the video file is opened successfully and if it is not, it prints an error message and exits the program.

The code then reads each frame of the video using the VideoCapture.read() method and displays it using the cv2.imshow() function. The code also waits for a keyboard event using the cv2.waitKey() function, and if the event corresponds to the 'q' key, the program breaks out of the loop.

Finally, the VideoCapture object is released using the VideoCapture.release() method, and the display window is closed using the cv2.destroyAllWindows() function.
