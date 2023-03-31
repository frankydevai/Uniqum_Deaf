from gif_generate import create_gif
import cv2



file = input("Enter word: ")
create_gif(file)


# create a VideoCapture object
cap = cv2.VideoCapture(f'gif/{file}.mp4')

# check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# read and display each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('My Video', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'): # press q to quit
        break

# release the VideoCapture object and close the display window
cap.release()
cv2.destroyAllWindows()

