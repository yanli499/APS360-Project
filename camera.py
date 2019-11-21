import cv2

<<<<<<< HEAD
key = cv2. waitKey(1)
=======
>>>>>>> ef6b0869a152e30e6348f050d2d38257cb6643fc
cam = cv2.VideoCapture(0)

cv2.namedWindow("ChatTime")

img_counter = 0

while True:
<<<<<<< HEAD
    ret, frame = cam.read()
    cv2.imshow("ChatTime", frame)
    if not ret:
        break

    key = cv2. waitKey(1)
    if key%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif key%256 == 32:
        # SPACE pressed
=======
    try:
        ret, frame = cam.read()
        if not ret:
            break

>>>>>>> ef6b0869a152e30e6348f050d2d38257cb6643fc
        img_name = "frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        img_counter += 1

<<<<<<< HEAD
cam.release()
cv2.destroyAllWindows()
=======
    except(KeyboardInterrupt):
        print("Turning off camera.")
        cam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break
>>>>>>> ef6b0869a152e30e6348f050d2d38257cb6643fc
