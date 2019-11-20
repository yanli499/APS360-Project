import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("ChatTime")

img_counter = 0

while True:
    try:
        ret, frame = cam.read()
        if not ret:
            break

        img_name = "frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        img_counter += 1

    except(KeyboardInterrupt):
        print("Turning off camera.")
        cam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break
