import cv2

key = cv2. waitKey(1)
cam = cv2.VideoCapture(0)

cv2.namedWindow("ChatTime")

img_counter = 0

while True:
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
        img_name = "frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
