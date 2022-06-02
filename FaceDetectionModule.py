import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, minDetectionCon=0.5):

        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                print(id, detection)
                print(detection.score)
                print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img,bbox)

                    cv2.putText(img, f'Score: {int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1)
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, thickness=2):
            x, y, w, h = bbox
            x1, y1 = x+w, y+h

            # Top, Left x, y
            cv2.line(img, (x, y), (x+l, y), (0, 255, 255), thickness)
            cv2.line(img, (x, y), (x, y+l), (0, 255, 255), thickness)

            # Top Right x1, y
            cv2.line(img, (x1, y), (x1-l, y), (0, 255, 255), thickness)
            cv2.line(img, (x1, y), (x1, l+y), (0, 255, 255), thickness)

            # Bottom Left x, y1
            cv2.line(img, (x, y1), (x+l, y1), (0, 255, 255), thickness)
            cv2.line(img, (x, y1), (x, y1-l), (0, 255, 255), thickness)

            # Bottom Right x1, y1
            cv2.line(img, (x1, y1), (x1-l, y1), (0, 255, 255), thickness)
            cv2.line(img, (x1, y1), (x1, y1-l), (0, 255, 255), thickness)

            return img


def main():
    ### Your path file
    cap = cv2.VideoCapture("videos/output.avi")
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        print(bboxs)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()

