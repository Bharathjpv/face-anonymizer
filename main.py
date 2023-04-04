import cv2
import mediapipe as mp
import argparse
import os

def process_img(img, face_detection):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    # print(out.detections)
    if out.detections is not None:
        for detections in out.detections:
            location_data = detections.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1*W)
            y1 = int(y1*H)
            w = int(w*W)
            h = int(h*H)

            # cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0,255,0), 10)

            ##
            img[y1:y1+h, x1:x1+w, :] = cv2.blur(img[y1:y1+h, x1:x1+w, :], (50,50))
    return img


args = argparse.ArgumentParser()

args.add_argument('--mode', default='webcam')
args.add_argument('--filepath', default="data/testvideo.mp4")

args = args.parse_args()

## detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    if args.mode in ["image"]:
        ### read image
        img = cv2.imread(args.filepath)

        H,W, _ = img.shape
    
        img = process_img(img, face_detection)

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.mode in ['video']:
        cap = cv2.VideoCapture(args.filepath)
        ret, frame = cap.read()
        output_dir = "./output"
        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                       cv2.VideoWriter_fourcc(*'MP4V'),
                                       25,
                                       (frame.shape[1], frame.shape[0]))
        while ret:
            
            H,W, _ = frame.shape

            frame = process_img(frame, face_detection)

            output_video.write(frame)

            ret, frame = cap.read()
            # cv2.imshow('img', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        cap.release()
        output_video.release()
    
    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()
        H,W, _ = frame.shape
        while ret:
            frame = process_img(frame, face_detection)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()

        cap.release()

# cv2.imwrite('data/testresult.jpg', img)