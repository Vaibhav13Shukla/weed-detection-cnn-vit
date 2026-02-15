#!/usr/bin/env python3
"""
Weed Detection - Jetson Nano Real-Time Inference
CNN-ViT Hybrid | MobileNetV3-Small + 4-layer ViT

Problem: Automate weed detection to reduce herbicide usage
         and increase precision agriculture yield.

Setup:
  1. Flash JetPack 4.6.1+ on Jetson Nano
  2. pip3 install numpy pillow opencv-python onnxruntime-gpu
  3. Convert ONNX to TensorRT (one-time, ~5 min):
     /usr/src/tensorrt/bin/trtexec \
         --onnx=weed_detector_cnn_vit.onnx \
         --saveEngine=weed_detector.trt \
         --fp16 --workspace=1024
  4. Run:
     python3 jetson_weed_detector.py --model weed_detector_cnn_vit.onnx --camera 0
"""

import numpy as np
import cv2
import time
import argparse

SPECIES = {
    0: 'Chinee apple', 1: 'Lantana', 2: 'Parthenium',
    3: 'Prickly acacia', 4: 'Rubber vine', 5: 'Siam weed',
    6: 'Snake weed', 7: 'Parkinsonia', 8: 'Negative'
}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 224

SPRAY_COLORS_BGR = {
    'NO_SPRAY':   (255, 180, 50),
    'SPOT_SPRAY': (0, 165, 255),
    'FULL_SPRAY': (0, 0, 255),
}


class WeedDetector:
    def __init__(self, model_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.fps_buf = []

    def preprocess(self, frame_bgr):
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    def predict(self, frame_bgr):
        inp = self.preprocess(frame_bgr)
        t0 = time.perf_counter()
        out = self.session.run(None, {'input': inp})[0][0]
        dt = (time.perf_counter() - t0) * 1000

        multi, binary = out[:9], out[9:]
        mp = np.exp(multi - multi.max()); mp /= mp.sum()
        bp = np.exp(binary - binary.max()); bp /= bp.sum()

        sid = int(np.argmax(mp))
        wc = float(bp[1])
        spray = 'FULL_SPRAY' if wc >= 0.7 else ('SPOT_SPRAY' if wc >= 0.3 else 'NO_SPRAY')

        self.fps_buf.append(1000 / max(dt, 0.1))
        if len(self.fps_buf) > 30:
            self.fps_buf.pop(0)

        return {
            'species': SPECIES[sid], 'conf': float(mp[sid]),
            'is_weed': sid != 8, 'weed_conf': wc,
            'spray': spray, 'ms': dt, 'fps': np.mean(self.fps_buf)
        }

    def overlay(self, frame, r):
        h, w = frame.shape[:2]
        c = SPRAY_COLORS_BGR[r['spray']]
        cv2.rectangle(frame, (4, 4), (w-4, h-4), c, 3)
        cv2.rectangle(frame, (0, 0), (w, 110), (0, 0, 0), -1)
        cv2.putText(frame, f"Species: {r['species']} ({r['conf']:.2f})",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        weed_txt = f"Weed: {'YES' if r['is_weed'] else 'NO'} ({r['weed_conf']:.2f})"
        cv2.putText(frame, weed_txt, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0) if r['is_weed'] else (200, 200, 200), 2)
        cv2.putText(frame, f"Spray: {r['spray']}", (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        cv2.putText(frame, f"FPS: {r['fps']:.1f} | {r['ms']:.0f}ms",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return frame

    def run(self, cam_id=0, W=640, H=480):
        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        if not cap.isOpened():
            print("[ERR] Camera failed")
            return
        print("[INFO] Running... Press q to quit, s to screenshot")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            r = self.predict(frame)
            frame = self.overlay(frame, r)
            cv2.imshow('Weed Detector - CNN-ViT Hybrid', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('s'):
                fn = f"capture_{int(time.time())}.jpg"
                cv2.imwrite(fn, frame)
                print(f"[SAVE] {fn}")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Weed Detection - Jetson Nano')
    ap.add_argument('--model', default='weed_detector_cnn_vit.onnx')
    ap.add_argument('--camera', type=int, default=0)
    args = ap.parse_args()
    WeedDetector(args.model).run(args.camera)
