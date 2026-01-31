import cv2
import numpy as np

class IntelligentSlicer:
    def __init__(self, padding_factor, info_threshold, min_crop_size):
        self.padding_factor = padding_factor
        self.info_threshold = info_threshold
        self.min_crop_size = min_crop_size

    def slice(self, original_image, heatmap):
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap.astype(np.float32), (w, h))

        mask = (heatmap_resized > self.info_threshold).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crops = []
        for i, cnt in enumerate(contours):
            x, y, bw, bh = cv2.boundingRect(cnt)

            pad_x = int(bw * self.padding_factor)
            pad_y = int(bh * self.padding_factor)

            crop_w = bw + 2 * pad_x
            crop_h = bh + 2 * pad_y

            if crop_w < self.min_crop_size:
                extra_pad = (self.min_crop_size - crop_w) // 2 + 1
                pad_x += extra_pad

            if crop_h < self.min_crop_size:
                extra_pad = (self.min_crop_size - crop_h) // 2 + 1
                pad_y += extra_pad

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)

            final_w = x2 - x1
            final_h = y2 - y1

            if final_w < self.min_crop_size:
                shortfall = self.min_crop_size - final_w
                if x1 > 0:
                    x1 = max(0, x1 - shortfall)
                else:
                    x2 = min(w, x2 + shortfall)

            if final_h < self.min_crop_size:
                shortfall = self.min_crop_size - final_h
                if y1 > 0:
                    y1 = max(0, y1 - shortfall)
                else:
                    y2 = min(h, y2 + shortfall)

            crop_img = original_image[y1:y2, x1:x2]

            if crop_img.size > 0:
                crops.append({
                    'id': i,
                    'image': crop_img,
                    'bbox': (x1, y1, x2-x1, y2-y1),
                    'score': np.mean(heatmap_resized[y:y+bh, x:x+bw])
                })

        print(f"  Generated {len(crops)} crops (min size: {self.min_crop_size}px)")
        return crops, mask, heatmap_resized
