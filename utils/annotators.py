import numpy as np
import cv2

from tqdm import tqdm
from core.detections import load_detections, Detections

def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

class HeatMapAnnotator:
    def __init__(
        self,
        canvas_width: int,
        canvas_height: int,
        increase_rate: int = 16,
        increase_sigma: int = 3.5,
        increase_area: int = 100,
        decay_rate: int = 16,
        decay_attack: int = 4
    ):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height


        self.increase_rate = increase_rate
        self.increase_sigma = increase_sigma
        self.increase_area = increase_area

        self.decay_rate = decay_rate

        self.max_intensity = self.increase_rate * 16

        nprange = np.arange(self.max_intensity)

        # Precompute cube root values for integers 0 to max_intensity
        decay_factors = (nprange ** decay_attack) / ((len(nprange) - 1) ** decay_attack) * self.decay_rate
        self.decay_lookup = np.clip(nprange - decay_factors, 0, self.max_intensity - 1).astype(np.uint16)

        # Initialize an empty list to store heatmaps
        self.heatmap = np.zeros((canvas_height, canvas_width), dtype=np.uint16)  # Use NumPy array

    def decay_heatmap(self):
        non_zero_indices = self.heatmap != 0

        clipped_values = self.heatmap[non_zero_indices]
        
        self.heatmap[non_zero_indices] = self.decay_lookup[clipped_values]
    
    def increase_heatmap(self, boxes: np.ndarray):
        h, w = self.heatmap.shape

        for (x1, y1, x2, y2) in boxes:
            max_side = max((x2 - x1), (y2 - y1))
            
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

            pad = max_side * (self.increase_area ** 0.5 - 1) / 2

            tlx, tly, brx, bry = (
                int(max(center_x - max_side / 2 - pad, 0)),
                int(max(center_y - max_side / 2 - pad, 0)),
                int(min(center_x + max_side / 2 + pad, w)),
                int(min(center_y + max_side / 2 + pad, h))
            )
            
            # Create a grid of x, y coordinates
            #x_grid, y_grid = np.meshgrid(np.arange(tlx, brx), np.arange(tly, bry))

            # Calculate Gaussian intensity based on distance from the center
            #sigma = pad / self.increase_sigma # Standard deviation of the Gaussian filter
            #gaussian_intensity_x = gaussian(x_grid, center_x, sigma)
            #gaussian_intensity_y = gaussian(y_grid, center_y, sigma)
            #gaussian_intensity = gaussian_intensity_x * gaussian_intensity_y * self.increase_rate

            # Update the heatmap
            #self.heatmap[tly:bry, tlx:brx] += gaussian_intensity.astype(np.uint16)

            # new approach, just draw a circle in the heatmap
            mask = np.zeros((bry - tly, brx - tlx), dtype=np.uint16)
            cv2.circle(mask, (int(brx - tlx) // 2, int(bry - tly) // 2), int(max_side / 2 + pad), self.increase_rate, -1)
            
            # add circle to the heatmap
            self.heatmap[tly:bry, tlx:brx] += mask

    def tick(self, detections: Detections):
        if self.decay_rate > 0:
            self.decay_heatmap()
        if self.increase_rate > 0:
            self.increase_heatmap(detections.xyxy)

    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        self.tick(detections)
        print(np.unique(self.heatmap))

        normalized_heatmap = np.minimum(self.heatmap / self.max_intensity, 1)

        scaled_heatmap = (normalized_heatmap * 255).astype(np.uint8)

        if scene is not None:
            # red color
            color = (0, 0, 255)

            # create mask of color with transparency (rgba)
            mask = np.zeros((scene.shape[0], scene.shape[1], 4), dtype=np.uint8)

            # create alpha channel from scaled_heatmap
            alpha = scaled_heatmap

            mask[:, :, :3] = color
            mask[:, :, 3] = alpha

            # combine mask and image
            h, w = scene.shape[:2]

            for y in range(h):
                for x in range(w):
                    # apply mask
                    mask_color = mask[y, x, :3]
                    mask_alpha = mask[y, x, 3] / 255

                    scene_color = scene[y, x]

                    composite_color = scene_color * (1 - mask_alpha) + mask_color * mask_alpha

                    scene[y, x] = composite_color
            
            # temp = cv2.applyColorMap(scaled_heatmap, cv2.COLORMAP_TURBO)
            
            # scene = cv2.addWeighted(scene, 0.6, temp, 0.4, 0)

            return scene
        else:
            frame = cv2.applyColorMap(scaled_heatmap, cv2.COLORMAP_TURBO)
            return frame

    def annotates(background: np.ndarray, overlay: np.ndarray):
        background = cv2.imread('field.jpg')
        overlay = cv2.imread('dice.png', cv2.IMREAD_UNCHANGED)  # IMREAD_UNCHANGED => open image with the alpha channel

        height, width = overlay.shape[:2]
        for y in range(height):
            for x in range(width):
                overlay_color = overlay[y, x, :3]  # first three elements are color (RGB)
                overlay_alpha = overlay[y, x, 3] / 255  # 4th element is the alpha channel, convert from 0-255 to 0.0-1.0

                # get the color from the background image
                background_color = background[y, x]

                # combine the background color and the overlay color weighted by alpha
                composite_color = background_color * (1 - overlay_alpha) + overlay_color * overlay_alpha

                # update the background image in place
                background[y, x] = composite_color

        cv2.imwrite('combined.png', background)

def annotate_frame(
        frame,
        poi=None,
        poi_clamped=None,
        poi_new=None,
        detections=None,
        xyxy=None,
        stats=None,
    ):

    poi_color = (55, 100, 200)
    poi_color_secondary = (poi_color[0] + 50, poi_color[1] + 50, poi_color[2])
    poi_crosshair_thickness = 3
    poi_crosshair_length = 15

    label_font = cv2.FONT_HERSHEY_SIMPLEX

    if poi:
        # Draw real poi crosshair
        cv2.line(frame, (poi[0] - 10, poi[1]), (poi[0] + 10, poi[1]), poi_color_secondary, 2)
        cv2.line(frame, (poi[0], poi[1] - 10), (poi[0], poi[1] + 10), poi_color_secondary, 2)

    # TODO remove Draw new poi crosshair
    if poi_new:
        poi_color_new = (200, 200, 205)
        cv2.line(frame, (poi_new[0] - 10, poi_new[1]), (poi_new[0] + 10, poi_new[1]), poi_color_new, 2)
        cv2.line(frame, (poi_new[0], poi_new[1] - 10), (poi_new[0], poi_new[1] + 10), poi_color_new, 2)

    if poi_clamped:
        # Draw clamped poi crosshair
        cv2.line(frame, (poi_clamped[0] - poi_crosshair_length, poi_clamped[1]), (poi_clamped[0] + poi_crosshair_length, poi_clamped[1]), poi_color, poi_crosshair_thickness)
        cv2.line(frame, (poi_clamped[0], poi_clamped[1] - poi_crosshair_length), (poi_clamped[0], poi_clamped[1] + poi_crosshair_length), poi_color, poi_crosshair_thickness)

    if stats:
        # Draw stats
        for idx, stat in enumerate(stats):
            cv2.putText(frame, f"{stat}", (4, 20 + (idx * 20)), label_font, 0.6, poi_color, 2)

    if xyxy:
        tlx, tly, brx, bry = xyxy
        # Draw frame box
        cv2.rectangle(frame, (tlx, tly), (brx, bry), poi_color, 2)

    if detections:
        for bbox, _, conf, *_ in detections:
            x1, y1, x2, y2 = map(lambda x: int(x), bbox)

            cv2.rectangle(frame, (x1, y1), (x2, y2), poi_color, 2, cv2.LINE_AA)

            label_text = f'{conf:.3f}'
            label_color = poi_color
            label_txt_color=(255, 255, 255)
            label_font_scale = 0.5
            label_thickness = 1

            label_w, label_h = cv2.getTextSize(label_text, label_font, fontScale=label_font_scale, thickness=label_thickness)[0]

            label_offset = label_h + 3
            outside = y1 >= label_offset

            cv2.rectangle(frame, (x1, y1), (x1 + label_w, y1 - label_offset if outside else y1 + label_offset), label_color, -1, cv2.LINE_AA)
            cv2.putText(frame, label_text, (x1, y1 - 2 if outside else y1 + label_h + 2), label_font, label_font_scale, label_txt_color, label_thickness, cv2.LINE_AA)

if __name__ == '__main__':
    detections_path = "/notebooks/source/2023_11_29_team_ama/detect_bk/2023-11-29 22h 01m 27s_detections.npy"
    loaded, loaded_detections = load_detections(detections_path)

    if loaded:
        # split detections into chunks of an arbitrary size
        chunk_size = 10000
        chunks = [loaded_detections[i:i + chunk_size] for i in range(0, len(loaded_detections), chunk_size)]
        print("Processing {} chunks of max size {}".format(len(chunks), chunk_size))
        print("Total frames: {}".format(len(loaded_detections)))
        print("Last chunk size: {}".format(len(chunks[-1])))

        for chunk_id, chunk in enumerate(chunks):
            annotator = HeatMapAnnotator(1920, 1440, increase_rate=4, increase_area=2, decay_rate=0)
            scene = cv2.imread("/notebooks/scene.png")

            pbar = tqdm(total=len(chunk))

            for detections_id, detections in enumerate(chunk):
                if detections_id == len(chunk) - 1: 
                    frame = annotator.annotate(scene, detections)

                    cv2.imwrite(f"/notebooks/heatmap{chunk_id}.jpg", frame)
                    pbar.update(1)

                    break
                annotator.tick(detections)
                pbar.update(1)
            pbar.close()