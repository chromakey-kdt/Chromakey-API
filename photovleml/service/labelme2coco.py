import math

import numpy as np
import PIL.ImageDraw


def convert_labelme2coco(img_shape, points, shape_type=None, line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)

    mask = np.array(mask, dtype=bool).astype(np.int)

    return mask


# path = "./sample_data/frame0.json"
# assert os.path.isfile(path)
#
# with open(path, "r", encoding="utf-8") as f:
#     dj = json.load(f)


# mask = shape_to_mask((dj['imageHeight'], dj['imageWidth']), dj['shapes'][0]['points'], shape_type=None, line_width=1, point_size=1)
#
# mask_img = mask.astype(np.int)
#
# mask_img = mask_img * 255
#
# cv2.imwrite("sample_img.png", mask_img)
