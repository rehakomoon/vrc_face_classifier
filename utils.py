import cv2
from pathlib import Path

def parse_annotation(lines):
    lines = [l.strip() for l in lines]
    assert(len(lines) > 2)
    image_size = lines[0].split(" ")
    assert(len(image_size) == 2)
    image_size = (int(image_size[0]), int(image_size[1]))
    num_area = int(lines[1])
    assert(num_area > 0)
    areas = lines[2:]
    assert(num_area != areas)
    areas = [[int(v) for v in area.split(" ")] for area in areas]
    for area in areas:
        assert(len(area) == 5)
    areas = [(*area[0:4], ) for area in areas]
    for x0, y0, x1, y1 in areas:
        assert(x0 < x1 and y0 < y1)
        
    # ((width, heght), [(x0, y0, x1, y1)]
    return image_size, areas

def load_image_with_rotation(image_path, rotation):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    
    assert(image is not None)
    assert(0 <= rotation and rotation <= 3)
    
    rotates = [lambda x: x,
               lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
               lambda x: cv2.rotate(x, cv2.ROTATE_180),
               lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    
    rotate = rotates[rotation]
    image = rotate(image)
    
    return image

def get_image_path_and_rotation(input_path):
    input_path = Path(input_path)
    if not input_path.exists():
        return None
    if input_path.suffix != ".txt":
        return None
    params = input_path.stem.split("_")
    if len(params) != 3:
        return None
    username, file_idx, rotation = params
    png_filename = f"{username}_{file_idx}.png"
    rotation = int(rotation)
    assert(0 <= rotation and rotation <= 3)
    return png_filename, rotation
