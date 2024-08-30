import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

# XML file path
xml_dir = "/workspace/dataset/data/VOCdevkit/VOC2007/Annotations"
# Image file path 
image_dir = "/workspace/dataset/data/VOCdevkit/VOC2007/JPEGImages"
# Mask image file path
mask_dir = "/workspace/dataset/data/VOCdevkit/VOC2007/SegmentationClass"

# Match class names to colors
class_color_map = {
    "0": (255, 0, 0),    # red
    "1": (0, 255, 0),    # green
    "2": (0, 0, 255),    # blue
    "3": (255, 255, 0),  # yellow
    "4": (0, 255, 255)   # cyan
}

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        polygon = []
        for pt in obj.find('segmentation').findall('point'):
            x = int(float(pt.find('x').text))
            y = int(float(pt.find('y').text))
            polygon.append((x, y))
        objects.append((name, polygon))
    
    return objects

# Generate mask image
def create_mask(image_size, objects):
    mask = Image.new('RGB', image_size, (0, 0, 0))  
    draw = ImageDraw.Draw(mask)
    
    for obj in objects:
        class_name, polygon = obj
        color = class_color_map.get(class_name, (0, 0, 0))  
        print(f"Class: {class_name}, Color: {color}") 
        draw.polygon(polygon, outline=color, fill=color)
    
    return mask

if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

for xml_file in os.listdir(xml_dir):
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(xml_dir, xml_file)
        image_file = xml_file.replace(".xml", ".jpg")
        image_path = os.path.join(image_dir, image_file)
        
        if os.path.exists(image_path):
            image = Image.open(image_path)
            image_size = image.size
            
            objects = parse_xml(xml_path)
            
            mask = create_mask(image_size, objects)
            
            mask_file = xml_file.replace(".xml", ".png")
            mask_path = os.path.join(mask_dir, mask_file)
            mask.save(mask_path)
            print(f"Saved mask to {mask_path}")

print("Mask image generation completed.")
