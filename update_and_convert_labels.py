import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import glob
import argparse

def json_to_pascal_voc(json_file, output_dir):
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)
    
    for image_info in data['images']:
        image_id = image_info['id']
        file_name = image_info['file_name']
        width = image_info['width']
        height = image_info['height']
        
        # Create XML structure
        annotation = ET.Element("annotation")
        
        folder = ET.SubElement(annotation, "folder").text = "VOC2012"
        filename = ET.SubElement(annotation, "filename").text = file_name
        
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"
        
        segmented = ET.SubElement(annotation, "segmented").text = "1"
        
        for annotation_info in data['annotations']:
            if annotation_info['image_id'] == image_id:
                obj = ET.SubElement(annotation, "object")
                category_id = annotation_info['category_id']
                
                # Find the category name
                category_name = ""
                for category in data['categories']:
                    if category['class_id'] == category_id:
                        category_name = category['class_name']
                        break
                
                ET.SubElement(obj, "name").text = category_name
                ET.SubElement(obj, "pose").text = "Unspecified"
                ET.SubElement(obj, "truncated").text = "0"
                ET.SubElement(obj, "difficult").text = "0"
                
                bndbox = ET.SubElement(obj, "bndbox")
                bbox = annotation_info['bbox']
                ET.SubElement(bndbox, "xmin").text = str(int(bbox[0]))
                ET.SubElement(bndbox, "ymin").text = str(int(bbox[1]))
                ET.SubElement(bndbox, "xmax").text = str(int(bbox[0] + bbox[2]))
                ET.SubElement(bndbox, "ymax").text = str(int(bbox[1] + bbox[3]))
                
                # Add segmentation points
                seg = ET.SubElement(obj, "segmentation")
                for point in annotation_info['segmentation']:
                    point_elem = ET.SubElement(seg, "point")
                    ET.SubElement(point_elem, "x").text = str(point['x'])
                    ET.SubElement(point_elem, "y").text = str(point['y'])
        
        # Write to XML file
        tree = ET.ElementTree(annotation)
        xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="    ")
        output_file = os.path.join(output_dir, f"{image_id}.xml")
        with open(output_file, "w") as f:
            f.write(xml_str)

def process_json_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    for json_file in json_files:
        json_to_pascal_voc(json_file, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert JSON annotations to Pascal VOC XML format.")
    parser.add_argument("--input_dir", required=True, help="Directory containing JSON files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save Pascal VOC XML files.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_json_files(args.input_dir, args.output_dir)
