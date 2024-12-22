import os
import xml.etree.ElementTree as ET

def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    label = root.find('object/name').text  # Extract the object name
    return label

def class_to_index(label, class_names):
    return class_names.index(label)
