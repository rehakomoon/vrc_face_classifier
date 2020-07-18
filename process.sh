python 01_collect_images.py
python 02_generate_labels_from_animeface_detector.py
python 03_generate_face_dataset.py

read -p "Update positive_list.txt from face classification. Wait..."

python 04_filter_labels.py
python 05_collect_corresponding_images_by_annotations.py
python 06_amalgamate_annotations.py
python 07_generate_full_dataset_for_inference.py

read -p "Update inference.pkl from face detection. Wait..."

python 08_convert_inferenced_data_to_annotation.py
python 09_remove_full_dataset_images.py
python 10_merge_annotations.py
