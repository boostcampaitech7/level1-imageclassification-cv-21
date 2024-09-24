1. EDA Method
   - If the number of objects in the image is not not one(1) -> Crop each object and add it to the class folder
   <img src = "https://github.com/user-attachments/assets/978ee913-0d9f-4f19-8329-8e2a7c4fe4c7" width=600>
   
   - If image is nearly-related to the class, remove it
   - If noise makes recognizing an object hard (ex. logo, watermark) -> Erase it using white eraser
   <img src = "https://github.com/user-attachments/assets/c27f6c81-4d96-4c9a-92b5-1d3a9bc8d0e8" width=600>

2. The standard of edited images name :
   - If the number of edited images is one(1) : Add '_ed'/'_fixed' behind the file name (ex. sketch_05 -> sketch_05_ed)
   - If the number of edited images is not-one : Add '_ed_n/_fixed_n' behind the file name (ex. sketch_08 -> sketch_08_ed, sketch_08_ed_02, sketch_08_ed_03)

3. Introduction about files
   - 'data_to_csv.py' : Transform image folders into csv format 
   - 'count_class.py' : Count the number of images in each class
   - 'yolo_object_detection.py' : Detect objects using YOLO and save to CSV format
