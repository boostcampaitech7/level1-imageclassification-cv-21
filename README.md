1. EDA Method
   - If the number of object in the image is not not one(1) -> Crop each objects and add it to the class folder
   ![case_02](https://github.com/user-attachments/assets/978ee913-0d9f-4f19-8329-8e2a7c4fe4c7 | width=100)
   - If image is nearly-related to the class -> Remove it
   - If noise makes recognizing object hard (ex. logo, watermark) -> Erase it using white eraser
   ![image (1)](https://github.com/user-attachments/assets/c27f6c81-4d96-4c9a-92b5-1d3a9bc8d0e8 | width=100)

2. The standard of naming editted image :
   - If the number of editted image is one(1) : Add '_ed'/'_fixed' behind the file name (ex. sketch_05 -> sketch_05_ed)
   - If the number of editted image is not-one : Add '_ed_n/_fixed_n' behind the file name (ex. sketch_08 -> sketch_08_ed, sketch_08_ed_02, sketch_08_ed_03)

3. Introduction about '.py' files
   -  'data_to_csv.py' : Transform image folders into csv format 
   -  'count_class.py' : Count the number of images in each class
