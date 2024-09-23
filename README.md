1. Image들을 어떤 방식으로 전처리했는가
   - 특정 image에 object이 여러개 있는 경우 -> 각각을 crop 방식으로 오려낸 후, 또다른 image로 추가
   ![case_02](https://github.com/user-attachments/assets/978ee913-0d9f-4f19-8329-8e2a7c4fe4c7)
   - 해당 class와는 무관해보이는 image가 존재하는 경우 -> 삭제
   - Watermark, logo 등이 object의 특징을 왜곡할만한 경우 -> 흰색으로 처리 (blur)
   ![image (1)](https://github.com/user-attachments/assets/c27f6c81-4d96-4c9a-92b5-1d3a9bc8d0e8)


2. 전처리된 image의 이름 : 하나인 경우에는 뒤에 '_ed'를, 다수인 경우에는 '_ed_n' 방식으로 원본 이미지와 구분
   - ex. 원본(sketch_04) -> Editted version(sketch_04_ed, sketched_04_ed_02)
  
3. 어디까지 전처리 되었는가?
   - Class 폴더명 기준 'n14' ~ 'n19' (전체 dataset의 약 15%)
   - 25~30% 정도까지 작업해보고, '유의미한 성능 향상'이 있다면 더 작업해볼 생각입니다.

4. 'data_to_csv.py' : Image들이 담긴 폴더를 csv 형식으로 변환해줍니다.