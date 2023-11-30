Computer_Vision_Project.zip   →  객체 검출을 통해 훈련 데이터를 얻는 코드

sv_project   →   main 파이썬 코드

trainData  →  수집한 훈련용 데이터 

# 블록도
![image](https://github.com/Choiseeun0815/Calendar/assets/103297048/129e3db6-6c4e-42fd-a722-051ff97653ec)

[문자 인식을 활용한 머신 러닝 번역기] 프로젝트. 
해당 프로젝트는 사용자 본인의 필체로 작성한 영단어를 인식하여 해당 단어의 번역 결과를 출력해주는 시스템이다.

# 1. 번역 영역 추출
![image](https://github.com/Choiseeun0815/Calendar/assets/103297048/ec0da7fb-4d64-46cc-9a56-c1744fc080da)

입력받은 흑백 영상을 800x800 형태로 크기를 조정해주고 마우스 이벤트 함수를 호출.

호출된 마우스 이벤트 함수에서는 원하는 사각형 영역의 네 좌표를 시계 방향으로 입력하고,

입력된 사각형 영역을 500x500으로 투시 변환해 준다. 

# 2. 이진화 및 단어 영역 인식
![image](https://github.com/Choiseeun0815/Calendar/assets/103297048/9da3f705-ff4b-448b-beed-f16d687e4a24)

배경 영역과 객체 영역(문자 영역)을 구분하기 위하여 이진화를 사용.

이진화 후에는 모폴로지 팽창 연산 및 레이블링을 수행하여 검출된 단어 영역을 하나의 바운딩 박스로 표시하고,

이의 좌표 정보를 저장한다. 

# 3. 문자 영역 인식
![image](https://github.com/Choiseeun0815/Calendar/assets/103297048/7a700677-523a-437e-931f-a2f27194c3f6)

단어 영역의 영역마다 레이블링을 통한 문자 단위의 객체 분류를 수행.

검출된 문자 영역을 100x100으로 크기 조정 후, 300x300 영상의 중앙에 오도록 이동 변환을 수행한다. 

# 4. 필기체 문자 학습
![image](https://github.com/Choiseeun0815/Calendar/assets/103297048/f5adec44-c94f-42f0-9aef-28121444bdf8)

상기의 과정들을 활용하여 알파벳 훈련 데이터를 수집한다. 

학습을 수행시킬 필기체 알파벳을 수기로 작성하고, 해당 이미지를 이진화 후 객체 검출을 통하여 .jpg 확장자로 저장한다.

13종류의 훈련 데이터(대략 2000개)를 티쳐블 머신을 통하여 모델 파일을 확보한다. 

# 5. 필기체 문자 인식
![image](https://github.com/Choiseeun0815/Calendar/assets/103297048/db33d58d-ca61-43d6-b05b-f81d153db3dd)

검출된 문자의 추론 결과를 보여준다. 

글씨를 과하게 기울여 쓰지 않는 이상 대부분 80%의 높은 인식률을 보여준다. 

# 6. 단어 번역
![image](https://github.com/Choiseeun0815/Calendar/assets/103297048/359e3be2-11f2-4afc-baed-40471bc7981b)

크롬을 사용한 네이버 영사전과의 웹크롤링 방식을 통하여 실시간으로 영단어 검색 및 이에 대한 결과를 반환.

존재하지 않는 단어가 입력되었을 경우, 예외 처리를 통하여 검색 실패 결과를 반환한다. 

# 7. 번역 화면
![image](https://github.com/Choiseeun0815/Calendar/assets/103297048/ca60ee92-5b17-4e05-8070-7db8936b0c04)

이진화가 완료된 영상에 번역 결과를 작성해준다. 

번역에 실패하였을 경우, 추론 결과를 적어준다. 

# 구동 영상

https://github.com/Choiseeun0815/Calendar/assets/103297048/7613b942-f719-410e-a720-eeb8aa058e2a

