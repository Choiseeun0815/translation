import cv2 #openCV를 사용하기 위한 패키지 import

#~ keras 모델 파일을 사용하기 위한 패키지 import ~
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from selenium import webdriver #웹브라우저를 직접적으로 조작할 수 있도록 해주는 패키지 import
from webdriver_manager.chrome import ChromeDriverManager #웹브라우저를 직접적으로 조작할 수 있도록 해주는 패키지 import
from selenium.webdriver.common.by import By #xpath를 이용한 크롤링을 위해 패키지 import

model = load_model('keras_model.h5',compile=False) #모델 파일 불러오기
class_names = open('labels.txt','r').readlines() #label 정보 불러오기
data = np.ndarray(shape=(1,224,224,3),dtype=np.float32) #224x224 형태로 변경

driver = webdriver.Chrome("C:\\Users\\user\\Desktop\\sv_project\\chromedriver.exe") #다운받은 크롬드라이버의 주소 
input_img = cv2.imread("img.jpg",cv2.IMREAD_GRAYSCALE) #저장해둔 .jpg 파일을 흑백 영상으로 불러옴
cnt_pts = 0
srcPts = np.zeros((4,2),dtype=np.float32) #입력 영상과 출력 영상에서의 모서리 점 좌표를 저장할 배열
dstPts = np.zeros((4,2),dtype=np.float32) #입력 영상과 출력 영상에서의 모서리 점 좌표를 저장할 배열 



# ~ 영상을 입력하고 해당 영상을 어파인 변환을 위한 함수에 전달하는 역할 ~ 
def Affine_img(src):
    global res_src #전역 변수로 선언
    res_src = cv2.resize(src,(800,800)) #입력 영상을 (800x800)으로 크기 조정 → 이미지 파일이 너무 클 경우를 대비

    global dst,my_dst #전역 변수로 선언
    dst = cv2.cvtColor(res_src,cv2.COLOR_GRAY2BGR) #resize가 완료된 객체를 color 영상으로 변환 → 마우스 이벤트를 통한 빨간 점 그리기 위함
    my_dst = cv2.cvtColor(res_src,cv2.COLOR_GRAY2BGR)
    cv2.namedWindow("src") #src 이름의 윈도우 창 생성
    cv2.setMouseCallback("src",on_mouse) #src 윈도우 창에 마우스 이벤트 call

    cv2.imshow("src",dst) #dst 이미지를 화면에 출력
    cv2.waitKey() #다음 입력까지 대기


# ~ 마우스 이벤트 함수 ~ 
def on_mouse(event, x, y, flags, para,):
    global cnt_pts #입력 받은 좌표의 수를 저장하기 위한 변수

    if event == cv2.EVENT_LBUTTONDOWN: #마우스 좌 클릭을 할 때, 
        if cnt_pts < 4: #cnt_pts 값이 4미만인 경우, 

            #srcPts[0] => 좌측 상단 모서리 좌표   srcPts[1] => 우측 상단 모서리 좌표  
            # srcPts[2] => 우측 하단 모서리 좌표  srcPts[3] => 좌측 하단 모서리 좌표 
            srcPts[cnt_pts] = [x,y] 

            cv2.circle(dst,(x,y),5,(0,0,255),-1) #클릭한 부분에 빨간색 circle을 그림
            cv2.imshow("src",dst) #빨간색 circle을 그린 결과를 화면에 출력
            cnt_pts += 1 #클릭 횟수를 저장

            if cnt_pts == 4: #cnt_pts의 값이 4일 때,
                w = 600 #영상의 행 정보
                h = 600 #영상의 열 정보

                #윈도우 창에서 사용자가 마우스로 선택한 모서리 좌표가 이동할 결과 영상 좌표를 설정
                dstPts[0] = [0,0]
                dstPts[1] = [w-1,0]
                dstPts[2] = [w-1,h-1]
                dstPts[3] = [0,h-1]

                pers = cv2.getPerspectiveTransform(srcPts,dstPts) #3x3 투시 변환 행렬을 pers 변수에 저장
                dst1 = cv2.warpPerspective(my_dst,pers,(w,h)) #투시 변환을 수행하여 w x h 크기의 결과 영상 dst1을 생성
                res_dst = cv2.resize(dst1,(500,500)) #투시 변환이 완료된 객체를 500x500으로 크기 조정
                cv2.imshow("res_dst",res_dst) #투시 변환 결과를 화면에 출력

                Get_word(res_dst) #함수 호출

                
# ~ 단어 영역을 검출하기 위한 함수 ~ 
def Get_word(res_dst):
    gray_res_dst = cv2.cvtColor(res_dst,cv2.COLOR_BGR2GRAY) #res_dst 객체를 흑백 영상으로 변환
    _, bin = cv2.threshold(gray_res_dst,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) #gray_res_dst의 이진화 수행이 완료된 결과를 bin에 저장

    global color_bin #이진 영상에 번역 결과를 작성하기 위하여 새로운 전역 객체 선언
    color_bin = cv2.cvtColor(bin,cv2.COLOR_GRAY2BGR) #bin 객체를 컬러 영상으로 변환
    color_bin = Image.fromarray(color_bin) #cv2 이미지를 pillow 이미지로 변환 (이미지에 한글 출력을 위함)

    cv2.imshow("bin",bin) #이진 영상을 화면에 출력

    kernel = np.ones((3, 3), np.uint8) #( 3x3 )커널

    dilate_dst = cv2.dilate(bin,kernel,iterations = 18) #bin 형상에 대해서 팽창 연산을 수행(18회)
    color_dilate_dst = cv2.cvtColor(dilate_dst,cv2.COLOR_GRAY2BGR) #dilate_dst 영상을 컬러 영상으로 저장

    cnt_object, labels, stats, centroids = cv2.connectedComponentsWithStats(dilate_dst) #객체에 대한 여러 정보를 저장

     #검출된 단어 단위 객체의 수만큼 반복
    for i in range(1,cnt_object):
        (x, y, w, h, area) = stats[i] #검출된 객체에 대한 정보를 저장
        if area < 100: #객체의 픽셀 개수가 100보다 작으면 잡음으로 간주하고 무시
            continue
        
       
        cv2.rectangle(color_dilate_dst,(x, y, w, h),(0,0,255)) #검출된 객체 영역을 감싸는 빨간색 바운딩 박스
        
        words = bin[y:y + h,x:x + w] #관심 영역의 범위를 저장

        res_words = cv2.resize(words,(400,400)) #검출된 단어를 400x400으로 크기 조정
        Get_Alpha(i,res_words,x,y) #입력으로 검출된 단어 정보, 바운딩 박스의 좌측 상단 x,y 좌표를 받음



    cv2.imshow("a",color_dilate_dst) #팽창 연산 수행 결과를 화면에 출력


# ~ 문자 영역을 검출하기 위한 함수 ~ 
def Get_Alpha(i,res_words,p_x,p_y): #검출된 문자 단위 객체의 수만큼 반복
    color_res_words = cv2.cvtColor(res_words,cv2.COLOR_GRAY2BGR) #res_words 객체를 컬러 영상으로 변환

    global color_res_words2 #빨간색 바운딩 박스를 그려주기 위한 객체
    color_res_words2 = cv2.cvtColor(res_words,cv2.COLOR_GRAY2BGR) #res_words 객체를 컬러 영상으로 변환
    cnt_object2,labels_for_words,stast_for_words,centroids_for_words = cv2.connectedComponentsWithStats(res_words) #객체에 대한 정보 저장

    global rows 
    rows = cnt_object2 - 1 #검출된 문자 단위 객체의 수를 저장

    global get_x
    get_x = [0] * rows #(1 x 검출된 문자 단위 객체 수) 크기의 list 자료형 행렬 

    global alpha
    alpha = [0] * rows #(1 x 검출된 문자 단위 객체 수) 크기의 list 자료형 행렬 

    #검출된 문자 단위 객체의 수만큼 반복 
    for j in range(1,cnt_object2):
        (x, y, w, h, area) = stast_for_words[j] #검출된 객체에 대한 정보를 저장
        if area < 50: #객체의 픽셀 개수가 50보다 작으면 잡음으로 간주하고 무시
            continue
        
        cv2.rectangle(color_res_words2,(x, y, w, h),(0,0,255)) #검출된 객체 영역을 감싸는 빨간색 바운딩 박스

        chars = color_res_words[y:y + h,x:x + w] #관심 영역의 범위를 저장
        res_chars = cv2.resize(chars,(100,100)) #검출된 문자 영역을 100x100으로 크기 조정
        mtrx = np.float32([[1,0,100], [0,1,100]]) #x축으로 100, y축으로 100만큼 이동하는 변환 행렬 생성
        
       
        move_chars = cv2.warpAffine(res_chars,mtrx,(300,300)) #기존의 문자 객체가 300x300 영상의 중앙에 오도록 이동 변환
        
        alpha[j-1] = Show_alpha(i,j,move_chars) #추론된 문자의 결과를 alpha 행렬의 [j-1]번째에 저장
        get_x[j-1] = x #추론된 문자 객체의 바운딩박스 좌측 상단 x좌표를 행렬의 [j-1]번째에 저장

    

    print(get_x) #저장된 x좌표를 출력
    print(alpha) #저장된 추론 결과를 출력
    Get_Voca(p_x,p_y) #단어 단위 객체의 바운딩 박스의 좌측 상단 x,y 좌표를 받음 → 해당 좌표에 번역 결과를 적어주기 위함
    # name1 = "w_{0}".format(i)
    # cv2.imshow(name1,color_res_words2)


# ~ 검출한 문자를 추론해주는 함수 ~ 
def Show_alpha(i,j,get_img):
    size = (224,224) #( 224x224 )
    img = cv2.resize(get_img,size) #입력된 이미지를 size 크기만큼 resize

    img_array = np.asarray(img) #이미지를 numpy 행렬으로 변환 

    normal_img_array = (img_array.astype(np.float32)/127.0) - 1 #스케일 정규화
    
    data = np.array([normal_img_array]) 

    prediction = model.predict(data) #추론 결과
    index = np.argmax(prediction) #인덱스 값
    class_name = class_names[index] #알파벳 추론 결과 (클래스 번호 + 클래스 이름 + '?')
    class_name = class_name[:-1] #추론 결과에서 마지막에 출력되는 '?'를 삭제
    class_name = class_name[-1:] #추론 결과의 클래스 이름만 저장
    
    confidence_sorce = prediction[0][index] #추론 결과의 인식률을 float 형태로 저장

    cf_str = str(confidence_sorce) #confidence_sorce를 string형으로 변환

    cv2.putText(get_img,class_name,(10,30),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2) #추론 결과 클래스의 이름을 화면에 출력
    cv2.putText(get_img,cf_str,(10,50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2) #추론 결과의 인식률을 화면에 출력
    #name1 = "w_{0}{1}".format(i,j)
    #cv2.imshow(name1,get_img) → 문자 추론 결과를 보여줌. 테스트 과정이 아니라면 주석처리

    return class_name #추론된 클래스의 이름을 반환

# ~ 사전에 검색할 단어를 정리해주는 함수 ~ 
def Get_Voca(x,y):
    arr = [[0 for k in range(2)]for j in range(rows)] #(rows x 2) 크기의 배열 선언

    for i in range(rows):
        arr[i][0] = get_x[i] #i행 0열에 문자의 x 좌표를 저장
        arr[i][1] = alpha[i] #i행 1열에 문자의 추론 결과를 저장

    arr.sort(key=lambda x:x[0]) #저장된 문자 영역을 x좌표 기준 오름차순으로 정렬
    print(arr)
    global voca1 #전역 변수
    voca1 = "" #변수 초기화 
    for i in range(rows): #검출된 문자 단위 객체의 수만큼 반복 
        voca1 = voca1 + arr[i][1] #arr의 [i]번째 문자 추론 결과를 voca1에 중첩하여 저장 
    
    print(voca1) #결과를 터미널 창에 출력

    global ans #번역 결과를 저장할 전역 변수
    ans = Find_word(voca1) #voca1의 번역 결과를 ans에 저장
    Write_ans(ans,x,y) #번역 결과와 단어 객체 영역 바운딩박스 좌측 상단 좌표 정보를 입력
     
# ~ 인식한 단어를 네이버 영사전에서 검색해주는 함수 ~ 
def Find_word(voca):
    
    url = 'https://en.dict.naver.com/#/main' #네이버 영어 사전의 url을 기입
    driver.get(url) #입력한 url을 브라우저에 띄움
    driver.implicitly_wait(10) #10초안에 웹페이지를 load하면 넘어가고, 그렇지 않을 경우 10초를 기다림

    xpath = '//input[@id="ac_input"]' #F12를 누르고 검색 영역에 대한 tag 정보를 입력

    driver.find_element(By.XPATH,xpath).send_keys(voca.lower()+'\n') #인식한 단어를 모두 소문자로 변경 후 네이버 사전에 실시간으로 검색

    xpath2 = '//div[@class="component_keyword has-saving-function"]' #F12를 누르고 검색 결과 영역에 대한 tag 정보를 입력
    xpath3 = '//strong[@class="highlight"]' #F12를 누르고 검색 결과 영역 -> 입력된 단어에 대한 tag 정보를 입력
    xpath4 = '//p[@class="mean"]' #단어 검색 결과가 (뜻 의미1, 의미2 .... )와 같이 저장됨
    
    try:
        temp_voca = driver.find_element(By.XPATH,xpath2 + xpath3).text #검색 결과로 나온 단어
        trans_voca = driver.find_element(By.XPATH,xpath4).text #번역 정보가 저장됨. (품사  단어1, 단어2 ...)
        
        xpath5 = '//span[@class="word_class "]' #단어 검색 결과의 품사가 저장됨
        get_word_class = driver.find_element(By.XPATH,xpath5).text #단어 검색 결과의 품사가 저장됨
        result = trans_voca.strip(get_word_class) #번여 정보에서 품사 정보를 제거함
        print("인식 단어 : " + voca + "   |   번역 결과 : " + result+"\n")
        
    except Exception: #예외 처리를 진행 / 사전에 존재하지 않는 단어가 검색 되었을 경우 진행
        result = voca1 #번역에 실패하였을 경우, 추론된 결과를 저장
        print("\n사전에 존재하지 않는 단어입니다.") #결과를 터미널창에 출력

    return result #번역 결과를 반환

# ~ 번역 결과를 윈도우 창에 출력해주는 함수 ~ 
def Write_ans(ans, x, y): #입력 : ( 번역 결과, 문자 객체의 바운딩 박스 x좌표, 문자 객체의 바운딩 박스 y좌표 )
    fontpath = "fonts/gulim.ttc"
    font = ImageFont.truetype(fontpath,18) #글씨체와 크기를 저장
    b,g,r,a = 0,255,255,255 #색상 정보를 저장
    draw = ImageDraw.Draw(color_bin,'RGBA') #color_bin 영상에 컬러로 작성
    draw.text((x,y),ans,font=font,fill=(b,g,r,a)) #검출된 단어 객체의 상단에 번역 결과를 노란색으로 작성 

    img_numpy = np.array(color_bin) #pillow 형태 이미지에서 cv2 이미지로 변환

    mtrx = np.float32([[1,0,100],[0,1,100]]) #x축으로 100, y축으로 100만큼 이동하는 변환 행렬 생성
    ans_img = cv2.warpAffine(img_numpy,mtrx,(800,800)) #영상의 크기를 800x800으로 변경
 
    cv2.imshow("result",ans_img) #결과를 화면에 출력

Affine_img(input_img) #전체 실행을 위한 이미지 객체 입력