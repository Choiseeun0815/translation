#include "opencv2/opencv.hpp" //헤더 파일 
#include <iostream> //헤더 파일
using namespace std; //std:: 생략
using namespace cv; //cv:: 생략
using namespace cv::dnn; //dnn:: 생략

Mat src, res_src, dst; //각각 원본 영상이 저장된 객체, 원본 영상을 (600x600)으로 크기 조정한 객체, 크기 조정이 완료된 객체를 color영상으로 변경한 객체.   src → res_src → dst
Mat dst1 ,res_dst ; //투시 변환 결과 영상이 저장된 객체, 해당 객체를 (500x500)으로 크기 조정한 객체
Mat  gray_res_dst, bin; //res_dst를 흑백 영상으로 변환한 결과가 저장된 객체, gray_res_dst의 이진화 결과를 저장한 객체.
Point2f srcPts[4], dstPts[4]; //입력 영상과 출력 영상에서의 네 점 좌표를 저장할 Point2f형 배열
Mat dst2;
void Save_trainData(int event, int x, int y, int flags, void* userdata); //마우스 이벤트 함수 선언부
int main()
{
	src = imread("a.jpg",IMREAD_GRAYSCALE); //미리 저장해둔 jpg 파일(원본 영상)을 흑백 영상으로 불러옴
	resize(src, res_src, Size(1000, 1000)); //원본 영상을 (600x600)으로 크기 조정
	cvtColor(res_src, dst, COLOR_GRAY2BGR); //크기 조정이 완료된 객체를 color영상으로 변환 (마우스 이벤트를 통한 빨간색 circle(점)을 그리기 위함. 
	cvtColor(res_src, dst2, COLOR_GRAY2BGR); //크기 조정이 완료된 객체를 color영상으로 변환 (마우스 이벤트를 통한 빨간색 circle(점)을 그리기 위함. 

	namedWindow("src"); //src 이름의 윈도우 창 생성
	setMouseCallback("src", Save_trainData); //src 윈도우 창에 마우스 이벤트 call

	imshow("src", dst); //dst 이미지를 화면에 출력 

	waitKey(0); //다음 입력까지 대기
	return 0; //함수 결과값 반환 
}
void Save_trainData(int event, int x, int y, int flags, void*) //마우스 이벤트 함수 정의부
{
	// ******************* 2022.11.27 새로 작성 *******************

	vector<string> classNames{ "A","B","C","E","H","I","L","M","N","O","R","T","V" };
	

	// ***********************************************************

	static int cnt = 0; //int형 정적 변수 선언 → 정적 변수로 선언하지 않을 경우, 마우스 이벤트가 call될 때마다 cnt값이 0으로 초기화됨. 이에 따라 하기의 if문이 실행되지 않음. 

	clock_t old_time = clock(); //함수 시작 시간에 발생하는 clock 수

	//[ 번역을 원하는 사각형 영역을 투시 변환을 이용하여 지정해주는 코드 ]
	if (event == EVENT_LBUTTONDOWN) //마우스 좌클릭을 할 때,
	{
		if (cnt < 4) //cnt의 값이 4미만인 경우, 
		{
			//투시 변환을 위해 사각형 영역을 좌측 상단 모서리 점부터 시작하여 시계 방향 순서로 선택함.
			srcPts[cnt++] = Point2f(x, y); //srcPts[0] => 좌측 상단 모서리 좌표   srcPts[1] => 우측 상단 모서리 좌표  srcPts[2] => 우측 하단 모서리 좌표  srcPts[3] => 좌측 하단 모서리 좌표 
			circle(dst, Point(x, y), 5, Scalar(0, 0, 255), -1); //마우스 좌클릭을 한 부분의 좌표에 빨간색 circle을 그림. 
			imshow("src", dst); //빨간색 circle을 그린 결과를 화면에 출력 

			if (cnt == 4) //cnt의 값이 4일 때, 
			{
				int w = res_src.cols, h = res_src.rows; //int형 변수 w는 res_src 영상의 행 정보를, h는 res_src 영상의 열 정보를 저장함.  => 투시 변환하여 만들 결과 영상의 가로와 세로 크기 

				//윈도우 창에서 사용자가 마우스로 선택한 사각형 꼭지점이 이동할 결과 영상 좌표를 설정
				dstPts[0] = Point2f(0, 0); dstPts[1] = Point2f(w - 1, 0); 
				dstPts[2] = Point2f(w - 1, h - 1); dstPts[3] = Point2f(0, h - 1);

				Mat pers = getPerspectiveTransform(srcPts, dstPts); //3x3 투시 변환 행렬을 pers 변수에 저장 
				
				warpPerspective(dst2, dst1, pers, Size(w, h)); //투시 변환을 수행하여 w x h 크기의 결과 영상 dst1을 생성한다. 
				resize(dst1, res_dst, Size(500, 500)); //투시 변환이 완료된 객체를 (500x500)으로 크기 조정 
				imshow("res_dst", res_dst); //투시 변환 및 크기 조정이 왼료된 영상을 "res_dst"윈도우 창에 출력  ************************
\
				cvtColor(res_dst, gray_res_dst, COLOR_BGR2GRAY); //res_dst 객체를 흑백 영상으로 변환 (이진화 수행을 위함)
				threshold(gray_res_dst, bin, 0, 255, THRESH_BINARY_INV | THRESH_OTSU); //gray_res_dst의 이진화 수행이 완료된 영상을 bin 객체에 저장. 배경이 흰색, 글씨가 검은색인 경우 THRESH_BINARY → THRESH_BINARY_INV를 이용
				
				Mat labels, stats, centroids; //bin 영상에 대해 레이블링을 수행하고, 각 객체 영역의 통계 정보가 저장될 객체 

				Mat color_dilate_dst, dilate_dst;
				dilate(bin, dilate_dst, Mat(), Point(-1, -1), 8); //bin 영상에 대해서 팽창 연산을 수행(10회)
				//cvtColor(bin, my_dst, COLOR_GRAY2BGR);
				cvtColor(dilate_dst, color_dilate_dst, COLOR_GRAY2BGR); //dilate_dst 영상을 컬러 영상으로 저장
				int cnt_object = connectedComponentsWithStats(dilate_dst, labels, stats, centroids); //객체의 개수를 저장할 int형 변수 

				for (int i = 1; i < cnt_object; i++) //검출된 단어 단위 객체의 수만큼 반복  
				{
					
					int* p = stats.ptr<int>(i); //배경 영역을 제외하고 흰색 객체 영역에 대해서만 for 반복문을 수행 
					if (p[4] < 20) continue; //객체의 픽셀 개수가 20보다 작으면 잡음으로 간주하고 무시
					

					rectangle(color_dilate_dst, Rect(p[0], p[1], p[2], p[3]), Scalar(0, 0, 255), 2); //검출된 객체 영역을 감싸는 바운딩 박스를 빨간색으로 그림 

					string desc = format("%d", i); //객체의 번호를 저장할 string형 변수
					putText(color_dilate_dst, desc, Point(centroids.at<double>(i, 0), centroids.at<double>(i,1))
						, FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 0),1); //객체의 무게 중심 좌표에 객체 번호를 파란색으로 출력

					Mat words = bin(Rect(p[0], p[1], p[2], p[3])); //이진화가 완료된 i번째 객체의 바운딩 박스(단어 영역으로 간주) 좌표를 객체에 저장
					Mat res_words; 
					resize(words, res_words, Size(500, 500)); //검출된 단어 영역을 (400x400)으로 크기 조정 

					//[결과 확인을 위한 코드]
					

					Mat labels_for_words, stats_for_words, centroids_for_words; //단어 영역의 영상에 대해 레이블링을 수행하고, 각 객체 영역의 통계 정보가 저장될 객체 
					
					Mat color_res_words;
					cvtColor(res_words, color_res_words, COLOR_GRAY2BGR); //res_words 객체를 컬러 영상으로 변환 
					int cnt_object2 = connectedComponentsWithStats(res_words, labels_for_words, stats_for_words, centroids_for_words); //객체의 개수를 저장할 int형 변수  

					for (int j = 1; j < cnt_object2; j++) //검출된 문자 단위 객체의 수만큼 반복  
					{
						int* k = stats_for_words.ptr<int>(j); //배경 영역을 제외하고 흰색 객체 영역에 대해서만 for 반복문 수행
						if (k[4] < 20) continue; //객체의 픽셀 개수가 20보다 작으면 잡음으로 간주하고 무시 

						rectangle(color_res_words, Rect(k[0], k[1], k[2], k[3]), Scalar(0, 0, 255), 2); //검출된 객체 영역을 감싸는 바운딩 박스를 빨간색으로 그림

						string desc_words = format("%d", j); //객체의 번호를 저장할 string형 변수 
						putText(color_res_words, desc_words, Point(centroids_for_words.at<double>(j, 0), centroids_for_words.at<double>(j, 1))
							, FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 0), 1); //객체의 무게 중심 좌표에 객체 번호를 파란색으로 출력

						Mat chars = res_words(Rect(k[0], k[1], k[2], k[3])); //j번째 객체의 바운딩 박스(문자 영역) 좌표를 객체에 저장 
						Mat res_chars; 
						resize(chars, res_chars, Size(100, 100)); //검출된 문자 영역을 (200x200)으로 크기 조정
						Mat M = Mat_<double>({ 2,3 }, { 1,0,100,0,1,100 });
						Mat rrr; 
						warpAffine(res_chars, rrr, M, Size(300, 300));

						// ******************* 2022.11.27 새로 작성 *******************

						//cout << i << "번째 단어의 문자 인식 결과 : ";
						Mat res_chars_color;
						cvtColor(rrr, res_chars_color, COLOR_GRAY2BGR);
						
						string filename = "word_boxN2" + to_string(i) + "_" + to_string(j) + ".jpg";
						imwrite(filename, res_chars_color);
						
					}


				}

				imshow("bin", bin); //이진화가 완료된 영상을 "bin"창에 출력
				imshow("dilate", color_dilate_dst); //팽창 연산이 완료된 영상을 "dilate"영상에 출력 

			}
		}
	}
}
