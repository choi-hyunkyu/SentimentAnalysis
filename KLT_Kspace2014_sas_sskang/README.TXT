한글 자동 띄어쓰기 실행파일 사용법


1) 압축 파일을 푼 다음에

2) 윈도 cmd창 실행 -- 윈도화면 좌하단 "시작" 버튼, "프로그램 및 파일 검색" 입력창에 "cmd" 입력한 후 Enter

3) cd 명령으로 프로그램이 설치된 폴더로 이동

    <방법> 윈도탐색기의 맨 윗쪽에 "폴더 경로" 를 cut-and-paste

              1. 절대경로를 마우스로 블록을 잡은 후에 ctrl-C 로 cut 한 다음에

              2. 아래와 같이 cmd 창에서 "cd " 입력한 후에 

                   C> cd 

              3. 마우스 우측버튼 눌러서 "붙여넣기"를 하고 <Enter> 키를 치면 해당 폴더로 이동함

4) 실행


   C> asp.exe <입력파일.txt> <출력파일.txt>


   <참고> "출력 파일" 명시하지 않으면 화면으로 출력함. "입-출력 파일" 모두 입력하지 않으면 키보드에서 입력을 받음


<<주의 사항>>

1. 입-출력 파일은 EUC-KR 로 되어 있음 -- UTF8 텍스트 파일은 아래 uniconv.exe에 의해 UEC-KR로 변환

   C> uniconv.exe -tk input.txt input2.txt

   C> asp.exe input2.txt ouput2.txt

   C> uniconv.exe -kt output2.txt output.txt


2. 입력파일이 1 라인으로 되어 있을 경우 -- 한 line의 최대길이가 4096 바이트로 제한되어 있음

   이 경우에 저에게 연락하기 바랍니다. nlpkang@gmail.com
