## 프로젝트 이름 
STD(Scene Text Detection)/STR(Scene Text Recognition)
<br>
### <a href="https://github.com/UpstageAILab5/industry-partnership-project-brainventures/blob/main/%EA%B8%B0%EC%97%85%EC%97%B0%EA%B3%84%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8_%EB%B8%8C%EB%A0%88%EC%9D%B8%EB%B2%A4%EC%B2%98%EC%8A%A4_PPT.pdf" target="_blank">최종 발표자료</a>

## 프로젝트 소개

- OCR (Scene Text Recognition)
- 자연스러운 환경에서 촬영된 한글 텍스트 이미지에서 문자를 인식하는 AI 모델 개발 
- 문서배경의 글자가 아닌 일상 자연환경 속의 글자를 인식



### 도전 과제
- 모든 문자를 읽겠다는 마음으로 최대한 가용한 오픈소스를 많이 테스트 해 보아서 최선의 솔루션을 도출하려고 노력했다.

<br>

## 팀 구성원   

![image](https://github.com/user-attachments/assets/135934f8-4646-41b9-b204-86243ad7f98c)

<br>

## 1. 개발 환경 및 기술 스택
- 주 언어 : Python
- 버전 및 이슈관리 : 3.10.14 ~ 3.10.16
- 협업 툴 : github, notion, zoom

<br>

## 2. 프로젝트 구조
📁 Project Root  
├── 📁 DU  
├── 📁 HI  
├── 📁 JE  
├── 📁 JH  
├── 📁 Mertric  
├── 📁 output  
├── 📄 .gitattributes  
├── 📄 .gitignore  
└── 📄 README.md


## 3. 구현 기능
### 기능1
- 데이터 정제 (AI_HUB에서 받은 이미지와 json 파일에서, json 파일의 bounding box 좌표 형식이 우리가 사용하려고 하는 labeling tool, model, metrix 소스코드와 달라 변형하는 작업이 필요하였다
### 기능2
- 모델을 통해 검출된 bbox를 이미지에 덧그려서 검수하는 작업이 필요하였다
### 기능3
- 최신 논문을 코드로 구현하여 성능 비교

<br>

## 4. 작품 아키텍처(필수X)
- #### _아래 이미지는 예시입니다_
![image](https://github.com/user-attachments/assets/4d68ba0c-da7b-45eb-97a1-a8f3e57e5a8c)


# 데이터 다운로드
# aihub 의 api 키로 데이터 요청 
다운받으려는 데이터 신청 접수( 다운로드 클릭후 팝업창 확인 요망 ) 
https://www.aihub.or.kr/devsport/apishell/list.do?currMenu=403&topMenu=100 에서 api 키 발급 

# AIHUB 데이터 다운 명령어
curl -o "aihubshell" https://api.aihub.or.kr/api/aihubshell.do

chmod +x aihubshell

 ls -al | grep aihubshell

aihubshell -mode l -datasetkey 105

aihubshell -mode d -datasetkey 105 -aihubapikey 'API_KEY키 발급'

# 특정 버전만 다운로드
aihubshell -mode d -datasetkey 105 -filekey 68070,68071,68072,68073 -aihubapikey 'API_KEY'

sudo apt update
sudo apt install locales
sudo locale-gen ko_KR.UTF-8
export LANG=ko_KR.UTF-8

unzip -O cp949 /data/ephemeral/home/industry-partnership-project-brainventures/data/01.data/1.Training/label/TL1.zip -d /data/ephemeral/home/industry-partnership-project-brainventures/data/01.data/1.Training/label
unzip -O cp949 /data/ephemeral/home/industry-partnership-project-brainventures/data/01.data/1.Training/origin/TS1.zip -d /data/ephemeral/home/industry-partnership-project-brainventures/data/01.data/1.Training/origin

unzip -O cp949 /data/ephemeral/home/industry-partnership-project-brainventures/data/01.data/2.Validation/label/VL1.zip -d /data/ephemeral/home/industry-partnership-project-brainventures/data/01.data/2.Validation/label
unzip -O cp949 /data/ephemeral/home/industry-partnership-project-brainventures/data/01.data/2.Validation/origin/VS1.zip -d /data/ephemeral/home/industry-partnership-project-brainventures/data/01.data/2.Validation

unzip -O cp949 /data/ephemeral/home/industry-partnership-project-brainventures/data/02.raw_data/1.Trainging/[라벨]Training.zip -d /data/ephemeral/home/industry-partnership-project-brainventures/data/02.raw_data
unzip -O cp949 /data/ephemeral/home/industry-partnership-project-brainventures/data/02.raw_data/Validation.zip -d /data/ephemeral/home/industry-partnership-project-brainventures/data/02.raw_data/2.Validation

37317 
# 데이터 압축풀기 
tar xvfz ./download.tar

# ai hub 명령어
aihubshell -help

