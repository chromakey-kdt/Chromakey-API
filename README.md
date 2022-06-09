# :bulb: Chromakey 모자이크 서비스 API
<br>
안녕하세요.<br>

누구나 쉽게 이용 가능한 인공지능 기반 **모자이크 자동화 서비스 OPEN API** 입니다.**:v:**<br>
:facepunch: 아래에 사용 설명서를 환경 별로 순서대로 기입하였으니 차근차근 진행해주세요. :point_down:

<br>
<br>


## 🚗 How to run Ubuntu

*우분투 & AWS or * 우분투 개인서버 -> 도커를 이용한 배포
1. 도커 관련 설치
```
  sudo apt-get update
  sudo apt install docker.io 
  sudo systemctl start docker
  sudo systemctl enable docker
```
2. 도커 허브(https://hub.docker.com/)에서 아나콘다 최신 이미지를 다운받아주세요.

```
  docker pull continuumio/anaconda3:latest
```


3. 로컬 환경에서 /chroma 폴더를 만들고, Chromakey-API repository를 clone해주세요.
```
  mkdir /chroma
  cd /chroma
  git clone https://github.com/chromakey-kdt/Chromakey-API.git
```

4. 사전 학습할 가중치가 담긴 모델 파일을 다운을 받습니다. 
```
  # 가중치를 담을 폴더 생성
  mkdir /chroma/Chromakey-API/pm
  cd /chroma/Chromakey-API/pm

  # wget 명령어로 가중치 파일을 다운 받습니다.
  wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TPaWtbRHe3dU27D7vUwCNN3F30KHofWs' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1TPaWtbRHe3dU27D7vUwCNN3F30KHofWs" -O pretrained_model.pth && rm -rf ~/cookies.txt
  
  # 현재 경로로 컴백
  cd ..
```

5. 격리된 컨테이너 공간을 생성합니다.
```
  # gpu가 없을때
  docker run -it --name chromaAPI -v /chroma/:/chroma -p 5000:5000 continuumio/anaconda3:latest

  # gpu가 있을때
  docker run -it --gpus "device=0" --name chromaAPI -v /chroma/:/chroma -p 5000:5000 continuumio/anaconda3:latest

  ## 도커 환경에 진입했다면, 파이썬 3.8 버전으로 가상환경 생성.
  conda create -n chroma python=3.8
  conda activate chroma

  ## 도커 환경에 진입하지 못했다면
  docker start chromaAPI
  docker exec -it chromaAPI bash

  ## 파이썬 3.8 버전으로 가상환경 생성.
  conda create -n chroma python=3.8
  conda activate chroma
```

6. 생성된 가상환경 내에서 필요한 패키지 설치해주세요.
```
  cd /chroma/Chromakey-API
  apt-get update
  
  # 남은 설치 파일은 requirements를 이용하여 설치한다.
  pip install -r requirements.txt
  
  # 파이토치 설치
  pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

  # opencv 설치
  sudo apt-get install libgl1-mesa-glx
  pip install opencv-contrib-python

```
7. 프로젝트 실행 전 필요한 환경 세팅(.env 파일)을 다음과 같이 준비합니다.
```
  < .env 파일 >
  # 해당 경로를 절대 경로로 수정합니다.
  TEMP_DATA_PATH = "/chroma/Chromakey-API/tmp/" 
```

8. Chroma API를 실행시켜주세요.
```
  python -m photovleml
```

9. 웹 사이트에 아래 링크를 입력해주세요.
```
  http://127.0.0.1:5000/ or http://외부IP:5000/ or http://도메인:5000/
```

10. 만약 AWS를 사용해 실행시키는 경우. 보안 그룹을 다음과 같이 설정해야 접속이 가능하다.
<br>
__주의사항__ 
    - 8000번이 아니라 __5000__ 번 입니다!

![aws 보안규칙](https://user-images.githubusercontent.com/46054315/152639107-711432db-85b5-4cd5-9746-0eaf73646740.PNG)
---
<br>
<br>

## 🚗 How to run Window

* 윈도우 -> 아나콘다
1. 아나콘다 프롬프트를 실행시켜 python 3.6.4의 가상환경을 만든다.
```
  conda create -n chroma python=3.8
```
2. vscode를 실행 시켜서 git clone을 진행한다.
```
  git clone https://github.com/chromakey-kdt/Chromakey-API.git
```
3. 아까 설치한 chroma 가상환경으로 설정 후, 필요한 패키지를 설치한다.
```
  # vs code 창에서 command Propt를 클릭하여 아래 명령어를 입력해 패키지를 설치한다.
  pip install -r requirements.txt
  
  # 파이토치 설치
  pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
  
```
4. 프로젝트 실행 전 필요한 환경 세팅(.env 파일)을 다음과 같이 준비합니다.
```
  < .env 파일 >
  # 해당 경로를 절대 경로로 수정합니다.
  TEMP_DATA_PATH = "/프로젝트 절대 경로/tmp/" 
```
5. 사전 학습할 가중치가 담긴 모델 파일을 다운을 받습니다. 
```
  # 가중치를 담을 폴더 생성
  mkdir Chromakey-API/pm
  cd Chromakey-API/pm

  # wget 명령어로 가중치 파일을 다운 받습니다.
  wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TPaWtbRHe3dU27D7vUwCNN3F30KHofWs' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1TPaWtbRHe3dU27D7vUwCNN3F30KHofWs" -O pretrained_model.pth && rm -rf ~/cookies.txt
  
  # 현재 경로로 컴백
  cd ..
```
6. 프로젝트 실행에 필요한 테이블을 생성해주세요.
```
  python -m photovleml
```

7. 웹 사이트에 아래 링크를 입력해주세요.
```
  http://127.0.0.1:5000/ or http://외부IP:5000/ or http://도메인:5000/
```


---

## ⚙ Environment

Backend

```
    - Flask version : 2.1.2
    - pytorch version : 1.7.1
```


<br>

## ⚡ tech-stack

### backend

- Flask
- pytorch

<br>


## 🌞 Contributors

- 라효진 👉 [ratataca](https://github.com/ratataca)
- 김종원 👉 [JONWON2](https://github.com/JONWON2)
- 이대용 👉 [baekgom78](https://github.com/baekgom78)
- 최승훈 👉 [owvwo](https://github.com/owvwo)
- 김윤중 👉 [yoonjoong](https://github.com/yoonjoong)
- 이지민 👉 [Jiminn](https://github.com/Jiminn)

<br>

## 📅 Development period

2022.05.20 ~ 2022.06.09 (3Week)
