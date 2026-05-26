# 핸드폰 연결 및 미러링 세팅

이 문서는 `conda cv` 환경에서 실험 스크립트를 돌리기 전에,  
안드로이드 폰을 USB로 연결하고 `scrcpy`로 미러링한 뒤  
컴퓨터에서 화면을 읽는 전체 흐름을 정리한 문서이다.

## 전체 구조

실험 구조는 아래와 같다.

```text
안드로이드 폰
  -> USB 연결
  -> scrcpy로 컴퓨터에 미러링
  -> Python 스크립트가 scrcpy 창을 화면 캡처
  -> 조건 만족 시 자동 캡처
```

중요한 점:

- Python 코드가 핸드폰에 직접 붙는 방식은 아니다.
- `scrcpy`가 띄운 미러링 창을 컴퓨터에서 읽는 방식이다.
- 따라서 연결이 먼저 되고, 그 다음에 실험 스크립트를 실행해야 한다.

## 1. 필요한 프로그램

필요한 것은 아래 3가지다.

- `adb`
- `scrcpy`
- `conda cv` 환경

다른 팀원이 처음부터 같은 세팅을 만들려면 아래 파일을 쓰면 된다.

- Python/Conda 환경: [environment.yml](/Users/iyeonglag/PycharmProjects/computervision-class/visionproject/environment.yml)
- macOS 도구 설치: [Brewfile](/Users/iyeonglag/PycharmProjects/computervision-class/visionproject/Brewfile)

현재 프로젝트 스크립트는 `cv` 환경을 기준으로 작성되어 있다.

```bash
conda activate cv
```

## 2. macOS 설치

맥에서 설치가 안 되어 있다면 아래 명령으로 설치한다.

```bash
brew install scrcpy
brew install --cask android-platform-tools
```

또는 저장된 Brewfile을 한 번에 사용할 수 있다.

```bash
brew bundle --file visionproject/Brewfile
```

Conda 환경도 아래처럼 한 번에 만들 수 있다.

```bash
conda env create -f visionproject/environment.yml
conda activate cv
```

설치 확인:

```bash
adb version
scrcpy --version
```

## 3. 안드로이드 폰 설정

폰에서 아래를 켜야 한다.

1. `개발자 옵션` 활성화
2. `USB 디버깅` ON

일반적인 개발자 옵션 활성화 방법:

1. `설정`
2. `휴대전화 정보`
3. `빌드 번호`를 여러 번 터치
4. 개발자 옵션이 활성화되면 뒤로 가서 `개발자 옵션` 메뉴 진입
5. `USB 디버깅` 켜기

추가로 아래 옵션도 켜면 실험 분석에 도움이 된다.

- `Show taps` 또는 `터치 표시`

이 옵션을 켜면 녹화 영상에서 사용자가 어디를 눌렀는지 더 잘 보인다.

## 4. USB 연결

1. 핸드폰을 USB 케이블로 컴퓨터에 연결한다.
2. 폰에 `USB 디버깅 허용` 팝업이 뜨면 허용한다.
3. 연결 확인:

```bash
adb devices
```

정상 연결이면 아래처럼 기기 목록이 보인다.

```text
List of devices attached
R3XXXXXXXXXX    device
```

만약 `unauthorized`로 보이면:

- 폰 화면을 다시 보고 허용 팝업을 승인
- 케이블을 다시 연결
- `adb devices`를 다시 실행

## 5. scrcpy 실행

연결이 확인되면 아래처럼 실행한다.

```bash
scrcpy --max-size=1080 --max-fps=30 --window-title HCI_phone
```

설명:

- `--max-size=1080`: 성능과 품질 균형
- `--max-fps=30`: 실험용으로 충분한 프레임
- `--window-title HCI_phone`: 나중에 OBS나 화면 캡처 시 창 식별이 쉬움

정상 실행되면 컴퓨터에 핸드폰 화면이 실시간으로 보인다.

## 6. 우리 실험 스크립트 실행

미러링 창이 뜬 다음에, 별도 터미널에서 아래처럼 실행한다.

```bash
conda activate cv
python visionproject/shot_capture_experiment.py \
  --source screen \
  --screen-region 100,80,420,900 \
  --crop 20,120,380,680
```

의미:

- `--source screen`
  - 웹캠이 아니라 컴퓨터 화면 일부를 읽겠다는 뜻
- `--screen-region`
  - `scrcpy` 창 전체 영역
- `--crop`
  - 그 안에서 실제 카메라 프리뷰 부분만 분석

즉, 구조는 다음과 같다.

```text
scrcpy 창 전체 = screen-region
scrcpy 창 안 카메라 미리보기 부분 = crop
```

## 7. 기본 실행 예시

### 웹캠 테스트

핸드폰 연결 전에 동작만 먼저 확인하고 싶으면:

```bash
conda activate cv
python visionproject/shot_capture_experiment.py --source webcam --webcam-index 0
```

### 핸드폰 미러링 테스트

```bash
scrcpy --max-size=1080 --max-fps=30 --window-title HCI_phone

conda activate cv
python visionproject/shot_capture_experiment.py \
  --source screen \
  --screen-region 100,80,420,900 \
  --crop 20,120,380,680
```

## 8. 현재 구현된 모드

키보드로 모드를 바꿀 수 있다.

- `1`: `manual`
- `2`: `person1`
- `3`: `count3`
- `4`: `ratio`
- `c` 또는 `space`: 즉시 수동 캡처
- `q`: 종료

### person1

기본 모드다. 아래 조건을 만족하면 자동 캡처한다.

- 사람 1명 검출
- 사람 bbox 크기가 적당함
- 화면 중앙 근처에 위치
- 프레임 가장자리에 너무 붙지 않음
- 몇 프레임 연속으로 안정적으로 유지

즉:

```text
"한 명이 제대로 프레임 안에 들어오면 자동으로 찍기"
```

## 9. 좌표 잡는 방법

처음에는 대충 큰 값으로 시작한 뒤 조정하면 된다.

예:

- `screen-region`: scrcpy 창 전체
- `crop`: 카메라 프리뷰만

실무적으로는 다음 순서가 편하다.

1. scrcpy 창을 원하는 위치에 고정
2. 창 크기를 고정
3. 대략적인 `screen-region` 입력
4. `crop`을 줄여 가며 상단 상태바, 하단 버튼 영역을 제외

이유:

- 상태바, 하단 버튼, 텍스트 UI가 많으면 사람 검출이 흔들릴 수 있음
- 카메라 프리뷰만 보게 해야 자동 캡처 조건이 안정적임

## 10. OBS와 같이 쓰는 경우

실험 녹화를 같이 하려면:

1. `scrcpy`로 폰 화면 미러링
2. OBS에서 `Window Capture`로 `HCI_phone` 창 추가
3. 필요하면 웹캠도 추가
4. 필요하면 마이크도 추가

추천 OBS 구성:

- 큰 화면: 핸드폰 미러링 화면
- 작은 화면: 손동작 또는 얼굴 웹캠
- 오디오: 참가자 발화
- 텍스트: 참가자 ID, 조건명

## 11. 자주 생기는 문제

### `adb devices`에 기기가 안 보임

- USB 케이블 불량 가능성
- 충전 전용 케이블일 수 있음
- 폰에서 `USB 디버깅 허용`을 안 눌렀을 수 있음

### `unauthorized`로 뜸

- 폰 화면을 열고 허용 팝업 승인
- 다시 꽂고 재시도

### scrcpy는 뜨는데 자동 캡처가 이상함

- `crop`이 잘못 잡혔을 가능성 큼
- 카메라 프리뷰 영역만 남기도록 `crop` 조정 필요

### 자동 캡처가 너무 안 됨

- `--stable-frames` 낮추기
- `--center-tolerance` 키우기
- `--one-person-ratio-min/max` 범위 넓히기

### 자동 캡처가 너무 쉽게 됨

- `--stable-frames` 올리기
- `--center-tolerance` 줄이기
- `--border-margin-ratio` 키우기

## 12. 최소 체크리스트

실험 전에 아래만 확인하면 된다.

- `conda activate cv`
- `adb devices`에서 기기 확인
- `scrcpy` 창 정상 실행
- 실험 스크립트 실행
- `person1` 자동 캡처 동작 확인
- 저장 폴더 생성 확인

## 관련 파일

- 실험 스크립트: [shot_capture_experiment.py](/Users/iyeonglag/PycharmProjects/computervision-class/visionproject/shot_capture_experiment.py)
- 실험 세팅 문서: [EXPERIMENT_SETUP.md](/Users/iyeonglag/PycharmProjects/computervision-class/visionproject/EXPERIMENT_SETUP.md)
