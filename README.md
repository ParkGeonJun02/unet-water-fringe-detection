# U-Net 기반 항공영상 수변 영역 분할 및 경계 분석

> 고해상도 항공영상에서 수변 관련 영역을 탐지하기 위해  
> Color + Texture 기반 Heuristic Baseline과 U-Net을 구성하고,  
> 영역 분할 성능뿐만 아니라 경계 위치 오차까지 정량적으로 비교·검증한 프로젝트입니다.

---

## 1. Project Overview

### 프로젝트 목표

고해상도 항공영상으로부터 수변 관련 영역을 탐지하고,
수역과 주변 지형의 경계를 영상 기반으로 분석하는 것을 목표로 하였습니다.

단순히 U-Net 모델을 학습하는 것에 그치지 않고,

**Heuristic Baseline → U-Net 기반 분할 → 후처리 → 정량 비교 → 경계 오차 분석**

순서로 성능을 검증하였습니다.

### 주요 입력 및 출력

**Input**
- RGB 고해상도 항공영상 (GeoTIFF)
- JSON Polygon Annotation

**Output**
- Water-related Binary Mask
- Water Probability Map
- Water / Forest / Sand-Road Boundary Visualization
- Heuristic vs U-Net 정량 비교 결과

### Development Flow

<img src="Input_data_enviroment_insert_image.png" width="400">


---

## 2. Environment & Constraints

본 프로젝트는 고해상도 항공영상을 이용하여 수변 관련 영역을 분할하고,
수변과 주변 지형의 경계를 분석하는 환경에서 수행하였습니다.

단순히 U-Net 모델을 학습하는 것에 그치지 않고,
입력 데이터 형식, Annotation 구조, 학습용 Mask 생성 방식,
학습 조건 및 성능 검증 조건을 함께 고려하여 개발 환경을 구성하였습니다.

---

### 2.1 Input Data Environment

본 프로젝트는 고해상도 RGB 항공영상을 입력으로 사용하여
수변 관련 영역을 분석하였습니다.

- **Input Image**: RGB 고해상도 항공영상 (GeoTIFF)
- **Annotation**: JSON Polygon Annotation
- **Model Input Size**: 512 × 512
- **Segmentation Type**: Binary Segmentation
- **Positive Class**: 프로젝트에서 정의한 수변 관련 영역
- **Negative Class**: 기타 영역

원본 항공영상과 JSON Polygon Annotation을 이용하여
U-Net 학습에 사용할 Binary Mask를 구성하였습니다.

<p align="center">
  <img src="Input_data_enviroment_insert_image.png" width="400">
</p>

---

### 2.2 Annotation Processing

JSON Annotation에 포함된 Polygon 좌표를 영상 좌표계에 맞게 변환한 뒤,
Raster Mask 형태로 변환하여 학습 데이터 구성에 사용하였습니다.

프로젝트 코드에서는 다음 Annotation Code를
수변 관련 Positive Class 구성에 사용하였습니다.

```text
20 / 40 / 50 / 511
```

각 Polygon 영역은 Binary Mask에서 다음과 같이 처리됩니다.

```text
Water-related region  → 1
Other region          → 0
```

Annotation Code의 공식 세부 의미를 임의로 재정의하지 않고,
프로젝트에서 실제 사용한 Code 조건을 그대로 유지하였습니다.

---

### 2.3 Training Mask Construction

학습용 Binary Mask는 JSON Annotation 기반 Mask와 함께,
항공영상의 RGB 색상 및 Local Texture 조건을 이용한 후보 영역을
보완적으로 결합하도록 구성하였습니다.

<p align="center">
  <img src="Training_Mask_Construction_inserted_image.png" width="400">
</p>

Color + Texture 기반 후보 영역은 다음 영상 특성을 이용합니다.

- RGB 채널 값
- Green / Red 채널 간 관계
- Local Standard Deviation
- 저질감 영역 여부

이를 통해 Annotation으로 정의된 영역과
영상 자체의 색상·질감 특성을 함께 반영하여
U-Net 학습용 Binary Mask를 생성하였습니다.

---

### 2.4 Model & Training Environment

U-Net은 512 × 512 크기의 RGB 영상을 입력으로 받아
픽셀 단위 Water-related Probability Map을 출력하도록 구성하였습니다.

주요 학습 조건은 다음과 같습니다.

| Item | Setting |
|---|---|
| Input Size | 512 × 512 |
| Input Channel | RGB, 3 channels |
| Output Channel | 1 |
| Batch Size | 2 |
| Epochs | 30 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss Function | BCEWithLogitsLoss |
| Positive Class Weight | 3.0 |
| Output Threshold | 0.5 |

수변 관련 Positive Class의 불균형을 고려하여
`BCEWithLogitsLoss`에 Positive Class Weight `3.0`을 적용하였습니다.

학습 과정에서는 Validation Loss를 기준으로 성능을 확인하고,
가장 낮은 Validation Loss를 기록한 모델을

`best_unet_model.pth`

로 저장하도록 구성하였습니다.

---

### 2.5 Development Constraints

본 프로젝트에서는 다음 조건을 고려하여 개발 및 검증을 수행하였습니다.

1. 고해상도 항공영상을 U-Net 입력 크기인 **512 × 512**로 변환해야 함
2. Polygon 형태의 JSON Annotation을 **Pixel-level Binary Mask**로 변환해야 함
3. 수변 영역과 주변 지형을 단순 RGB 조건만으로 완전히 구분하기 어려움
4. 영역 분할 성능뿐만 아니라 **실제 수변 경계 위치의 정확도**도 평가할 필요가 있음
5. U-Net의 성능을 판단하기 위해 기존 **Color + Texture Heuristic Baseline과 동일 조건 비교**가 필요함

따라서 프로젝트의 전체 개발·검증 과정은 다음과 같이 구성하였습니다.

<p align="center">
  <img src="Development_Constraints_inserted_image.png" width="400">
</p>

이를 통해 단순히 모델의 출력 영상을 확인하는 것이 아니라,
**Baseline 대비 성능 개선 여부와 수변 경계 위치의 정확성까지 정량적으로 검증**
할 수 있도록 구성하였습니다.



---

## 3. Requirements & Evaluation KPI

본 프로젝트에서는 모델의 성능을 단순히 결과 영상의 시각적 품질만으로 판단하지 않고,
**영역 분할 정확도와 경계 위치 정확도**를 각각 정량적으로 평가하도록 검증 기준을 구성하였습니다.

즉, 다음 두 가지 질문을 중심으로 성능을 확인하였습니다.

> **1. 수변 관련 영역을 얼마나 정확하게 분할하는가?**  
> **2. 실제 Annotation 경계에 얼마나 가까운 위치에서 경계를 검출하는가?**

이를 위해 Region-level Evaluation과 Boundary-level Evaluation을 분리하여 수행하였습니다.

---

## 3. Requirements & Initial Approach

### 3.1 Requirements & KPI

본 프로젝트에서는 단순히 수변 영역이 시각적으로 잘 검출되는지를 확인하는 것이 아니라,
**영역 분할 정확도와 경계 위치 정확도를 함께 검증하는 것**을 주요 요구사항으로 설정하였습니다.

주요 개발 요구사항은 다음과 같습니다.

- GeoTIFF 항공영상과 JSON Polygon Annotation 처리
- 수변 관련 영역의 Pixel-level Binary Segmentation
- 초기 Baseline과 U-Net을 동일 조건에서 비교
- 영역 분할 성능과 수변 경계 정확도를 각각 정량 검증

성능 평가는 다음 두 관점으로 구성하였습니다.

| Evaluation | KPI |
|---|---|
| **Region-level** | IoU, Precision, Recall, F1-Score |
| **Boundary-level** | Mean Boundary Distance (MBD), Boundary F1 |

이를 통해 **“영역을 얼마나 정확하게 분할했는가”와
“수변 경계를 얼마나 정확한 위치에서 검출했는가”**를 분리하여 평가하였습니다.

---

### 3.2 Initial Approach — From Color to Color + Texture

초기에는 물 영역이 주변 지형과 색상 차이를 보일 것이라고 판단하여,
**RGB 색상 조건만으로 수변 영역을 구분할 수 있을 것이라고 예상**하였습니다.

하지만 실제 항공영상을 분석하면서 다음 문제를 확인하였습니다.

- 숲 영역이 어두운 녹색 계열로 나타남
- 일부 수역 역시 짙은 녹색 계열로 관측됨
- 따라서 색상 정보만으로 물과 숲을 안정적으로 구분하기 어려움

이 문제를 해결하기 위해 색상 이외의 영상 특성을 추가로 고려하였습니다.

항공영상에서 숲은 나무와 식생으로 인해 상대적으로 **거친 Texture**를 나타내는 반면,
수면은 상대적으로 **매끄러운 Texture**를 나타낸다는 점에 착안하였습니다.

따라서 초기 Color 기반 접근을 다음과 같이 확장하였습니다.

<p align="center">
  <img src="Initial_Approach_From_Color_to_Color_and_Texture_insert_image.png" width="400">
</p>

최종 Heuristic Baseline에서는 다음 정보를 함께 사용하였습니다.

- RGB Channel 조건
- Green / Red Channel 관계
- Local Standard Deviation
- Morphological Processing
- Connected Component 분석

---

### 3.3 Baseline as a Comparison Reference

Color + Texture 방식은 최종 모델이 아니라,
이후 학습 기반 방법의 성능 개선 여부를 확인하기 위한 **Baseline**으로 활용하였습니다.

동일한 Annotation 조건에서 Heuristic과 U-Net을 비교하여,

> **학습 기반 접근이 규칙 기반 접근보다 실제로
> 영역 분할과 경계 검출 성능을 개선하는가?**

를 정량적으로 확인하는 방향으로 프로젝트를 진행하였습니다.
