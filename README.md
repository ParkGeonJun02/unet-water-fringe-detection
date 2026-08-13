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

### 3.1 Functional Requirements

프로젝트의 주요 기능 요구사항은 다음과 같이 정의하였습니다.

| Requirement | Description |
|---|---|
| R1. Aerial Image Processing | GeoTIFF 형식의 RGB 항공영상을 입력으로 처리 |
| R2. Annotation Processing | JSON Polygon Annotation을 Pixel-level Mask로 변환 |
| R3. Binary Segmentation | 수변 관련 영역과 기타 영역을 Binary 형태로 분할 |
| R4. Baseline Comparison | Color + Texture 기반 Heuristic과 U-Net을 동일 조건에서 비교 |
| R5. Region-level Evaluation | 분할 영역의 중첩, 오탐 및 미탐 성능을 정량 평가 |
| R6. Boundary-level Evaluation | 예측 경계와 Annotation 경계 사이의 위치 정확도를 정량 평가 |

이와 같이 모델 학습 자체만을 목표로 하지 않고,
**입력 → 분할 → 비교 → 검증**까지 하나의 평가 가능한 구조로 구성하였습니다.

---

### 3.2 Region-level Evaluation KPI

영역 단위 성능은 JSON Annotation으로 생성한 Reference Mask와
예측 결과의 Pixel-level 비교를 통해 평가하였습니다.

주요 KPI는 다음과 같습니다.

| Metric | Evaluation Purpose | Interpretation |
|---|---|---|
| **IoU** | 예측 영역과 Reference 영역의 중첩 정도 평가 | 높을수록 우수 |
| **Precision** | 수변으로 예측한 영역 중 Reference와 일치하는 비율 | 높을수록 오탐이 적음 |
| **Recall** | Reference 수변 영역 중 실제로 탐지한 비율 | 높을수록 미탐이 적음 |
| **F1-Score** | Precision과 Recall의 균형 평가 | 높을수록 우수 |

#### IoU

IoU는 예측 영역과 Reference 영역이 얼마나 겹치는지를 평가합니다.

```text
                   Intersection
IoU = ---------------------------------------
       Prediction ∪ Ground Truth
```

수변 영역의 전체적인 분할 정확도를 확인하기 위한
대표적인 Region-level KPI로 사용하였습니다.

#### Precision & Recall

Precision은 모델이 수변이라고 판단한 영역에서
잘못 탐지한 영역이 얼마나 적은지를 확인하는 지표입니다.

Recall은 실제 수변 관련 영역 중
모델이 놓치지 않고 탐지한 영역이 얼마나 되는지를 평가합니다.

따라서 두 지표를 함께 확인하여
**False Positive와 False Negative 관점의 성능을 동시에 분석**하였습니다.

#### F1-Score

Precision과 Recall 중 한쪽 성능만 높게 나타나는 경우를 고려하기 위해
두 지표의 균형을 나타내는 F1-Score를 추가로 사용하였습니다.

이를 통해 Heuristic Baseline과 U-Net의
전반적인 Binary Segmentation 성능을 비교하였습니다.

---

### 3.3 Boundary-level Evaluation KPI

수변 분석에서는 영역의 전체적인 중첩 성능뿐만 아니라,
**수변과 육지의 경계가 실제 Annotation 경계와 얼마나 가까운 위치에 존재하는지**
평가하는 것도 중요하다고 판단하였습니다.

따라서 Binary Mask에서 Boundary를 별도로 추출한 뒤
다음 두 가지 지표를 이용하여 경계 정확도를 평가하였습니다.

| Metric | Evaluation Purpose | Interpretation |
|---|---|---|
| **Mean Boundary Distance (MBD)** | 예측 경계와 Reference 경계 사이 평균 거리 측정 | 낮을수록 우수 |
| **Boundary F1** | 일정 Pixel 허용 오차 내에서 두 경계의 일치 정도 평가 | 높을수록 우수 |

---

### 3.4 Mean Boundary Distance (MBD)

Mean Boundary Distance는
예측 Boundary와 Reference Boundary 사이의 평균적인 위치 차이를
Pixel 단위로 측정하는 지표입니다.

본 프로젝트에서는 한쪽 방향의 거리만 계산하지 않고,
두 경계 사이의 거리를 양방향으로 계산하여 평균하는 방식으로
Boundary 위치 오차를 평가하였습니다.

```text
Reference Boundary
        │
        │  Distance
        ▼
Predicted Boundary
```

따라서 MBD 값이 작을수록
예측된 수변 경계가 Reference 경계와 더 가까운 위치에 존재한다는 것을 의미합니다.

---

### 3.5 Boundary F1 @ Pixel Tolerance

항공영상의 Pixel 단위 경계는 완전히 동일한 위치에서 겹치지 않더라도,
일정 범위 안에서 존재하면 실질적으로 유사한 경계로 판단할 수 있습니다.

따라서 다음 Pixel Tolerance 조건에서 Boundary F1을 비교하였습니다.

```text
1 px
5 px
10 px
15 px
```

각 Tolerance 범위 내에 존재하는 예측 Boundary와
Reference Boundary의 일치 여부를 기준으로 Precision과 Recall을 계산하고,
이를 Boundary F1-Score로 평가하였습니다.

이를 통해 단순히 경계가 검출되었는지를 보는 것이 아니라,

**허용 가능한 위치 오차 범위에 따라 U-Net의 경계 검출 성능이
Heuristic Baseline 대비 어떻게 변화하는지 분석**

할 수 있도록 하였습니다.

---

### 3.6 Evaluation Strategy

최종적으로 본 프로젝트의 검증 구조는 다음과 같이 구성하였습니다.

```text
                    Prediction
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
 Region-level Evaluation      Boundary-level Evaluation
          │                           │
          ▼                           ▼
 IoU / Precision / Recall      Boundary Extraction
          │                           │
          ▼                           ▼
         F1                  MBD / Boundary F1
          │                           │
          └─────────────┬─────────────┘
                        ▼
          Heuristic vs U-Net Comparison
```

이를 통해 모델의 성능을 하나의 지표에만 의존하지 않고,

**영역 분할 성능 + 경계 위치 정확도**

두 관점에서 종합적으로 검증하도록 평가 체계를 구성하였습니다.

> **Note**  
> Portfolio의 정량 성능 비교는 JSON Annotation이 존재하는 데이터를 기준으로
> Heuristic Baseline과 U-Net을 동일 조건에서 비교한 결과를 사용합니다.
> 해당 비교 데이터는 학습 데이터에 포함된 Annotation 기반 데이터이므로,
> 독립적인 Test-set Generalization 성능으로 해석하지 않습니다.
