---
layout: post
title: "기계학습, Pre-Processing Data"
description: "데이터 전처리, Pre Processing Data"
categories: [MachineLearning]
tags: [Machine Learning, Supervised Learning, Unsupervised Learning, Pre Processing Data, Standardization, Normalization, Regularization]
use_math: true
redirect_from:
  - /2021/07/08/
---

* Kramdown table of contents
{:toc .toc}

[참고 사이트 : SuanLab](http://suanlab.com/){:target="_ blank"}    

# 데이터 품질

- 잡음 Noise     
측정 과정에서 무작위로 발생하여 측정값의 에러를 발생시키는 것     
<br/>
- 아티펙트 Artifact      
어떠한 요인으로 인해 반복적으로 발생하는 왜곡이나 에러를 의미        
ex) 영상 데이터에서 카메라 렌즈에 묻은 얼룩 등      
<br/>
- 정밀도 Precision       
동일한 대상을 반복적으로 측정하였을 때의 각 결과의 친밀성을 나타내는 것       
측정 결과의 표준편차로 나타낼 수 있음      
<br/>
- 바이어스 Bias
측정 장비에 포함된 시스템적인 변동        
ex) 영점 조절되지 않은 체중계
<br/>
- 정확도 Accuracy
정밀도와 바이어스에 기인한 것이지만, 이를 명시적으로 나타낼 수 있는 수식은 없음    
다만 유효 숫자의 사용에 있어 중요한 측면을 가지고 있음    
<br/>
- 이상치 OutLier
다른 데이터들과 다른 특성을 보이는 등, 유별난 값을 가지는 데이터     
잡음은 임의로 발생하는 예측하기 어려운 요임인에 반해    
이상치는 적법한 하나의 데이터로서 그 자체가 중요한 분석의 목적이 될 수도 있음    
ex) 불법적인 접속 시도 등     
<br/>
- 결측치 Missing Values     
설문조사의 경우 몇몇 사람들이 나이나 몸무게 등 사적인 정보를 공개하기 꺼려하며, 이러한 값은 결측값으로 남게되는 경우가 있음      
<br/>
- 모순, 불일치 Inconsistent values      
동일한 개체에 대한 측정 데이터가 다르게 나타나는 경우    
ex) 주소와 우편번호가 다른 경우    
<br/>
- 중복 Duplicate Data    
중복된 데이터 사이에서도 속성의 차이나 값의 불일치가 발생할 수 있음    
모든 속성및 값이 동일하다면 하나를 삭제하면 되지만,     
그렇지 않은 경우에는 중복 데이터를 합쳐서 하나의 개체로 만들거나     
적법한 속성을 가진 데이터 하나를 선택하는 등의 작업이 필요    


# 데이터 전처리     

> 데이터를 분석 및 처리에 적합한 형태로 만드는 과정을 총칭하는 개념    
> 실무에 사용되는 데이터셋은 바로 분석이 불가능할 정도로 지저분(messy)하다       
> 분석이 가능한 상태로 만들기 위해 아래와 같은 전처리 방식이 자주 사용된다      


## 결측치 처리    

1. 결측치 사례 제거    
2. 평균이나 중앙치로의 대체    
3. 간단한 예측 모델로 대체    

수집된 데이터가 많다면 행을 제거하는 것이 가능하겠지만, 데이터가 충분하지 못하다면 이는 그닥 좋은 선택이 아님    

Missing Value 파악을 위해 `df.info()` 을 가장 처음에 이용하는 것이 일반적    

결측치가 단순히 결측이 되지 않은 값인지 혹은 단순히 없는 0인 값인지를 파악하는 것이 필요    

결측치를 매꿀 때는 `.fillna()`, `.replace()`를 이용

## 이상치 처리

표준점수(평균 0, 표준편차 1)로 변환 후 -3 이하 및 +3 제거

~~~ python
# 표준점수 기반 예제 코드
def std_based_outlier(df):
    for i in range(0, len(df.iloc[1])): 
        df.iloc[:,i] = df.iloc[:,i].replace(0, np.NaN) # optional
        df = df[~(np.abs(df.iloc[:,i] - df.iloc[:,i].mean()) > (3*df.iloc[:,i].std()))].fillna(0)
~~~


IQR 방식

~~~ python
# IQR 기반 예제 코드
def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))
~~~

## 데이터 분포 변환
대부분의 모델은 변수가 특정 분포를 따른다는 가정을 기반으로 함      
예를 들어 선형 모델의 경우 독립변수, 종속변수 모두 정규분포를 가진다는 것을 기반으로 만들어졌기에    
정규분포와 유사할 경우 성능이 높아짐       
이에 log, exp, sqrt 등을 이용해 데이터 분포를 변환    

~~~ python
import math
from sklearn import preprocessing
# 특정 변수에만 함수 적용
df['X_log'] = preprocessing.scale(np.log(df['X']+1)) # 로그
df['X_sqrt'] = preprocessing.scale(np.sqrt(df['X']+1)) # 제곱근

# 데이터 프레임 전체에 함수 적용 (단, 숫자형 변수만 있어야 함)
df_log = df.apply(lambda x: np.log(x+1))     
~~~

## 데이터 단위 변환

데이터의 단위가 다를 경우 분류모델 등에서 부정적인 영향을 미치므로,         
스케일링을 통해 단위를 일정하게 맞추는 작업을 진행해야 함   
대부분의 통계 분석 방법이 정규성 가정을 기반으로 하므로, 완벽하지 않더라도 최대한 정규분포로 변환하는 노력이 필요함

- Scaling : 평균이 0, 분산이 1인 분포로 변환     
    - $ x_ {new\_i} = \frac{x_ i - mean(x)}{std(x)} $        
<br/>

- MinMax Scaling : 특정 범위(0~1)로 모든 데이터를 변환     
    - $ x_ {new\_i} = \frac{x_ i - min(x)}{max(x) - min(x)} $   
<br/>

- Box-Cox : 여러 k 값중 가장 작은 SSR(Residual) 선택      
    - $
y_ i ^ {(\lambda)} = 
\begin{cases}
 \frac{y_ i ^ {\lambda} - 1}{\lambda} & \text{ if } \lambda \neq 0  \\\ 
 ln(y_ i)& \text{ if }  \lambda = 0
\end{cases}
$
 
- Robust_scale: median, interquartile range 사용(outlier 영향 최소화)
     

~~~ python
from scipy.stats import boxcox

# 변수별 scaling 적용
df['X_scale'] = preprocessing.scale(df['X']) 
df['X_minmax_scale'] = preprocessing.MinMaxScaler(df['X']
df['X_boxcox'] = preprocessing.scale(boxcox(df['X']+1)[0])
df['X_robust_scale'] = preprocessing.robust_scale(df['X'])

# 데이터 프레임 전체에 scaling 적용
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# StandardScaler
for c in df:
    df_sc[c] = StandardScaler().fit_transform(df[c].reshape(-1,1)).round(4)

# MinMaxScaler
for c in df:
    df_minmax[c] = MinMaxScaler().fit_transform(df[c].reshape(-1,1).round(4))
~~~


[pipeline](https://hhhh88.tistory.com/6)

# 표준화, Standardization

> 값의 범위(scale)를 평균 0, 분산 1이 되도록 변환        
> 머신러닝에서 scale이 큰 feature의 영향이 비대해지는 것을 방지
> 딥러닝에서 Local Minima에 빠질 위험 감소(학습 속도 향상)
> 정규분포를 표준정규분포로 변환하는 것과 같음

## 표준 점수, Z-score    

$ z_ i = \frac{x_ i - mean(x)}{std(x)} $        

> 데이터를 표준 정규 분포(가우시안 분포)에 해당하도록 값을 바꿔주는 과정           
> -1 ~ 1 사이에 68%가 있고, -2 ~ 2 사이에 95%가 있고, -3 ~ 3 사이에 99%가 있음
> -3 ~ 3의 범위를 벗어나면 outlier일 확률이 높음



# 일반화(정규화), Normalization
               
> 머신러닝 모델은 데이터가 가진 feature(특징)을 뽑아서 학습하는데, 이때 모델이 받아들이는 데이터의 크기가 들쑥날쑥하다면              
> 모델이 데이터를 이상하게 해석할 우려가 있음              
> ex) 아파트 가격에서 연식, 가격, 방갯수를 feature로 받았을 때 각각이 2000, 2억, 5 라고 한다면, 2억만 중요하게 여길 수 있음               
> 
> 즉 일반화란, 데이터의 범위(단위)를 일정하게(0 ~ 1) 만들어 모든 데이터가 같은 정도의 스케일(중요도)로 반영되도록 해주는 것              
> min-max의 편차가 크거나 다른 열에 비해 데이터가 지나치게 큰 열에 사용    


## Min-Max Normalization (최소-최대 정규화)

$ $ x_ {new\_i} = \frac{x_ i - min(x)}{max(x) - min(x)} $   $

모든 데이터 중에서 가장 작은 값을 0, 가장 큰 값을 1로 두고, 나머지 값들은 비율을 맞춰서 모두 0과 1 사이의 값으로 스케일링해주는 것             
그러나 이상치(outlier)에 대해 취약하다는 단점이 있음                 


# 정규화(규제), Regularization


