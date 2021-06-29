---
layout: post
title: "Kaggle, House Price Regression"
description: "Regression, 회귀분석"
categories: [Machine Learning]
tags: [Machine Learning, Supervised Learning, Kaggle, Python, Regression, CV]
use_math: true
redirect_from:
  - /2021/06/25/
---

* Kramdown table of contents
{:toc .toc}

[Competition : House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques){: target="_blank"}   
[House Prices, 참고 사이트](https://www.kaggle.com/kongnyooong/house-price-tutorial-for-korean-beginners){: target="_blank"}   
[House Prices : 내가 작성한 코드](https://www.kaggle.com/s1hyeon/house-price-regression/edit){: target="_blank"}   

# 1. 데이터 체크 
## 변수간 관계 파악    
### 수치형 변수     
Heat Map, Scatter Plot 등으로 변수들 간에 관계 파악     
1. Heat Map    
2. Zoomed Heat Map    
3. Pair Plot    
4. Scatter Plot    
~~~ python
corr_data = df_train[['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SalePrice']] 

*** 1. Heat Map ***
# corr_data에 저장된 변수들간에 관계를 파악하기 위함
colormap = plt.cm.PuBu 
sns.set(font_scale=1.0) 
f , ax = plt.subplots(figsize = (14,12)) 
plt.title('Correlation of Numeric Features with Sale Price',y=1,size=18) 
sns.heatmap(corr_data.corr(),square = True, linewidths = 0.1, cmap = colormap, linecolor = "white", vmax=0.8)

~~~
![Heat Map]()    
~~~ python
*** 2. Zoomed Heat Map ***
# 전체적으로 살펴본 것중, 연관이 강한것 위주로 다시 출력하여 확인
k = 11
cols = corr_data.corr().nlargest(k, 'SalePrice')['SalePrice'].index
print(cols)
cm = np.corrcoef(df_train[cols].values.T)
f, ax = plt.subplots(figsize = (12,10))
sns.heatmap(cm, vmax=.8, linewidths=0.1,square=True,annot=True,cmap=colormap, linecolor="white",xticklabels = cols.values ,annot_kws = {'size':14},yticklabels = cols.values)
~~~
![Zoomed Heat Map]()    
~~~ python
*** 3. Pair Plot ***
# 그중에서도 연관이 강한것들로 산점도를 그려봄
sns.set() 
columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageCars','FullBath','YearBuilt','YearRemodAdd'] 
sns.pairplot(df_train[columns],size = 2 ,kind ='scatter',diag_kind='kde') 
plt.show()
~~~
![Pair Plot]()    
~~~ python
*** 4. Scatter Plot & Line Plot ***
# 위 산점도를 바탕으로 Scatter Plot 과 line Plot(선형회귀 적합선)을 함께 출력하여 확인

fig, ((ax1, ax2), (ax3, ax4),(ax5,ax6), (ax7,ax8)) = plt.subplots(nrows=4, ncols=2, figsize=(16,13)) 

# 각각의 x에 대한 산점도를 그려줌
OverallQual_scatter_plot = pd.concat([df_train['SalePrice'],df_train['OverallQual']],axis = 1) 
sns.regplot(x='OverallQual',y = 'SalePrice',data = OverallQual_scatter_plot,scatter= True, fit_reg=True, ax=ax1) 

TotalBsmtSF_scatter_plot = pd.concat([df_train['SalePrice'],df_train['TotalBsmtSF']],axis = 1) 
sns.regplot(x='TotalBsmtSF',y = 'SalePrice',data = TotalBsmtSF_scatter_plot,scatter= True, fit_reg=True, ax=ax2) 

GrLivArea_scatter_plot = pd.concat([df_train['SalePrice'],df_train['GrLivArea']],axis = 1) 
sns.regplot(x='GrLivArea',y = 'SalePrice',data = GrLivArea_scatter_plot,scatter= True, fit_reg=True, ax=ax3) 

GarageCars_scatter_plot = pd.concat([df_train['SalePrice'],df_train['GarageCars']],axis = 1) 
sns.regplot(x='GarageCars',y = 'SalePrice',data = GarageCars_scatter_plot,scatter= True, fit_reg=True, ax=ax4) 

FullBath_scatter_plot = pd.concat([df_train['SalePrice'],df_train['FullBath']],axis = 1) 
sns.regplot(x='FullBath',y = 'SalePrice',data = FullBath_scatter_plot,scatter= True, fit_reg=True, ax=ax5) 

YearBuilt_scatter_plot = pd.concat([df_train['SalePrice'],df_train['YearBuilt']],axis = 1) 
sns.regplot(x='YearBuilt',y = 'SalePrice',data = YearBuilt_scatter_plot,scatter= True, fit_reg=True, ax=ax6) 

YearRemodAdd_scatter_plot = pd.concat([df_train['SalePrice'],df_train['YearRemodAdd']],axis = 1)
sns.regplot(x='YearRemodAdd', y='SalePrice', data=YearRemodAdd_scatter_plot, scatter=True, fit_reg=True, ax=ax7)
YearRemodAdd_scatter_plot.plot.scatter('YearRemodAdd','SalePrice')
~~~
![Scatter Plot]()    

### 범주형 변수    
Box Plot등으로 변수들 간에 관계 파악     
~~~ python    
*** Box Plot ***
# 범주형 변수를 박스플롯으로 확인
# Boxplot의 범주마다 보여지는 SalePrice 편차 정도에 따라 영향의 크기가 나눠진다고 볼 수 있음
li_cat_feats = list(categorical_feats)
nr_rows = 15
nr_cols = 3

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4, nr_rows*3))

for r in range(0, nr_rows):
    for c in range(0, nr_cols):
        i = r*nr_cols+c
        if i < len(li_cat_feats):
            sns.boxplot(x=li_cat_feats[i], y=df_train["SalePrice"], data=df_train, ax = axs[r][c])

plt.tight_layout()
plt.show()
~~~
![Box Plot]()    
# 2. 데이터 분류 및     
~~~ python    
# 수치형 변수중 연관이 강한것과 약한것 분류
num_strong_corr = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageCars', 'FullBath','YearBuilt','YearRemodAdd'] 
num_weak_corr = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF','OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'] 

# 범주형 변수중 연관이 강한것과 약한것 분류
catg_strong_corr = ['MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType'] 
catg_weak_corr = ['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterCond', 'Foundation', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleCondition' ]
~~~

## 수치형 변수
### 정규근사화
회귀분석을 하기 위해선 잔차의 분포가 정규성을 만족해야함
따라서 비대칭과 첨도가 관찰된 SalePrice을 하기 위해선 정규근사화 해야함
-> 값에 로그를 취하여 정규근사화
~~~ python    
df_train["SalePrice_Log"] = df_train["SalePrice"].map(lambda i:np.log(i) if i>0 else 0) 

f, ax = plt.subplots(1, 1, figsize = (10,6)) 
g = sns.distplot(df_train["SalePrice_Log"], color = "b", label="Skewness: {:2f}".format(df_train["SalePrice_Log"].skew()), ax=ax) 
g = g.legend(loc = "best") 
print("Skewness: %f" % df_train['SalePrice_Log'].skew()) 
print("Kurtosis: %f" % df_train['SalePrice_Log'].kurt()) 

# 로그로 변환한 이후에는 기존 값을 사용하지 않기에 삭제
df_train.drop('SalePrice', axis= 1, inplace=True)
~~~
![SalePrice]()
![SalePriceLog]()

### 이상치 제거    
**IQR** $ IQR = Q3 - Q1 $  Q3:상위 25%, Q1:하위25%    
상위25 ~ 하위25 를 기준으로 이보다 더 밖에 있는 값들은 이상치라고 판단할 수 있음    

~~~ python
def detect_outliers(df, n, features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        
        outlier_step = 1.5 * IQR
        
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] >  Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    return multiple_outliers

# 행번호 list return
Outliers_to_drop = detect_outliers(df_train, 2, ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'])

# 발견된 이상치 확인
df_train.loc[Outliers_to_drop]
# 이상치 삭제 후 인덱스 재정렬
df_train = df_train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True) 
~~~    

### 결측치 처리
1. 관측이 안된게 아니라 값이 없는 경우 : 값이 없다는 의미로 None으로 변경    
~~~ python    
cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu', 'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical', 'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st', 'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2', 'MSZoning', 'Utilities'] 
for col in cols_fillna: 
    df_train[col].fillna('None',inplace=True) 
    df_test[col].fillna('None',inplace=True)
~~~    
2. 실제로 결측치인 경우 : 수치형 변수들이므로 평균으로 대체    
~~~ python    
# 결측 데이터 확인
total = df_train.isnull().sum().sort_values(ascending=False) 
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False) 
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) 
# 결측 데이터 평균으로 
df_train.fillna(df_train.mean(), inplace=True) 
df_test.fillna(df_test.mean(), inplace=True)

~~~    

### 유의하지 않은 변수 삭제
~~~ python
id_test = df_test['Id'] 
to_drop_num = num_weak_corr 
to_drop_catg = catg_weak_corr 
# 삭제시켜야 할 모든 카테고리를 리스트로 저장
cols_to_drop = ['Id'] + to_drop_num + to_drop_catg 

for df in [df_train, df_test]: 
    df.drop(cols_to_drop, inplace= True, axis = 1)
~~~

## 범주형 변수
### 수치형 변수로 변경   
1. SalePrice의 평균으로 각 카테고리를 변경. ex) WD = 12.11 ...
2. 변경된 카테고리가 유사한 값들 끼리 같은 그룹으로 묶음
~~~ python    
# Sale Price의 평균으로 변경
for catg in catg_list:
    g = df_train.groupby(catg)["SalePrice_Log"].mean() 
    print(g)
    
# 위의 수치들을 참고하여 각 변수들을 그룹화
# 값이 유사하면 같은 그룹, 동일한 수준으로 봐도 무방
# 'MSZoning' 
msz_catg2 = ['RM', 'RH'] 
msz_catg3 = ['RL', 'FV'] 

# Neighborhood 
nbhd_catg2 = ['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert', 'NWAmes', 'Somerst', 'Timber', 'Veenker'] 
nbhd_catg3 = ['NoRidge', 'NridgHt', 'StoneBr']

# Condition2 
cond2_catg2 = ['Norm', 'RRAe'] 
cond2_catg3 = ['PosA', 'PosN'] 

# SaleType 
SlTy_catg1 = ['Oth'] 
SlTy_catg3 = ['CWD'] 
SlTy_catg4 = ['New', 'Con']

# 위에서 그룹화한걸 바탕으로 수치형으로 변환

for df in [df_train, df_test]: 
    
    df['MSZ_num'] = 1 
    df.loc[(df['MSZoning'].isin(msz_catg2) ), 'MSZ_num'] = 2 
    df.loc[(df['MSZoning'].isin(msz_catg3) ), 'MSZ_num'] = 3 
    
    df['NbHd_num'] = 1 
    df.loc[(df['Neighborhood'].isin(nbhd_catg2) ), 'NbHd_num'] = 2 
    df.loc[(df['Neighborhood'].isin(nbhd_catg3) ), 'NbHd_num'] = 3 
    
    df['Cond2_num'] = 1 
    df.loc[(df['Condition2'].isin(cond2_catg2) ), 'Cond2_num'] = 2 
    df.loc[(df['Condition2'].isin(cond2_catg3) ), 'Cond2_num'] = 3 
    
    df['Mas_num'] = 1 
    df.loc[(df['MasVnrType'] == 'Stone' ), 'Mas_num'] = 2 
    
    df['ExtQ_num'] = 1 
    df.loc[(df['ExterQual'] == 'TA' ), 'ExtQ_num'] = 2 
    df.loc[(df['ExterQual'] == 'Gd' ), 'ExtQ_num'] = 3 
    df.loc[(df['ExterQual'] == 'Ex' ), 'ExtQ_num'] = 4 
    
    df['BsQ_num'] = 1 
    df.loc[(df['BsmtQual'] == 'Gd' ), 'BsQ_num'] = 2 
    df.loc[(df['BsmtQual'] == 'Ex' ), 'BsQ_num'] = 3 
    
    df['CA_num'] = 0 
    df.loc[(df['CentralAir'] == 'Y' ), 'CA_num'] = 1 
    
    df['Elc_num'] = 1 
    df.loc[(df['Electrical'] == 'SBrkr' ), 'Elc_num'] = 2 
    
    df['KiQ_num'] = 1 
    df.loc[(df['KitchenQual'] == 'TA' ), 'KiQ_num'] = 2 
    df.loc[(df['KitchenQual'] == 'Gd' ), 'KiQ_num'] = 3 
    df.loc[(df['KitchenQual'] == 'Ex' ), 'KiQ_num'] = 4 
    
    df['SlTy_num'] = 2 
    df.loc[(df['SaleType'].isin(SlTy_catg1) ), 'SlTy_num'] = 1 
    df.loc[(df['SaleType'].isin(SlTy_catg3) ), 'SlTy_num'] = 3 
    df.loc[(df['SaleType'].isin(SlTy_catg4) ), 'SlTy_num'] = 4
~~~

### 변경된 변수들의 연관관계 파악 후 삭제

~~~ python
# 수치형으로 바꿧으니 이제 HeatMap으로 확인 가능
new_col_HM = df_train[['SalePrice_Log', 'MSZ_num', 'NbHd_num', 'Cond2_num', 'Mas_num', 'ExtQ_num', 'BsQ_num', 'CA_num', 'Elc_num', 'KiQ_num', 'SlTy_num']] 
colormap = plt.cm.PuBu 
plt.figure(figsize=(10, 8)) 
plt.title("Correlation of New Features", y = 1.05, size = 15) 
sns.heatmap(new_col_HM.corr(), linewidths = 0.1, vmax = 1.0, square = True, cmap = colormap, linecolor = "white", annot = True, annot_kws = {"size" : 12})

~~~
![Heat Map]()

~~~ python    
# 범주형 변수로 박스플롯등으로 그려서 확인했을 때에는 상관이 있는것 처럼 보였지만
# 수치형 변수들로 바꿔 HeatMap으로 확인해보니 
# 상관이 없어 보이는 값들이 보임
# 수치형 변수로 바꿧으니 이제 필요없는 범주형 변수들과
# 연관이 약한 변수들 삭제

df_train.drop(['MSZoning','Neighborhood' ,'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType', 
               'Cond2_num', 'Mas_num', 'CA_num', 'Elc_num', 'SlTy_num'], axis = 1, inplace = True) 
df_test.drop(['MSZoning', 'Neighborhood' , 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType', 
              'Cond2_num', 'Mas_num', 'CA_num', 'Elc_num', 'SlTy_num'], axis = 1, inplace = True)

~~~

# 3. 학습    
~~~ python     
from sklearn.model_selection import train_test_split
from sklearn import metrics 

# 학습 모델 생성
# Validation
# 기존 데이터 학습 데이터 셋의 가격을 지우고 맞춰봄
X_train = df_train.drop("SalePrice_Log", axis = 1).values 
target_label = df_train["SalePrice_Log"].values 
X_test = df_test.values 
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size = 0.2, random_state = 2000)

# 선형 회귀 분석 실시
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_tr, y_tr)

~~~ 

~~~ python    
# Train 데이터셋으로 학습
y_hat = regressor.predict(X_tr) 

plt.scatter(y_tr, y_hat, alpha = 0.2) 
plt.xlabel('Targets (y_tr)',size=18) 
plt.ylabel('Predictions (y_hat)',size=18) 
plt.show()

regressor.score(X_tr,y_tr)
~~~
![pridict]()

~~~ python    
# 같은 방법으로 Train 데이터셋으로 따로 빼낸 validation도 확인

y_hat_test = regressor.predict(X_vld) 
plt.scatter(y_vld, y_hat_test, alpha=0.2) 
plt.xlabel('Targets (y_vld)',size=18) 
plt.ylabel('Predictions (y_hat_test)',size=18) 
plt.show()
regressor.score(X_vld,y_vld)
~~~
![pridict]()

~~~ python    
# K-fold validaion 수행

from sklearn.model_selection import cross_val_score 

accuracies = cross_val_score(estimator = regressor, X = X_tr, y = y_tr, cv = 10)

# 학습 정확도 확인
print(accuracies.mean()) # 0.8203834733226796
print(accuracies.std())  # 0.1182535686269993
~~~

~~~ python    
# 학습시킨 회귀모델에 테스트 값을 넣어서 결과 출력
pred_xgb = regressor.predict(X_test) 

sub_xgb = pd.DataFrame() 
sub_xgb['Id'] = id_test 
sub_xgb['SalePrice'] = pred_xgb 
sub_xgb['SalePrice'] = np.exp(sub_xgb['SalePrice'])     
sub_xgb.to_csv('xgb.csv',index=False)
~~~
