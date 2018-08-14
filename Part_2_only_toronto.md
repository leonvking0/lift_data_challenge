
## A reccommendation system with collaborative filtering


```python
import json
import numpy as np
import pandas as pd
from scipy import optimize
```


```python
# load the data
def json_to_df(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame.from_dict(data)
```


```python
review_df = json_to_df('yelp_dataset/yelp_academic_dataset_review.json')
business_df = json_to_df('yelp_dataset/yelp_academic_dataset_business.json')
#user_df = json_to_df('yelp_dataset/yelp_academic_dataset_user.json')
```


```python
## Subset the business to only contain toronto restaurants (mainly because of memory issue)
business_df = business_df[business_df['categories'].str.contains('Restaurants') == True]
business_df = business_df[business_df['city'] == 'Toronto']
```


```python
business_id_list = list(business_df['business_id'])
```


```python
business_df.shape
```




    (7578, 15)




```python
## Subset reviews to reviews on restaurants only
review_df = review_df[review_df['business_id'].isin(business_id_list)]
```


```python
## Find user id for users who reviewed restaurants
user_id_list = list(review_df['user_id'].unique())
```


```python
print("There are {} restaurants and {} users who reviewed them.".\
      format(len(business_id_list), len(user_id_list)))
```

    There are 7578 restaurants and 80854 users who reviewed them.



```python
review_df.sort_values(by='user_id', inplace=True)
```


```python
## Make a dummy variable which index user_id
id_list = list(review_df['user_id'])
dummy = [0] * len(review_df)
last = None
counter = -1
for i in range(len(dummy)):
    current = id_list[i]
    if current != last:
        counter += 1
    dummy[i] = counter
    last = current
```


```python
review_df['dummy_user_id'] = dummy
```


```python
review_df.sort_values(by='business_id', inplace=True)
```


```python
## Make a dummy variable which index business_id
id_list = list(review_df['business_id'])
dummy = [0] * len(review_df)
last = None
counter = -1
for i in range(len(dummy)):
    current = id_list[i]
    if current != last:
        counter += 1
    dummy[i] = counter
    last = current
```


```python
review_df['dummy_business_id'] = dummy
```


```python
## Create a m*k matrix which holds rating for restaurants(star)
## Each row represents a unique restaurant and each column represents a unique user.
## All ratings should be from 1-5, 0 represents missing value
m = len(business_id_list)
k = len(user_id_list)
rating_mtrx = np.zeros((m,k))
did_rate = np.zeros((m,k))
```


```python
## store rated ratings into the matrix
ratings = list(review_df['stars'])
i_index = list(review_df['dummy_business_id'])
j_index = list(review_df['dummy_user_id'])
for c in range(len(ratings)):
    ## rating valid only if > 0
    if ratings[c] > 0:
        rating_mtrx[i_index[c], j_index[c]] = ratings[c]
        did_rate[i_index[c], j_index[c]] = 1
```


```python
rating_mtrx.shape
```




    (7578, 80854)




```python
## Mean Normalize All The Ratings, only on entries that has ratings

def normalize_ratings(ratings, did_rate):
    num_restaurants = ratings.shape[0]
    ratings_mean = np.zeros(shape = (num_restaurants, 1))
    ratings_norm = np.zeros(shape = ratings.shape)

    for i in range(num_restaurants):
        # Get all the indexes where there is a 1
        idx = np.where(did_rate[i] ==1)[0]

        # Calculate mean rating of ith restaurant only from user's that gave a rating
        ratings_mean[i] = np.mean(ratings[i, idx])
        ratings_norm[i, idx] = ratings[i, idx] - ratings_mean[i]

    return (ratings_norm, ratings_mean)
```


```python
ratings_norm, ratings_mean = normalize_ratings(rating_mtrx, did_rate)
```


```python
## Define cost function
def calculate_cost(X_and_theta, ratings, did_rate, num_users, num_restaurants, num_features, reg_param):
    # Retrieve the X and theta matrixes from X_and_theta, based on their dimensions
    # ------------------------------------------------------------------------------------------------------
    # Get the first (m * 3) rows in the 3*(m+k) X 1 column vector
    first_3m = X_and_theta[:num_restaurants * num_features]
    # Reshape this column vector into a m X 3 matrix
    X = first_3m.reshape((num_features, num_restaurants)).transpose()
    # Get the rest of the numers, after the first 3*m
    last_3k = X_and_theta[num_restaurants * num_features:]
    # Reshape this column vector into a k X 3 matrix
    theta = last_3k.reshape(num_features, num_users).transpose()

    # we calculate the sum of squared errors here.  
    # in other words, we calculate the squared difference between our predictions and ratings
    cost = np.sum( (X.dot( theta.T ) * did_rate - ratings) ** 2 ) / 2
    print(cost)
    # we get the sum of the square of every element of X and theta
    regularization = (reg_param / 2) * (np.sum( theta**2 ) + np.sum(X**2))
    print('--')
    print(regularization)
    return cost + regularization
```


```python
def calculate_gradient(X_and_theta, ratings, did_rate, num_users, num_restaurants, num_features, reg_param):
    # Retrieve the X and theta matrixes from X_and_theta, based on their dimensions 
    # --------------------------------------------------------------------------------------------------------
    # Get the first (m * 3) rows in the 3*(m+k) X 1 column vector
    first_3m = X_and_theta[:num_restaurants * num_features]
    # Reshape this column vector into a m X 3 matrix
    X = first_3m.reshape((num_features, num_restaurants)).transpose()
    # Get the rest of the numers, after the first 3*m
    last_3k = X_and_theta[num_restaurants * num_features:]
    # Reshape this column vector into a k X 3 matrix
    theta = last_3k.reshape(num_features, num_users).transpose()

    # we multiply by did_rate because we only want to consider observations for which a rating was given
    difference = X.dot(theta.T) * did_rate - ratings

    # we calculate the gradients (derivatives) of the cost with respect to X and theta
    X_grad = difference.dot( theta ) + reg_param * X
    theta_grad = difference.T.dot( X ) + reg_param * theta

    # wrap the gradients back into a column vector 
    return np.r_[X_grad.T.flatten(), theta_grad.T.flatten()]
```


```python
X_and_theta = initial_X_and_theta
first_3m = X_and_theta[:num_restaurants * num_features]
# Reshape this column vector into a m X 3 matrix
X = first_3m.reshape((num_features, num_restaurants)).transpose()
# Get the rest of the numers, after the first 3*m
last_3k = X_and_theta[num_restaurants * num_features:]
# Reshape this column vector into a k X 3 matrix
theta = last_3k.reshape(num_features, num_users).transpose()
```


```python
num_restaurants, num_users = m, k
num_features = 3

# Initialize Parameters theta (user_prefs), X (restaurant_features)
restaurant_features = np.random.randn(num_restaurants, num_features)
user_prefs = np.random.randn(num_users, num_features)
initial_X_and_theta = np.r_[restaurant_features.T.flatten(), user_prefs.T.flatten()]
```


```python

# Regularization paramater
reg_param = 30.0

# fprime simply refers to the derivative (gradient) of the calculate_cost function
# We iterate 100 times
minimized_cost_and_optimal_params = optimize.fmin_cg(calculate_cost, fprime=calculate_gradient, x0=initial_X_and_theta, \
                            args=(rating_mtrx, did_rate, num_users, num_restaurants, num_features, reg_param), \
                            maxiter=100, disp=True, full_output=True )
```

    2970200.9450644115
    --
    3964363.309003002
    2957586.4456436317
    --
    3952160.4604763077
    2909774.5315716956
    --
    3903655.096369522
    2757567.616272013
    --
    3714530.1199423824
    2597943.093631479
    --
    3036373.8942338326
    2825175.3462391687
    --
    1730983.3628583185
    3136005.5957635585
    --
    896025.8539524083
    2541970.2075725854
    --
    1226811.810483822
    3038868.8058608314
    --
    816752.6241703387
    2451069.876944005
    --
    1033020.0908068882
    2366230.669043193
    --
    726687.6447741814
    5030889.853515998
    --
    348404.3461208316
    2399536.729741383
    --
    474599.3493470634
    3244406.1373481257
    --
    214275.64433952398
    2360093.7025356437
    --
    289107.89096901205
    2145955.107376609
    --
    188058.43303563894
    1858906.2342516426
    --
    216200.69599597677
    1832994.459647082
    --
    435774.1981282416
    1669984.3308034488
    --
    275921.3641172167
    1541288.9625791858
    --
    266233.0577411079
    1554616.1207574545
    --
    283921.5839481097
    1474117.6842916596
    --
    270838.5470608167
    1359375.2417066635
    --
    300854.7629288458
    1255764.6974733698
    --
    355488.31568402087
    1263352.616301798
    --
    335196.04229393695
    1176324.0263195774
    --
    401544.42155328405
    1189146.7376765816
    --
    370116.92238599726
    1089280.5025525072
    --
    416181.37031031516
    1054117.3388829587
    --
    452542.9337110515
    1046169.1270874479
    --
    432534.1920442838
    1006080.3322389021
    --
    465389.3092249544
    1012504.6700198601
    --
    450516.4697482726
    962863.9736680267
    --
    477196.813182481
    935366.9853755014
    --
    499779.11718908325
    848186.154449481
    --
    566909.2855422079
    763396.3551052837
    --
    653121.4006173881
    797661.1986923359
    --
    607244.3154306868
    788206.2161977793
    --
    605486.3565863578
    792486.3527992362
    --
    602700.6588273484
    784332.6076070807
    --
    604070.7448528616
    785583.6813511229
    --
    602029.5745536878
    782659.9345343997
    --
    602863.7990099963
    778708.7546900617
    --
    603041.4763905818
    775145.2920404181
    --
    604239.9854190781
    777149.9055279338
    --
    608913.7027051294
    773897.6053837077
    --
    604981.5221989001
    772988.3722934505
    --
    604964.7147658404
    771261.0922761194
    --
    604915.5988413944
    771024.8821056813
    --
    604385.9922692371
    770015.6113782435
    --
    604577.7618268469
    766941.1355852582
    --
    606342.7032693538
    761252.2937697248
    --
    610335.5238363459
    755663.1721157506
    --
    613944.1962539041
    747745.6773064985
    --
    621375.3017728811
    750366.910232968
    --
    618094.4205419326
    742150.2472794338
    --
    624845.2638978842
    739010.479380838
    --
    627068.1074719132
    735959.4422775201
    --
    631177.0639649148
    737439.5568338613
    --
    628343.704088647
    736844.0233987238
    --
    628484.4581758004
    736513.6023018965
    --
    628642.3535342905
    734521.0399909174
    --
    630280.0936368427
    734943.9713180399
    --
    629771.3263229063
    731700.4807428232
    --
    632308.9641212143
    727613.2057379911
    --
    636033.5776738825
    718886.0618939602
    --
    643178.1355098155
    711836.2835014757
    --
    649806.7185234227
    694767.5837393994
    --
    666596.585276852
    702090.3759418302
    --
    658453.5283575698
    691599.223924842
    --
    667638.7133230905
    686114.4120217111
    --
    672639.0386892122
    687776.4762671442
    --
    670651.056479386
    686818.161727422
    --
    673090.5087323037
    686950.7699056347
    --
    671261.1018183182
    686570.1181101429
    --
    671418.5299411392
    687102.0897161
    --
    670640.9043694878
    688406.0402752538
    --
    668968.0214260938
    689830.1536981424
    --
    667429.0830313229
    693641.0236239781
    --
    663330.1054647366
    692425.0119749163
    --
    664487.9456833004
    694672.9954654251
    --
    661804.0756746158
    694943.4604351154
    --
    661211.3319778044
    695793.8630176228
    --
    660327.9072360549
    695250.2894612584
    --
    660732.9825736112
    695184.1006786901
    --
    660539.9558966919
    695285.7753652843
    --
    660353.5501638495
    696441.8989276799
    --
    659643.2138777751
    695403.1297855733
    --
    660130.3404620799
    695596.9369136355
    --
    659855.8993542528
    695784.7332205572
    --
    659552.7667623478
    695965.806003051
    --
    659347.6596917021
    695871.8210930255
    --
    659311.2493308575
    695734.3081559093
    --
    659325.5315298857
    695550.5837401012
    --
    659361.3015739319
    695714.2465268272
    --
    659278.7154119706
    695526.2264982047
    --
    659326.8678698094
    695576.7428712824
    --
    659194.0501441098
    695759.9270051741
    --
    659039.4718914509
    695607.9190385821
    --
    659127.4133038763
    695672.4415753429
    --
    659013.2007343422
    695752.0380931352
    --
    658908.4642278797
    695709.1883278289
    --
    658942.9726266615
    695793.5909874035
    --
    658810.9975192879
    695872.8314657013
    --
    658722.3392745295
    695992.4096113606
    --
    658502.341426179
    696463.884034771
    --
    657906.7706378575
    696850.183267919
    --
    657214.9752030019
    696975.5813935938
    --
    656998.7399260332
    696799.9248290972
    --
    657084.2340041382
    697422.2747985509
    --
    656835.8293206965
    696820.0680169859
    --
    657018.8921166204
    696818.8235722306
    --
    656950.1695200788
    696865.0177283314
    --
    656876.3111544963
    696995.9878434538
    --
    656727.492035
    696895.3625109255
    --
    656791.7455534459
    696917.5920237851
    --
    656691.7088181041
    696968.4005154114
    --
    656624.2164103024
    696990.6363408365
    --
    656499.8561558988
    696941.842368991
    --
    656434.112936137
    696850.1198812607
    --
    656409.6334589937
    696700.8510047306
    --
    656563.5764979906
    696728.2425738549
    --
    656473.8873218129
    696498.057914739
    --
    656612.7405887113
    696239.6455157098
    --
    656827.9154385421
    695547.9414964394
    --
    657391.068736112
    694759.3478850541
    --
    658092.5185120751
    693549.6018720742
    --
    659279.1731840277
    694062.8912250538
    --
    658737.9937250692
    694086.5513770082
    --
    658689.0563002835
    694061.3503105937
    --
    658705.4140674795
    694359.9877562119
    --
    658400.1461608013
    694211.5184605576
    --
    658536.1849050255
    694563.6660040066
    --
    658162.1335276419
    695003.7645551834
    --
    657702.3880286927
    695715.6606358442
    --
    656985.1617737564
    695407.5389120511
    --
    657287.163446766
    695631.2743896423
    --
    657051.0105238046
    695780.3019453397
    --
    656892.4720379149
    695944.7896109186
    --
    656727.6977068868
    695859.7240549256
    --
    656808.1764159465
    695826.8563162722
    --
    656833.0735416153
    695781.1314146764
    --
    656873.8408980917
    695623.3106244892
    --
    657034.1245832779
    695702.1161829726
    --
    656946.88876796
    695552.6739934561
    --
    657088.4324926642
    695391.6690275663
    --
    657245.9096565607
    695446.5812491523
    --
    657189.4038200575
    695265.8264352178
    --
    657366.6055601311
    695068.100994719
    --
    657558.9085658689
    694864.1739432532
    --
    657760.9216322284
    694717.8816490461
    --
    657898.0834813734
    694663.3684649118
    --
    657946.5098321659
    694656.1593137977
    --
    657981.4783687396
    694657.4163664975
    --
    657951.5182431078
    694662.2998107945
    --
    657944.9109282972
    694685.4268767001
    --
    657918.5356801554
    694758.6302547487
    --
    657844.8917524796
    694721.3564832578
    --
    657879.9801790159
    694815.0311835645
    --
    657783.38690315
    694904.1227422974
    --
    657689.4420608724
    695071.5555496918
    --
    657518.5219715064
    695357.754929807
    --
    657233.2119862812
    695203.8014454317
    --
    657382.2762280868
    Warning: Maximum number of iterations has been exceeded.
             Current function value: 1352586.077674
             Iterations: 100
             Function evaluations: 163
             Gradient evaluations: 163



```python
# Retrieve the minimized cost and the optimal values of the movie_features (X) and user_prefs (theta) matrices
cost, optimal_restaurant_features_and_user_prefs = \
minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]
```


```python
first_3m = optimal_restaurant_features_and_user_prefs[:num_restaurants * num_features]
restaurant_features = first_3m.reshape((num_features, num_restaurants)).transpose()
last_3k = optimal_restaurant_features_and_user_prefs[num_restaurants * num_features:]
user_prefs = last_3k.reshape(num_features, num_users ).transpose()
```


```python
## Making predictions
all_predictions = restaurant_features.dot(user_prefs.T) + ratings_mean
```
