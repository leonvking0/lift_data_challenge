

```python
import json
import numpy as np
import pandas as pd
```

## Loading datasets


```python
# load and preview the data
def json_to_df(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame.from_dict(data)
```


```python
business_df = json_to_df('yelp_dataset/yelp_academic_dataset_business.json')
review_df = json_to_df('yelp_dataset/yelp_academic_dataset_review.json')
user_df = json_to_df('yelp_dataset/yelp_academic_dataset_user.json')
checkin_df = json_to_df('yelp_dataset/yelp_academic_dataset_checkin.json')
tip_df = json_to_df('yelp_dataset/yelp_academic_dataset_tip.json')
```


```python
business_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>address</th>
      <th>attributes</th>
      <th>business_id</th>
      <th>categories</th>
      <th>city</th>
      <th>hours</th>
      <th>is_open</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>name</th>
      <th>neighborhood</th>
      <th>postal_code</th>
      <th>review_count</th>
      <th>stars</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1314 44 Avenue NE</td>
      <td>{'BikeParking': 'False', 'BusinessAcceptsCredi...</td>
      <td>Apn5Q_b6Nz61Tq4XzPdf9A</td>
      <td>Tours, Breweries, Pizza, Restaurants, Food, Ho...</td>
      <td>Calgary</td>
      <td>{'Monday': '8:30-17:0', 'Tuesday': '11:0-21:0'...</td>
      <td>1</td>
      <td>51.091813</td>
      <td>-114.031675</td>
      <td>Minhas Micro Brewery</td>
      <td></td>
      <td>T2E 6L6</td>
      <td>24</td>
      <td>4.0</td>
      <td>AB</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>{'Alcohol': 'none', 'BikeParking': 'False', 'B...</td>
      <td>AjEbIBw6ZFfln7ePHha9PA</td>
      <td>Chicken Wings, Burgers, Caterers, Street Vendo...</td>
      <td>Henderson</td>
      <td>{'Friday': '17:0-23:0', 'Saturday': '17:0-23:0...</td>
      <td>0</td>
      <td>35.960734</td>
      <td>-114.939821</td>
      <td>CK'S BBQ &amp; Catering</td>
      <td></td>
      <td>89002</td>
      <td>3</td>
      <td>4.5</td>
      <td>NV</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1335 rue Beaubien E</td>
      <td>{'Alcohol': 'beer_and_wine', 'Ambience': '{'ro...</td>
      <td>O8S5hYJ1SMc8fA4QBtVujA</td>
      <td>Breakfast &amp; Brunch, Restaurants, French, Sandw...</td>
      <td>Montréal</td>
      <td>{'Monday': '10:0-22:0', 'Tuesday': '10:0-22:0'...</td>
      <td>0</td>
      <td>45.540503</td>
      <td>-73.599300</td>
      <td>La Bastringue</td>
      <td>Rosemont-La Petite-Patrie</td>
      <td>H2G 1K7</td>
      <td>5</td>
      <td>4.0</td>
      <td>QC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>211 W Monroe St</td>
      <td>None</td>
      <td>bFzdJJ3wp3PZssNEsyU23g</td>
      <td>Insurance, Financial Services</td>
      <td>Phoenix</td>
      <td>None</td>
      <td>1</td>
      <td>33.449999</td>
      <td>-112.076979</td>
      <td>Geico Insurance</td>
      <td></td>
      <td>85003</td>
      <td>8</td>
      <td>1.5</td>
      <td>AZ</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005 Alyth Place SE</td>
      <td>{'BusinessAcceptsCreditCards': 'True'}</td>
      <td>8USyCYqpScwiNEb58Bt6CA</td>
      <td>Home &amp; Garden, Nurseries &amp; Gardening, Shopping...</td>
      <td>Calgary</td>
      <td>{'Monday': '8:0-17:0', 'Tuesday': '8:0-17:0', ...</td>
      <td>1</td>
      <td>51.035591</td>
      <td>-114.027366</td>
      <td>Action Engine</td>
      <td></td>
      <td>T2H 0N5</td>
      <td>4</td>
      <td>2.0</td>
      <td>AB</td>
    </tr>
  </tbody>
</table>
</div>




```python
review_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_id</th>
      <th>cool</th>
      <th>date</th>
      <th>funny</th>
      <th>review_id</th>
      <th>stars</th>
      <th>text</th>
      <th>useful</th>
      <th>user_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>iCQpiavjjPzJ5_3gPD5Ebg</td>
      <td>0</td>
      <td>2011-02-25</td>
      <td>0</td>
      <td>x7mDIiDB3jEiPGPHOmDzyw</td>
      <td>2</td>
      <td>The pizza was okay. Not the best I've had. I p...</td>
      <td>0</td>
      <td>msQe1u7Z_XuqjGoqhB0J5g</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pomGBqfbxcqPv14c3XH-ZQ</td>
      <td>0</td>
      <td>2012-11-13</td>
      <td>0</td>
      <td>dDl8zu1vWPdKGihJrwQbpw</td>
      <td>5</td>
      <td>I love this place! My fiance And I go here atl...</td>
      <td>0</td>
      <td>msQe1u7Z_XuqjGoqhB0J5g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>jtQARsP6P-LbkyjbO1qNGg</td>
      <td>1</td>
      <td>2014-10-23</td>
      <td>1</td>
      <td>LZp4UX5zK3e-c5ZGSeo3kA</td>
      <td>1</td>
      <td>Terrible. Dry corn bread. Rib tips were all fa...</td>
      <td>3</td>
      <td>msQe1u7Z_XuqjGoqhB0J5g</td>
    </tr>
    <tr>
      <th>3</th>
      <td>elqbBhBfElMNSrjFqW3now</td>
      <td>0</td>
      <td>2011-02-25</td>
      <td>0</td>
      <td>Er4NBWCmCD4nM8_p1GRdow</td>
      <td>2</td>
      <td>Back in 2005-2007 this place was my FAVORITE t...</td>
      <td>2</td>
      <td>msQe1u7Z_XuqjGoqhB0J5g</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ums3gaP2qM3W1XcA5r6SsQ</td>
      <td>0</td>
      <td>2014-09-05</td>
      <td>0</td>
      <td>jsDu6QEJHbwP2Blom1PLCA</td>
      <td>5</td>
      <td>Delicious healthy food. The steak is amazing. ...</td>
      <td>0</td>
      <td>msQe1u7Z_XuqjGoqhB0J5g</td>
    </tr>
  </tbody>
</table>
</div>




```python
user_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>average_stars</th>
      <th>compliment_cool</th>
      <th>compliment_cute</th>
      <th>compliment_funny</th>
      <th>compliment_hot</th>
      <th>compliment_list</th>
      <th>compliment_more</th>
      <th>compliment_note</th>
      <th>compliment_photos</th>
      <th>compliment_plain</th>
      <th>...</th>
      <th>cool</th>
      <th>elite</th>
      <th>fans</th>
      <th>friends</th>
      <th>funny</th>
      <th>name</th>
      <th>review_count</th>
      <th>useful</th>
      <th>user_id</th>
      <th>yelping_since</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>None</td>
      <td>0</td>
      <td>None</td>
      <td>0</td>
      <td>Susan</td>
      <td>1</td>
      <td>0</td>
      <td>lzlZwIpuSWXEnNS91wxjHw</td>
      <td>2015-09-28</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>None</td>
      <td>0</td>
      <td>None</td>
      <td>0</td>
      <td>Daipayan</td>
      <td>2</td>
      <td>0</td>
      <td>XvLBr-9smbI0m_a7dXtB7w</td>
      <td>2015-09-05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>None</td>
      <td>0</td>
      <td>None</td>
      <td>0</td>
      <td>Andy</td>
      <td>1</td>
      <td>0</td>
      <td>QPT4Ud4H5sJVr68yXhoWFw</td>
      <td>2016-07-21</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.05</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>None</td>
      <td>0</td>
      <td>None</td>
      <td>0</td>
      <td>Jonathan</td>
      <td>19</td>
      <td>0</td>
      <td>i5YitlHZpf0B3R0s_8NVuw</td>
      <td>2014-08-04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>None</td>
      <td>0</td>
      <td>None</td>
      <td>0</td>
      <td>Shashank</td>
      <td>3</td>
      <td>0</td>
      <td>s4FoIXE_LSGviTHBe8dmcg</td>
      <td>2017-06-18</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
checkin_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_id</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7KPBkxAOEtb3QeIL9PEErg</td>
      <td>{'Fri-0': 2, 'Sat-0': 1, 'Sun-0': 1, 'Wed-0': ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>kREVIrSBbtqBhIYkTccQUg</td>
      <td>{'Mon-13': 1, 'Thu-13': 1, 'Sat-16': 1, 'Wed-1...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tJRDll5yqpZwehenzE2cSg</td>
      <td>{'Thu-0': 1, 'Mon-1': 1, 'Mon-12': 1, 'Sat-16'...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tZccfdl6JNw-j5BKnCTIQQ</td>
      <td>{'Sun-14': 1, 'Fri-18': 1, 'Mon-20': 1}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>r1p7RAMzCV_6NPF0dNoR3g</td>
      <td>{'Sat-3': 1, 'Sun-18': 1, 'Sat-21': 1, 'Sat-23...</td>
    </tr>
  </tbody>
</table>
</div>



## Subsetting dataset for our purpose of analysis


```python
trt_business_df = business_df[business_df['city'] == 'Toronto']
```


```python
trt_business_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>address</th>
      <th>attributes</th>
      <th>business_id</th>
      <th>categories</th>
      <th>city</th>
      <th>hours</th>
      <th>is_open</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>name</th>
      <th>neighborhood</th>
      <th>postal_code</th>
      <th>review_count</th>
      <th>stars</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>631 Bloor St W</td>
      <td>{'BusinessParking': '{'garage': False, 'street...</td>
      <td>9A2quhZLyWk0akUetBd8hQ</td>
      <td>Food, Bakeries</td>
      <td>Toronto</td>
      <td>None</td>
      <td>0</td>
      <td>43.664378</td>
      <td>-79.414424</td>
      <td>Bnc Cake House</td>
      <td>Koreatown</td>
      <td>M6G 1K8</td>
      <td>7</td>
      <td>4.0</td>
      <td>ON</td>
    </tr>
    <tr>
      <th>11</th>
      <td>595 Markham Street</td>
      <td>{'Alcohol': 'full_bar', 'Ambience': '{'romanti...</td>
      <td>tZnSodhPwNr4bzrwJ1CSbw</td>
      <td>Cajun/Creole, Southern, Restaurants</td>
      <td>Toronto</td>
      <td>{'Tuesday': '17:0-1:0', 'Wednesday': '17:0-1:0...</td>
      <td>0</td>
      <td>43.664125</td>
      <td>-79.411886</td>
      <td>Southern Accent Restaurant</td>
      <td>Palmerston</td>
      <td>M6G 2L7</td>
      <td>146</td>
      <td>4.0</td>
      <td>ON</td>
    </tr>
    <tr>
      <th>23</th>
      <td>746 Street Clair Avenue W</td>
      <td>{'BikeParking': 'True', 'BusinessAcceptsCredit...</td>
      <td>5J3b7j3Fzo9ISjChmoUoUA</td>
      <td>Food, Bakeries, Coffee &amp; Tea</td>
      <td>Toronto</td>
      <td>{'Monday': '7:30-19:0', 'Tuesday': '7:30-19:0'...</td>
      <td>1</td>
      <td>43.681328</td>
      <td>-79.427884</td>
      <td>Mabel's Bakery</td>
      <td>Wychwood</td>
      <td>M6C 1B5</td>
      <td>23</td>
      <td>4.0</td>
      <td>ON</td>
    </tr>
    <tr>
      <th>27</th>
      <td>99 Yorkville Avenue</td>
      <td>{'Ambience': '{'romantic': False, 'intimate': ...</td>
      <td>PMDlKLd0Mxj0ngCpuUmE5Q</td>
      <td>Restaurants, Food, Canadian (New), Coffee &amp; Tea</td>
      <td>Toronto</td>
      <td>None</td>
      <td>0</td>
      <td>43.670885</td>
      <td>-79.392379</td>
      <td>The Coffee Mill Restaurant</td>
      <td>Yorkville</td>
      <td>M5R 3K5</td>
      <td>25</td>
      <td>3.5</td>
      <td>ON</td>
    </tr>
    <tr>
      <th>43</th>
      <td>3280 Kingston Road</td>
      <td>None</td>
      <td>zHwXoh40k86P0aiN1aix9Q</td>
      <td>Hotels, Hotels &amp; Travel, Event Planning &amp; Serv...</td>
      <td>Toronto</td>
      <td>None</td>
      <td>1</td>
      <td>43.733395</td>
      <td>-79.224206</td>
      <td>Super 8 by Wyndham Toronto East ON</td>
      <td>Scarborough</td>
      <td>M1M 1P8</td>
      <td>3</td>
      <td>2.0</td>
      <td>ON</td>
    </tr>
  </tbody>
</table>
</div>




```python
## perform one-hot encoding on business categories
trt_business_categories = trt_business_df['categories'].str.get_dummies(', ')
```


```python
print(trt_business_categories.columns)
```

    Index(['& Probates', '3D Printing', 'Acai Bowls', 'Accessories', 'Accountants',
           'Acne Treatment', 'Active Life', 'Acupuncture', 'Adult',
           'Adult Education',
           ...
           'Wine & Spirits', 'Wine Bars', 'Wine Tasting Classes', 'Wine Tours',
           'Wineries', 'Women's Clothing', 'Wraps', 'Yelp Events', 'Yoga', 'Zoos'],
          dtype='object', length=890)



```python
## only keep businesses which have 'Restaurants' in categories 
## (ignored other related categories such as 'Food', 'Bakeries', etc.)
trt_resaurant_df = trt_business_df[trt_business_categories['Restaurants'] > 0]
```

## Find most popular restaurants based on number of checkins

**popular** *[attributive]* (of cultural activities or products) intended for or suited to the taste, understanding, or means of the general public rather than specialists or intellectuals. --Oxford Dictionary


```python
## Find the total number of checkins for all businesses
checkin_df['total_checkins'] = list(map(lambda x: sum(x.values()), checkin_df['time']))
```


```python
checkin_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_id</th>
      <th>time</th>
      <th>total_checkins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7KPBkxAOEtb3QeIL9PEErg</td>
      <td>{'Fri-0': 2, 'Sat-0': 1, 'Sun-0': 1, 'Wed-0': ...</td>
      <td>151</td>
    </tr>
    <tr>
      <th>1</th>
      <td>kREVIrSBbtqBhIYkTccQUg</td>
      <td>{'Mon-13': 1, 'Thu-13': 1, 'Sat-16': 1, 'Wed-1...</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>tJRDll5yqpZwehenzE2cSg</td>
      <td>{'Thu-0': 1, 'Mon-1': 1, 'Mon-12': 1, 'Sat-16'...</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>tZccfdl6JNw-j5BKnCTIQQ</td>
      <td>{'Sun-14': 1, 'Fri-18': 1, 'Mon-20': 1}</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>r1p7RAMzCV_6NPF0dNoR3g</td>
      <td>{'Sat-3': 1, 'Sun-18': 1, 'Sat-21': 1, 'Sat-23...</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
popular_restaurants = pd.merge(trt_resaurant_df, checkin_df, on='business_id', how='left')
```


```python
popular_restaurants = popular_restaurants.sort_values(by='total_checkins', ascending=False)
popular_restaurants = popular_restaurants[['name', 'review_count', 'stars', 'total_checkins']]
```


```python
## Display Top 10 restaurants with higest popularity
popular_restaurants.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>review_count</th>
      <th>stars</th>
      <th>total_checkins</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2814</th>
      <td>Pai Northern Thai Kitchen</td>
      <td>1837</td>
      <td>4.5</td>
      <td>2550.0</td>
    </tr>
    <tr>
      <th>3983</th>
      <td>Insomnia Restaurant &amp; Lounge</td>
      <td>741</td>
      <td>4.0</td>
      <td>2095.0</td>
    </tr>
    <tr>
      <th>1172</th>
      <td>Pearl Diver</td>
      <td>484</td>
      <td>4.0</td>
      <td>2076.0</td>
    </tr>
    <tr>
      <th>4598</th>
      <td>Banh Mi Boys</td>
      <td>999</td>
      <td>4.5</td>
      <td>1759.0</td>
    </tr>
    <tr>
      <th>2358</th>
      <td>Salad King Restaurant</td>
      <td>855</td>
      <td>3.5</td>
      <td>1649.0</td>
    </tr>
    <tr>
      <th>2419</th>
      <td>Sansotei Ramen</td>
      <td>762</td>
      <td>4.0</td>
      <td>1627.0</td>
    </tr>
    <tr>
      <th>1783</th>
      <td>KINKA IZAKAYA ORIGINAL</td>
      <td>1306</td>
      <td>4.0</td>
      <td>1627.0</td>
    </tr>
    <tr>
      <th>4663</th>
      <td>Hokkaido Ramen Santouka</td>
      <td>713</td>
      <td>4.0</td>
      <td>1625.0</td>
    </tr>
    <tr>
      <th>1106</th>
      <td>Khao San Road</td>
      <td>1336</td>
      <td>4.0</td>
      <td>1467.0</td>
    </tr>
    <tr>
      <th>4054</th>
      <td>Seven Lives Tacos Y Mariscos</td>
      <td>1048</td>
      <td>4.5</td>
      <td>1442.0</td>
    </tr>
  </tbody>
</table>
</div>



## How many Canadian residents reviewed the business “Mon Ami Gabi” in last 1 year?


```python
## Find the business_id of "Mon Ami Gabi"
gabi_id = business_df[business_df['name'] == "Mon Ami Gabi"]['business_id']
gabi_id
```




    137635    4JNXUYY8wbaaDmk3BPzlWw
    Name: business_id, dtype: object




```python
## Since there is only one “Mon Ami Gabi”, we only need to
## consider one particular business id
gabi_id = gabi_id.iloc[0]
```

We do not have information about the residency of reviewers from 
provided datasets. However, we can infer that a reviewer is a canadian
resident if more than 70% of his/her reviews are on canadian businesses. 


```python
## list of canadian provinces
canadian_prov = ['ON', 'QC', 'NS', 'NB', 'MB', 'BC', 'PE', 'SK' \
                 'AB', 'NL']
```


```python
## subset businesses on canadian provinces
ca_business_df = business_df[business_df['state'].isin(canadian_prov)]
ca_business_id = list(ca_business_df['business_id'])
```


```python
## find ids of all users made a least one review
reviewer_id = pd.Series(review_df['user_id']).unique()
```


```python
## To avoid the string comparison overhead, create dummy id for user
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
reviewer_id.sort()
```


```python
## subset reviews on canadian businesses
ca_review_df = review_df[review_df['business_id'].isin(ca_business_id)]
```


```python
import time
start = time.time()
ca_review_df['dummy_user_id'] == 0
review_df['dummy_user_id'] == 0
end = time.time()
print(end - start)
```

    0.008049249649047852



```python
print("Following code takes around {} hours to run".format((end - start) \
                                                           * len(reviewer_id) / 60 / 60))
```

    takes 3.3944781362348135 hours to run



```python
## find canadian residents base on their reviews
## (This takes lots of time to run, total time is estimated above for a signle thread)
ca_resident_dummy_id = []
for i in range(len(reviewer_id)):
    confidency = sum(ca_review_df['dummy_user_id'] == i) / \
                 sum(review_df['dummy_user_id'] == i)
    if confidency > 0.7:
        ca_resident_dummy_id.append(i)
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-48-2ce852a2da4c> in <module>()
          3 ca_resident_dummy_id = []
          4 for i in range(len(reviewer_id)):
    ----> 5     confidency = sum(ca_review_df['dummy_user_id'] == i) /                  sum(review_df['dummy_user_id'] == i)
          6     if confidency > 0.7:
          7         ca_resident_dummy_id.append(i)


    KeyboardInterrupt: 



```python
gabi_review = review_df[review_df['business_id'] == gabi_id]
```


```python
## subset gabi_review for last 1 year
gabi_review = gabi_review[('2018-08-13' > gabi_review['date']) & (gabi_review['date'] > '2017-08-13')]
```


```python
## compute the number of Canadian residents reviewed on Gabi last year
print (len(gabi_review[gabi_review['dummy_user_id'].\
                       isin(ca_resident_dummy_id)]['dummy_user_id'].unique()))
```

    0


## Top 10 most common words in the reviews of the business “Chipotle Mexican Grill”


```python
## Find the business_id of "Chipotle Mexican Grill"
chipotle_id_list = list(business_df[business_df['name'] == "Chipotle Mexican Grill"]['business_id'])
```


```python
## Subset reviews on list of business ids
chipotle_review = review_df[review_df['business_id'].isin(chipotle_id_list)]
```


```python
chipotle_review_text = list(chipotle_review['text'])
```


```python
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
## convert stop words list to set since list is slow
stop_words = set(stopwords.words("english") + list(punctuation))
```


```python
## obtain a list of words for all reviews, filering out stopwords
review_words_list = []
for t in chipotle_review_text:
    words = word_tokenize(t.lower())
    for w in words:
        if w not in stop_words:
            review_words_list.append(w)    
```


```python
from collections import Counter
review_words_counter = Counter(review_words_list)
```


```python
review_words_df = pd.DataFrame.from_dict(review_words_counter, \
                                         orient='index').reset_index()
review_words_df.columns = ['word', 'frequency']
```


```python
review_words_df.sort_values(by='frequency', \
                            ascending=False, inplace=True)
```


```python
## Print top 10 words and frequencies
review_words_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>chipotle</td>
      <td>8460</td>
    </tr>
    <tr>
      <th>53</th>
      <td>food</td>
      <td>6754</td>
    </tr>
    <tr>
      <th>10</th>
      <td>n't</td>
      <td>6512</td>
    </tr>
    <tr>
      <th>129</th>
      <td>'s</td>
      <td>5062</td>
    </tr>
    <tr>
      <th>56</th>
      <td>location</td>
      <td>4546</td>
    </tr>
    <tr>
      <th>5</th>
      <td>burrito</td>
      <td>3859</td>
    </tr>
    <tr>
      <th>123</th>
      <td>get</td>
      <td>3564</td>
    </tr>
    <tr>
      <th>19</th>
      <td>one</td>
      <td>3464</td>
    </tr>
    <tr>
      <th>204</th>
      <td>time</td>
      <td>3359</td>
    </tr>
    <tr>
      <th>39</th>
      <td>like</td>
      <td>3292</td>
    </tr>
  </tbody>
</table>
</div>



## Find percentage of users, who reviewed “Mon Ami Gabi”, and also reviewed at least 10 other restaurants located in Ontario


```python
## Find the business_id of "Mon Ami Gabi"
gabi_id = business_df[business_df['name'] == "Mon Ami Gabi"]['business_id'].iloc[0]
```


```python
## Find the unique users who reviewed "Mon Ami Gabi", assume tip is also kind of review
user_gabi = list(review_df['user_id'][review_df['business_id'] == gabi_id]) + \
            list(tip_df['user_id'][tip_df['business_id'] == gabi_id].values)
user_gabi = pd.Series(user_gabi).unique()
```


```python
## Find ontario restaurants
on_business_df = business_df[business_df['state'] == 'ON']
on_restaurants_df = on_business_df[on_business_df['categories'].str.contains('Restaurants') == True]
```


```python
on_restaurants_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>address</th>
      <th>attributes</th>
      <th>business_id</th>
      <th>categories</th>
      <th>city</th>
      <th>hours</th>
      <th>is_open</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>name</th>
      <th>neighborhood</th>
      <th>postal_code</th>
      <th>review_count</th>
      <th>stars</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>3417 Derry Road E, Unit 103</td>
      <td>{'Alcohol': 'none', 'BusinessAcceptsCreditCard...</td>
      <td>6OuOZAok8ikONMS_T3EzXg</td>
      <td>Restaurants, Thai</td>
      <td>Mississauga</td>
      <td>None</td>
      <td>1</td>
      <td>43.712946</td>
      <td>-79.632763</td>
      <td>Thai One On</td>
      <td>Ridgewood</td>
      <td>L4T 1A8</td>
      <td>7</td>
      <td>2.0</td>
      <td>ON</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4568 Highway 7 E</td>
      <td>{'GoodForKids': 'True', 'NoiseLevel': 'loud', ...</td>
      <td>KapTdGyGs7RK0c68Z6hhhg</td>
      <td>Restaurants, Japanese</td>
      <td>Markham</td>
      <td>{'Monday': '11:30-23:0', 'Tuesday': '11:30-23:...</td>
      <td>0</td>
      <td>43.862484</td>
      <td>-79.306960</td>
      <td>Sushi 8</td>
      <td>Unionville</td>
      <td>L3R 1M5</td>
      <td>12</td>
      <td>1.5</td>
      <td>ON</td>
    </tr>
    <tr>
      <th>11</th>
      <td>595 Markham Street</td>
      <td>{'Alcohol': 'full_bar', 'Ambience': '{'romanti...</td>
      <td>tZnSodhPwNr4bzrwJ1CSbw</td>
      <td>Cajun/Creole, Southern, Restaurants</td>
      <td>Toronto</td>
      <td>{'Tuesday': '17:0-1:0', 'Wednesday': '17:0-1:0...</td>
      <td>0</td>
      <td>43.664125</td>
      <td>-79.411886</td>
      <td>Southern Accent Restaurant</td>
      <td>Palmerston</td>
      <td>M6G 2L7</td>
      <td>146</td>
      <td>4.0</td>
      <td>ON</td>
    </tr>
    <tr>
      <th>27</th>
      <td>99 Yorkville Avenue</td>
      <td>{'Ambience': '{'romantic': False, 'intimate': ...</td>
      <td>PMDlKLd0Mxj0ngCpuUmE5Q</td>
      <td>Restaurants, Food, Canadian (New), Coffee &amp; Tea</td>
      <td>Toronto</td>
      <td>None</td>
      <td>0</td>
      <td>43.670885</td>
      <td>-79.392379</td>
      <td>The Coffee Mill Restaurant</td>
      <td>Yorkville</td>
      <td>M5R 3K5</td>
      <td>25</td>
      <td>3.5</td>
      <td>ON</td>
    </tr>
    <tr>
      <th>53</th>
      <td>788 Wilson Avenue</td>
      <td>{'BusinessAcceptsCreditCards': 'False', 'GoodF...</td>
      <td>Be7Mwq06nf1eNLblo1ekow</td>
      <td>Bakeries, Food, Latin American, Restaurants, S...</td>
      <td>North York</td>
      <td>None</td>
      <td>1</td>
      <td>43.731316</td>
      <td>-79.465133</td>
      <td>La Rosa Chilena</td>
      <td>Downsview</td>
      <td>M3K 1E2</td>
      <td>5</td>
      <td>2.5</td>
      <td>ON</td>
    </tr>
  </tbody>
</table>
</div>




```python
review_and_tip = pd.concat([review_df[['business_id', 'user_id']], \
                            tip_df[['business_id', 'user_id']]], ignore_index=True)
```


```python
## Pull all reviews done by users who reviewed "Mon Ami Gabi"
user_gabi = pd.DataFrame(pd.Series(user_gabi, name='user_id'))
review_all_user_gabi = pd.merge(user_gabi, review_and_tip, on='user_id', how='left')
```


```python
## Subset reviews by restaurants located in ontario
review_on_user_gabi = pd.merge(on_restaurants_df[['business_id', 'name', 'city']], \
                               review_all_user_gabi, on='business_id', how='inner')
```


```python
review_on_user_gabi['review_count'] = 1
review_on_user_gabi_count = review_on_user_gabi.groupby('user_id').sum()['review_count']
```


```python
qualified_user_num = sum(review_on_user_gabi_count >= 10)
```


```python
## Compute the percentage of users who reviewed “Mon Ami Gabi”, and also reviewed at
## least 10 other restaurants located in Ontario
print("{0:.3%}".format(qualified_user_num / user_gabi.size))
```

    0.737%


## Two more analytics which provide insights and help existing/future Business Owners​

1. Ratings and text review of 'Elite' users. 'Elite' users can be users who has most friend connections/most reviews/most reviews considered as useful or cool
2. Correlation between business atrributes and ratings. Business arrtibutes are for example, bike parking, car parking, credit card acceptence, has alcohol berverage etc.
