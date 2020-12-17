# Project 2 : Features that Drive House Prices (Ames)


## Problem Statement
This is a unidimensional regression problem which predicts housing prices based on several features of a house (Eg.house living area, garage capacity, lot frontage, house type, masonry etc.), based on Ames Housing Dataset.

## Introduction
As a real estate agent, knowing the top features that drive house prices, we would be able to understand what features of a house are more valuable to home buyers and subsequently, drive efforts towards successful sale by sharing this information to home owners who intend to sell their house. Based on the Ames Housing Dataset, we apply a supervised learning model, from which we would be able to ascertain which are the top features that drive house prices, more than others.

## Data Dictionary
Below the description of each features in the final dataset used in the model.  
| Feature | Type | Description |
| :-----: | :--: | :---------- |
| overall_qual | ordinal, categorical | rating of overall material and finish of the house |
| mas_vnr_area | numerical, continuous | masonry veneer area (sq ft) |
| exter_qual | ordinal, categorical | rating of quality of the material on the exterior |
| bsmt_qual | ordinal, categorical | evaluates the height of the basement (inches) |
| heating_qc | ordinal, categorical | healting quality and condition |
| electrical | ordinal, categorical | electrical system types |
| kitchen_qual | ordinal, categorical | kitchen quality |
| totrms_abvgrd | numerical, discrete | total rooms above grade, excluding bathrooms |
| functional | ordinal, categorical | home functionality |
| garage_finish | ordinal, categorical | interior finish of the garage |
| garage_area | numerical, continuous | size of garage (sq ft) |
| paved_drive | ordinal, categorical | type of paved driveway |
| age | numerical, continuous | age of the property. this feature is engineered from year house was built and the year it was sold |
| total_hse_area | numerical, continuous | total area of the house which includes the ground living area and total basement (sqft). this feature is engineered. |
| total_bathrms | numerical, discrete | this feature is engineered. total number of bathrooms in the house, including basement bathrooms. Half bathrooms represented as 0.5 and full bathrooms represented as 1 |
| ms_zoning | ordinal, categorical | identifies the general zoning classification of the sale |
| garage_type | nominal, categorical | garage location within house compound |
| neighborhood | nominal, categorical | physical locations within Ames city limits |
| condition_2 | nominal, categorical | proximity to various conditions (ie. roads) |
| central_air | nominal, categorical | existence of central air conditioning |
| sale_type | nominal, categorical | type of sale |
| roof_matl | nominal, categorical | type of roof material |

A description of the full dataset is found [here](http://jse.amstat.org/v19n3/decock/DataDocumentation.txt)

## Executive Summary
The objective of this project is to identify three areas of focus that drive house prices in Ames. This is done by building a good predictive model coupled with exploratory data analysis, which can provide insights as to those areas of focus. 

#### Analysis and Findings
[<u>Data Cleaning and Exploratory Data Analysis</u>](code/01-data-cleaning-and-eda.ipynb))is first done on the dataset. It is necessary to first remove outliers per the Data Documentation text that came with the dataset(De Cock, 2010). For missing values, I follow a general rule to drop columns with >60% null values (Berdikulov, 2019) and for the rest of the missing values, perform imputations of median and mode accordingly as appropriate. 

While reviewing each feature, I also consider its interactions with the target variable, Sale Price, and perform feature engineering based on insights from analysis. For basement area, I added both the Basement 1 and 2 area to form the total area, which had a much stronger correlation to the sale price, compared to each one (Figure 1). For the model, I would just take the total basement area and drop the individual features from the model. 

[insert pic for basement]
Figure 1:

Another finding is how the area of any part of the house seem to always have a relatively strong correlation (>0.5) with sale price. When I add the basement area and ground living area, I get the total house area which is highly correlated (0.8) with sale price (Figure 2). This makes logical sense as well as bigger homes are expected to fetch higher prices.

[insert pic for house area]
Figure 2:

Generally for features with little to no correlation (0.5) to sale price, I would exclude them from my model, as these would only introduce noise to my model, and lower its accuracy. Prior to excluding any feature, I would consider its nature and whether there are other features that interact with that feature, which together may result in stronger correlation to sale price. I also exercise judgment and domain knowledge of house purchases. 

Basement quality and condition are two measures of the same factor, from a home buyer's perspective. In other words, I would consider both factors together when deciding between one house or another. As expected, the better the quality and condition of the house, the higher price it can fetch (Figure 3).

[insert basement quality and condition]
Figure 3: 

We can see the same trend for exterior material and overall quality and condition (Figure 4). 
[overall quality and condition]
Figure 4:

One of the most important considerations when purchasing a house is location. We can see that the median sale price across each neighbourhood differ from each other (Figure 5). Some neighbourhoods are able to fetch higher price than others.  

[neighbourhood]
Figure 5:


Once data is cleaned and low correlation or subsumed features are dropped, I check for multicollinearity using a heatmap (Figure 6). Excessive multicollinearity can be a problem (Frost, 2020) as putting both variables through the model would violate the assumptions of linear regression which is that variables must be independent of each other. 

[heatmap]
Figure 6:

From here, I drop the garage cars features as it is strongly positively correlated with garage area (0.9) . In any case, they both reflect the same underlying feature which is the capacity of the garage. 

[<u>Preprocessing and Modelling</u>](code/02-preprocessing-and-modelling.ipynb)
At this stage, I prepare the data for modelling. From the data dictionary, we can see that there are categorical variables - nominal and ordinal. These need to be encoded to be used in the model.For nominal categories, I use pd.get_dummies to one hot encode it, whereas for ordinal categories, I map a rating value according to the scale of the respective features. 

Before applying a regression model, its variables ideally should be normally distributed for us to rely on the coefficients of the variables from the model(Wu, 2020). Seeing that there are variables with skewed distributions, I apply the boxcox1p transformation(Deepnote, 2019) which transforms the skewed data closer to a normal distribution. 

Once transformed, I fit and cross validate in the following four models. Cross validation is done with kfolds = 10, as this is generally optimal (Brownlee, 2020).

I considered two success metrics to determine which model is the best among the rest - R2 score and Root Mean Squared Error(RMSE). The R2 score ranges from 0 to 1. An R2 score that is closer to 1 means that the model is able to account for more variability of data. Between the models, I would pick one that is closest to 1. The RMSE is the root of the sum of square of prediction error, divided by the number of data points. It basically gives us a number on how much the predicted results deviate from the actual number (Wu,2020). For this  metric, the smaller the number would mean the model is better. 

Below the results of the regression for the four models:

| Evaluation | Linear Regression | Ridge Model | Lasso Model | Elastic Net Model |
| :------: | :---------------: | :---------: | :---------: | :---------------: |
| R2 score | -5.902*e^21  | 0.875 | 0.877 | 0.876 |
| RMSE | 26104940455 | 0.836 | 0.832 | 0.830 |

From above, we see that generally regularized models work better. Lasso Model has the best performance, being able to account for 87.7% of variability of data. On the RMSE metric, Elastic Net Model did the best but only slightly better than Lasso model. From here, I go forward with Lasso Model. 

[<u>Model Tuning and Benchmarking</u>](code/03-model-tuning-and-recommendations.ipynb)
With the Lasso Model, I try to improve the model performance by removing zero-coefficient variables and low-impact variables (those with coefficients <0.1). However, it appears that doing both did not improve the R2 score. 

To ascertain how good the model is, I also benchmark it against a baseline model. In deciding the baseline model, I considered using sklearn's DummyRegressor function (Albon, 2017), but it results in an R2 score of -0.016 which is a very low bar to benchmark against. Instead, I go with the Lasso Regression Model (which is the best model selected earlier), and regress the top 5 numeric features with strong correlation with sale price (overall quality, basement quality, exterior quality, total house area and garage area). 

| Evaluation | Lasso Model | Baseline Model (Dummy Regressor) | Baseline Model (Top 5) |
| :------: | :---------------: | :---------: | :-----------: |
| R2 score | 0.877 | -0.016 | 0.836 |
| RMSE | 0.832 | 2.483 | 0.993 |

From here we see that Lasso Model (0.877) is a better fit than our baseline model (0.836) as it can account for more variability in data. Therefore, Lasso Model is our production model.

<u>Model Deployment and Conclusion</u>
From the production model, I found the top 10 features that affect house prices in Ames (Figure 7).

[10 features impact]
Figure 7:

The most important quality of a house is the total area. This makes sense, since the bigger the house, the more expensive it should be. From above, the garage area also comes in third in priority. For a city where cars are an important mode of transport, it is no wonder that they value the area of garage for parking their cars. 

Apart from that, we also see that quality is important. The overall quality of material, kitchen quality, basement quality and exterior quality came in the top 10  most important. 

An interesting insight is also that location also plays a part, with Crawford Neighborhood ranking in the top 10 as well. This generally means that a house in Crawford could fetch a higher price than any other neighbourhoods in Ames. 

#### Recommendations

Considering results of the model and exploratory data analysis, there are three main focus areas that are important in a house : Area, Quality and Condition, and Location.

**Area**
For bigger houses, during sales pitch to potential home buyers, real estate agents should emphasise the large area. For instance, the living area, garage area, basement area.

When speaking to home owners who intend to sell their house, make this known and if their house is objectively small, possibly consider interior architecture that could increase the size of available space. For instance, foldable bed, wall beds or hidden beds, or a couch that could transition to a bed and storage saving spaces. Alternatively, consider other aesthetics to create the illusion of a bigger house. For instance, adding mirrors. These features and changes would be more likely to boost the value of their house. 

**Quality and Condition**
In terms of quality, it appears that there is high value in both form and function. People of Ames look to exterior quality and overally quality of the material, which is the form of the house. During home viewing, the home owner should ensure that exterior forms like the wall, flooring, does not show signs of impairment. 

Function-wise, it makes sense that people would want to buy a house with fixtures that work. For instance, in the kitchen, the stove should be operating well. Should there be anything that is not working, it is wise that the real estate agent advise homeowners to fix it first. Should potential home buyers be aware that some items are not working, it becomes a valid point to lower the house price. This is best avoided. 

**Location** 
Crawford was the only neighbourhood that came up in the top 10 features, among others. This may mean several things. First, that just by having a house in Crawford, one can command a higher price. It is possible that the accessibilty, amenities and facilities available in Crawford allows it to fetch a higher price. Real estate agents should note the base price for Crawford. Second, that each neighborhood is likely to have an attractive factor that boost house prices. It is wise to identify these to better sell a house. 

<u>General Limitations</u>
It is wise to note that there are some limitations of this model. This model is purely based on Ames Housing Data, and hence is applicability is limited to this city in Iowa. While the above insights may be relevant to Ames, it may not apply to other cities in Iowa or other states in the US.

<u>Scaling</u>
Considering the above, with the right data, there are ample opporutnities to build upon this model. I could scale this model down to respective neighbourhoods in cities, and find top features that is most valued by the people in the neighbourhood. There is also potential to scale this upwards, to other cities in Iowa, or even to other states in the US. It would be exciting to see how people's tastes and preferences towards home differ across states. 

## References
Data Documentation - Ames Housing Dataset (De Cock, 2010).
http://jse.amstat.org/v19n3/decock/DataDocumentation.txt

"Dealing with Missing Data"(Berdikulov, 2019) 
https://medium.com/@danberdov/dealing-with-missing-data-8b71cd819501

"When Should I Use Regression Analysis"(Frost, 2020)
https://statisticsbyjim.com/regression/when-use-regression-analysis/

"Is Normal Distribution Necessary in Regression? How to track and fix it? (Wu, 2020)
https://towardsdatascience.com/is-normal-distribution-necessary-in-regression-how-to-track-and-fix-it-494105bc50dd

"How to preprocess data in machine learning?" (Deepnote, 2019)
http://deepnote.me/2019/06/13/how-to-preprocess-data-in-machine-learning/

"How to Configure k-Fold Cross-Validation" (Brownlee, 2020)
https://machinelearningmastery.com/how-to-configure-k-fold-cross-validation/

"3 Best Metrics to evaluate Regression Model" (Wu, 2020)
https://towardsdatascience.com/what-are-the-best-metrics-to-evaluate-your-regression-model-418ca481755b

"Create Baseline Regression Model" (Albon, 2017)
https://chrisalbon.com/machine_learning/model_evaluation/create_baseline_regression_model/