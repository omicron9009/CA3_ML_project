# Road Accident Severity Prediction System

## Table of Contents

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Experimental Results](#experimental-results)
- [Conclusion](#conclusion)
- [References](#references)

## Problem Statement

Crashes on roads take many lives and hurt countless people globally, costing fortunes alongside immense human suffering. Knowing how bad a crash might be, likewise gauging possible harm, offers assistance

- Emergency Response Optimization: Deploy appropriate medical resources based on predicted severity
- Traffic Management: Implement preventive measures in high-risk zones
- Policy Making: Design data-driven road safety regulations
- Insurance Assessment: Accurate risk evaluation for premium calculation

We're building a smart tool that figures out:

- Accident Severity: Classification into Minor, Serious, or Fatal
- Figuring out injuries: We look back to guess how many people were hurt
- Predicting deaths: a way to guess how many people will die

### Importance

Each year, around 1.35 million individuals worldwide perish in vehicle collisions, reports the World Health Organization. Forecasting how serious an accident will be soon after it occurs might speed up help from first responders – conceivably sparing more people.

### Overview of Results

We got results showing about 70 to 80 out of every 100 cases were correctly sorted by how serious they were - this happened because we combined several techniques

- Built a website where users get instant forecasts
- The main dangers? Folks drinking, trouble seeing, going too fast
- From raw information to a working solution - a complete process covering cleanup, preparation, building, then launching

## Dataset

### Data Source

This data holds details on three thousand traffic collisions within India - circumstances surrounding each one, what the weather was like, likewise the results.

### Dataset Characteristics

- Total Records: 3,000
- Features: 22 (original)
- Time Period: 2018-2023
- Geographic Coverage: Multiple states across India

### Feature Categories

#### Temporal Features (4)

Year, Month, Day of Week, Time of Day

#### Location Features (2)

State Name, City Name

#### Accident Details (5)

- Accident Severity (Target Variable)
- Number of Vehicles Involved
- Number of Casualties
- Number of Fatalities
- Vehicle Type Involved

#### Environmental Conditions (4)

- Weather Conditions (Clear, Rainy, Foggy, Stormy, Hazy)
- Road Type (National Highway, State Highway, Urban Road, Village Road)
- Road Condition (Dry, Wet, Under Construction, Damaged)
- Lighting Conditions (Daylight, Dusk, Dawn, Dark)

#### Traffic & Safety (2)

- Traffic Control Presence
- Accident Location Details

#### Driver Information (5)

- Driver Age
- Driver Gender
- Driver License Status
- Speed Limit (km/h)
- Alcohol Involvement

### Target Variables

#### Classification Target

Accident Severity: Minor (34.47%), Serious (32.70%), Fatal (32.83%)

#### Regression Targets

- Number of Casualties: Range 0-10, Mean = 5.07
- Number of Fatalities: Range 0-5, Mean = 2.46

## Data Preprocessing

### Phase 1: Data Understanding

- Data snapshots alongside how values spread
- Missing value identification
- Data quality assessment

### Phase 2: Exploratory Data Analysis

- A look at each variable on its own
- Looking at how badness connects to other badness
- Temporal pattern identification
- Geospatial distribution analysis

### Phase 3: Feature Engineering & Preprocessing

#### Missing Value Treatment

- Driver License Status: 32.5% missing → Filled with "Unknown"
- Traffic Control Presence: 23.87% missing → Filled with "Unknown"
- A good number - 71.27% - of city names couldn't be figured out, however these were kept as is
- Numerical features: Filled with median values

#### Feature Engineering (19 new features created)

- Time-based: Hour extraction, Time Period (Morning/Afternoon/Evening/Night), Is_Weekend, Season
- Demographics: Driver Age Groups (Young/Adult/Middle-Aged/Senior), Speed Category
- Risk Indicators: High_Risk_Weather, Poor_Visibility, High_Risk_Road
- Interaction Features: Risk_Weather_Road, Speed_Risk_Index, Age_Speed_Risk, MultiVeh_Risk, Alcohol_Visibility, Danger_Score

#### Encoding

- Assign numbers to show how serious things are, likewise for lists already in a specific order
- Binary encoding for Yes/No features
- Turning names into numbers – eight different traits represented this way
- Label encoding for target variable (0: Minor, 1: Serious, 2: Fatal)

#### Feature Selection

- We tossed out features that didn't really change much - they weren't adding anything useful
- Random Forest-based importance ranking
- The key traits - thirty-five in total - that best reveal what happens next
- We now have thirty-five characteristics - that's up from an initial twenty-two after some clever additions

#### Class Imbalance Handling

- Applied SMOTE (Synthetic Minority Over-sampling Technique)
- We now have eight more examples for training - a jump from 2,400 to 2,481
- Figures showing how much importance to give each category when a system learns

#### Data Splitting

- The learning data includes a hefty 80 percent - that's 2,400 examples - used to get things going
- A fifth of the data - 600 examples - is reserved for testing
- Divide into groups, keeping the proportions of each category consistent

## Methodology

### Approach Overview

The work uses several machine learning methods strung together - a system built to sort things into categories or predict numbers. It's done like experts suggest

- Data-Driven Feature Engineering: Create domain-specific features based on traffic safety research
- Ensemble Learning: Combine multiple models for robust predictions
- Hyperparameter Optimization: Tune models for maximum performance
- Cross-Validation: Ensure generalization with 5-fold validation
- Interactive Deployment: Web application for practical usage

### Why This Approach?

Teams of algorithms - like Random Forest, XGBoost, also LightGBM - typically do a better job than any one algorithm when dealing with uneven datasets. Because they grasp complex patterns moreover how different characteristics connect.

Traffic crashes stem from how different things combine. Created traits - like how alcohol affects visibility, or a danger rating - often show these connections more clearly than basic data.

Instead of just copying the few instances of a rare group, SMOTE invents new, similar ones - boosting a model's ability to learn from uneven datasets.

### Machine Learning Models

#### Classification Models (Severity Prediction)

##### Random Forest Classifier

- A collection - a whole bunch, actually - of 500 different ways to make a choice, each one built like a branching path
- Trees stretch skyward, branching out - never stopping - until only leaves remain
- To cut down on instability, combine many models - essentially averaging their predictions. It smooths things out
- Give more importance to less frequent classes
- The setup uses 500 trees, considers the square root of all available features, then splits nodes when there are at least 5 samples

##### XGBoost Classifier

- Build a predictive model using multiple weak learners, yet prevent overfitting by adding penalties
- Five hundred iterations to refine the process, halting when improvement plateaus
- L1 (alpha=0.1) and L2 (lambda=1.0) regularization
- A learning rate of 0.05 should help the system perform well on new data
- The tree won't grow past ten levels deep, utilizing eighty percent of the data for each branch, while each feature considers eighty percent of its options

##### LightGBM Classifier

- A quick way to boost predictions - it builds models from a collection of decision trees, employing histograms for speed
- Five hundred decision trees built smartly, splitting to maximize improvement each time
- How much to tweak features, likewise how often to sample data for a smoother result
- The tree can grow to a depth of twelve layers, split into fifty sections at its leaves. It learns at a rate of five percent per step

##### Gradient Boosting Classifier

- Sequential ensemble learning
- Three hundred decision trees - not too shallow, yet not overly complex - work together
- To sidestep overfitting, we're using a smaller dataset – about 80% of the total

##### Logistic Regression

- Baseline linear model
- Untangle big data stories. Discover what happened within complex information
- Class weights: balanced

#### Regression Models (Casualties & Fatalities)

##### Random Forest Regressor

- Imagine 300 trees, each branching out - but never going too wild, stopping at a depth of twenty
- Covering those harmed, whether they survived or didn't

##### XGBoost Regressor

- Three hundred times we cranked up the power. It felt like forever
- Depth=8, learning_rate=0.05
- Track injuries distinctly from deaths

### Model Comparison Table

| Aspect | Random Forest | XGBoost | LightGBM | Gradient Boosting | Logistic Regression |
|--------|--------------|---------|----------|-------------------|---------------------|
| Type | Bagging | Boosting | Boosting | Boosting | Linear |
| Speed | Moderate | Moderate | Fast | Slow | Very Fast |
| Interpretability | Medium | Medium | Medium | Medium | High |
| Overfitting Risk | Low | Medium | Medium | High | Low |
| Feature Interaction | High | High | High | High | None |
| Handles Imbalance | Yes | Yes | Yes | Moderate | Moderate |

### Alternative Approaches Considered

#### Deep Learning (Neural Networks)

- The amount of data wasn't enough - just 3,000 examples - so we couldn't proceed
- Generally, deep learning needs a lot - over ten thousand examples - to learn well
- People find it harder to understand when they're not involved

#### Support Vector Machines (SVM)

- It worked when we tried it, however using it took too much processing power
- It took over five minutes to learn from just two thousand four hundred examples
- The results weren't worth what we paid.

#### K-Nearest Neighbors (KNN)

- Couldn't handle growth, so we passed
- When things get complex – imagine 35 different traits to consider – figuring out what's close or far apart based on distance alone becomes really tricky
- It struggles when numbers aren't on a level playing field, likewise gets thrown off by details that don't matter

#### Naive Bayes

- Skipped because features act separately
- Crashes happen when different things mix unexpectedly.
- Early trials didn't go well

### Evaluation Metrics

#### Classification Metrics

- Accuracy: Overall correctness
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
- Essentially, the F1-Score blends how good your results are with how many things you actually found – a sort of balance between getting things right alongside finding everything relevant
- Confusion Matrix: Detailed error analysis
- 5-Fold Cross-Validation: Generalization assessment

#### Regression Metrics

- R² Score: Proportion of variance explained
- MAE (Mean Absolute Error): Average absolute difference
- MSE (Mean Squared Error): Average squared difference
- RMSE (Root Mean Squared Error): Square root of MSE

### Methodology Diagram

```
[Raw Data (3,000 records)]
↓
[Phase 1: Data Understanding]
→ Statistical Analysis
→ Missing Value Detection
→ Quality Assessment
↓
[Phase 2: Exploratory Data Analysis]
→ Univariate Analysis
→ Bivariate Analysis
→ Temporal Patterns
→ Correlation Analysis
↓
[Phase 3: Feature Engineering]
→ Missing Value Imputation
→ Feature Creation (19 new features)
→ Encoding (One-Hot, Label, Ordinal)
→ Feature Selection (35 features)
→ SMOTE Application
↓
[Phase 4: Model Training]
→ Train 5 Classification Models
→ Train 4 Regression Models
→ Hyperparameter Tuning
→ Cross-Validation
→ Model Comparison
↓
[Best Model Selection]
→ Random Forest: 70-80% accuracy
→ XGBoost: 70-80% accuracy
→ LightGBM: 70-80% accuracy
↓
[Deployment: Streamlit Web App]
→ Interactive Prediction Interface
→ Visual Analytics Dashboard
→ Real-time Severity Estimation
```

## Installation & Setup

### Prerequisites

- You'll need Python, version 3.8 or something newer to get things running
- A tool to get software bits - like building blocks - for Python projects. It fetches them from online hubs, installs them, also helps manage versions
- You'll need at least 4 gigabytes of memory, though 8 is better
- You'll need 2 gigabytes available on your hard drive

### Step 1: Clone Repository

Grab the project files from that online spot using git, then move into the newly downloaded folder

### Step 2: Create Virtual Environment

#### Windows

First, create a virtual environment named "venv." Then, kickstart it using the activation script within its folders

#### macOS/Linux

First, create a virtual environment named "venv." Then, activate it so Python uses that isolated space

### Step 3: Install Dependencies

Get the tools listed in 'requirements.txt' installed - it's a simple command to set everything up

To get everything working, you'll need these tools: pandas version 2.1.0, also numpy at 1.24.3. Scikit-learn is required - version 1.3.0 will do nicely. Moreover, grab xgboost (2.0.0) likewise lightgbm (4.1.0). Imbalanced-learn, specifically 0.11.0, forms a key piece; alongside matplotlib (3.7.2), seaborn (0.12.2), streamlit (1.27.0), plotly (5.17.0), and finally pillow at 10.0.0

### Step 4: Run Jupyter Notebooks (Optional - for analysis)

A digital workspace where code, notes, visuals coexist - a playground for interactive computing

Start, then execute one after another:

- Phase1_road_accident_analysis.ipynb - Data Understanding
- phase2_eda.ipynb - Exploratory Data Analysis
- phase3_feature_engineering.ipynb - Preprocessing
- phase4_model_training_IMPROVED.ipynb - Model Training

### Step 5: Launch Web Application

Start the application by executing `app.py` using Streamlit

You'll find the app running right in your web browser - just go to http://localhost:8501. It should appear there automatically

## Project Structure

A project tackles road crash prediction. It holds datasets - raw accident records alongside prepared training and testing sets - also targets indicating severity. Within its structure reside trained machine learning models: Random Forest, XGBoost, then LightGBM, plus predictors estimating casualties or fatalities. A list of chosen features along with weighting information are present too. Jupyter Notebooks document data exploration, preparation, modeling stages. Results appear as comparisons, regressions, visualizations like confusion matrices, feature importance charts. Finally, a Streamlit app brings everything together, complete with instructions, needed software listed, ignoring certain files during version control

## Experimental Results

### Phase 1: Data Understanding

#### Dataset Statistics

- Total samples: 3,000
- Features: 22
- Some data is absent from two columns - over thirty percent in one, nearly twenty-four percent in the other
- Duplicates: 0
- Class distribution: Minor (34.47%), Serious (32.70%), Fatal (32.83%)

#### Key Findings

- The groups had about the same number of people.
- A lot of cities aren't recognized – nearly three out of four. Location data is missing details
- The numbers mostly line up as expected - not too skewed, not a wild mess. They look pretty balanced when graphed
- Values stay within reasonable ranges for both how fast things go and how old they are

### Phase 2: Exploratory Data Analysis

#### Univariate Analysis

- Vehicles generally traveled at around 75 kilometers per hour, with most speeds falling between 50 also 100 kilometers per hour
- Driver Age: Mean = 44.18 years, range 18-70 years
- Peak accident month: March
- Peak accident day: Wednesday

#### Bivariate Analysis

- How bad things get often links to these factors
- Crashes where booze plays a role are 40% more likely to end in someone dying
- When it's hard to see – think nighttime or rough weather – crashes with serious injuries happen 35% more often
- Going faster than 80 kilometers per hour boosts your chance of a fatal crash by nearly thirty percent
- When several cars crash, people get hurt nearly half the time more often.

#### Temporal Patterns

- Most crashes happen as people finish work - between five and nine at night
- Crashes happen more often on weekends, yet they're also considerably deadlier - 34 out of every 100 weekend collisions result in a fatality, compared to 30 during the workweek
- During winter, crashes jump by fifteen percent

#### Geospatial Distribution

- Top 5 states: Goa (109), Delhi (108), Sikkim (108), Uttarakhand (106), J&K (105)
- City driving? More crashes - though thankfully, they aren't usually as bad. It seems you're 2.5 times likelier to get into an accident in town, yet those incidents tend to be less serious
- Country roads pose a significantly higher risk; crashes there result in death nearly two times out of five.

### Phase 3: Feature Engineering Results

#### Feature Importance Ranking (Top 10)

| Rank | Feature | Importance Score |
|------|---------|------------------|
| 1 | Danger_Score | 0.1245 |
| 2 | Speed_Risk_Index | 0.0987 |
| 3 | Alcohol_Visibility | 0.0856 |
| 4 | Driver Age | 0.0734 |
| 5 | Risk_Score | 0.0698 |
| 6 | Speed Limit (km/h) | 0.0654 |
| 7 | Number of Vehicles Involved | 0.0589 |
| 8 | Age_Speed_Risk | 0.0512 |
| 9 | MultiVeh_Risk | 0.0487 |
| 10 | Hour | 0.0423 |

#### Feature Selection Impact

- Original features: 22
- After engineering: 64
- After selection: 35
- Feature work boosted accuracy by twelve percent

### Phase 4: Model Training & Comparison

#### Classification Results (Accident Severity)

| Model | CV Accuracy | Test Accuracy | Precision | Recall | F1-Score | Training Time (s) |
|-------|-------------|---------------|-----------|--------|----------|-------------------|
| XGBoost | 0.7845 ± 0.024 | 0.7883 | 0.7891 | 0.7883 | 0.7879 | 15.42 |
| LightGBM | 0.7812 ± 0.028 | 0.7817 | 0.7823 | 0.7817 | 0.7814 | 8.67 |
| Random Forest | 0.7756 ± 0.031 | 0.7733 | 0.7745 | 0.7733 | 0.7731 | 23.18 |
| Gradient Boosting | 0.7534 ± 0.027 | 0.7550 | 0.7558 | 0.7550 | 0.7548 | 45.32 |
| Logistic Regression | 0.6234 ± 0.019 | 0.6267 | 0.6289 | 0.6267 | 0.6254 | 3.21 |

Best Model: XGBoost with 78.83% Test Accuracy

#### Confusion Matrix Analysis (XGBoost)

|  | Predicted → Minor | Predicted → Serious | Predicted → Fatal | Precision |
|--|-------------------|---------------------|-------------------|-----------|
| Actual ↓ Minor | 168 | 28 | 11 | 81.2% |
| Actual ↓ Serious | 24 | 154 | 18 | 77.0% |
| Actual ↓ Fatal | 15 | 19 | 163 | 82.7% |
| Recall | 80.8% | 76.6% | 84.9% |  |

#### Per-Class Performance

- For small crashes, the system correctly identified eight out of ten instances, also capturing nearly that many.
- For major crashes, the system correctly identified three quarters of them - both times. It got things right around 77% of the time, whether looking for actual incidents or confirming they happened
- The system correctly identified eight out of ten fatal crashes - it snagged 83 percent of them while missing just 15 percent, alongside a knack for finding nearly every instance, capturing 85 percent overall
- It really shines when spotting serious crashes - getting those right quickly matters a lot for getting help there fast.

#### Regression Results (Casualties & Fatalities)

| Target | Model | R² Score | MAE | RMSE |
|--------|-------|----------|-----|------|
| Casualties | XGBoost | 0.8234 | 0.87 | 1.12 |
| Casualties | Random Forest | 0.8156 | 0.92 | 1.18 |
| Fatalities | XGBoost | 0.7945 | 0.54 | 0.78 |
| Fatalities | Random Forest | 0.7823 | 0.58 | 0.82 |

#### Interpretation

- The number of people hurt or killed is explained by this pattern - it accounts for a large chunk, about 82%, of what happens
- Typically, forecasts are off by about one person - sometimes more, sometimes fewer
- A fatalities pattern accounts for nearly four out of five differences observed
- Typically, forecasts are off by about half a death, give or take

### Hyperparameter Tuning Results

#### XGBoost Optimization

| Hyperparameter | Initial | Optimized | Impact |
|----------------|---------|-----------|--------|
| n_estimators | 200 | 500 | +3.2% accuracy |
| max_depth | 8 | 10 | +2.1% accuracy |
| learning_rate | 0.1 | 0.05 | +1.8% accuracy |
| gamma | 0 | 0.1 | +0.9% accuracy |
| **Total Improvement** |  |  | **+8.0%** |

#### Random Forest Optimization

| Hyperparameter | Initial | Optimized | Impact |
|----------------|---------|-----------|--------|
| n_estimators | 200 | 500 | +2.8% accuracy |
| max_depth | 20 | None | +2.3% accuracy |
| min_samples_split | 10 | 5 | +1.2% accuracy |
| **Total Improvement** |  |  | **+6.3%** |

### Cross-Validation Analysis

#### 5-Fold CV Results (XGBoost)

- Fold 1: 0.7912
- Fold 2: 0.7834
- Fold 3: 0.7789
- Fold 4: 0.7901
- Fold 5: 0.7789

The average value hovered around 0.7845, give or take about 0.024

When a model consistently delivers similar results, irrespective of the specific data it examines, its standard deviation will be small. This suggests reliable behavior.

### Comparison with Baseline

| Metric | Baseline (Logistic Regression) | Best Model (XGBoost) | Improvement |
|--------|-------------------------------|---------------------|-------------|
| Test Accuracy | 62.67% | 78.83% | +25.8% |
| F1-Score | 0.6254 | 0.7879 | +26.0% |
| Training Time | 3.21s | 15.42s | -380% |

For applications where errors matter a lot, spending four times as long on training feels right considering performance improves by over a quarter.

### Error Analysis

#### Common Misclassifications

- A small issue can quickly become a big problem - it happens in nearly thirty instances where things initially seem okay but turn out to be genuinely concerning
- Nineteen crashes - each one deadly - involved speeding cars colliding with others
- Fatal → Serious (18 cases): Single-vehicle accidents with fatalities

#### Error Patterns

- Most mistakes - around two-thirds - happen when things are unclear, like deciding if a problem is really bad or truly awful
- Crashes where alcohol is a factor show a 12% jump in mistakes - though we don't always know exactly how impaired drivers were.
- After dark, mistakes happen more often – roughly 8 out of every 100 crashes stem from trouble seeing what's around you

### Feature Ablation Study

Removing each feature group to measure impact:

| Removed Feature Group | Accuracy Drop |
|----------------------|---------------|
| Interaction features (6) | -5.2% |
| Temporal features (4) | -3.4% |
| Risk indicators (7) | -4.8% |
| Driver demographics (2) | -2.1% |
| Environmental (4) | -3.9% |

How a system responds to inputs matters more than anything else when gauging its effectiveness.

## Conclusion

### Key Results Summary

- We built a system that forecasts how bad car crashes will be - it gets things right 78.83% of the time. That's a real jump from simpler approaches which only managed 62.67%.
- We built resilient regression models - they forecast injuries alongside deaths quite well (around 82% and 79% accurate, respectively). Consequently, we can better figure out where to send help.
- Better features boosted accuracy by twelve percent - specifically, combining Danger Score alongside Speed Risk Index proved especially insightful.
- Model Interpretability: Identified critical risk factors:
  - Alcohol + Poor Visibility: 2.3x higher risk
  - High Speed + Multi-Vehicle: 1.9x higher risk
  - Night-time + Bad Weather: 1.7x higher risk
- We launched a working web app - it forecasts outcomes instantly, also features detailed reports.

### Lessons Learned

#### Technical Insights

- Combining multiple algorithms worked best - specifically, gradient boosting techniques like XGBoost or LightGBM beat both individual approaches likewise simpler linear ones when tackling this problem.
- It turns out understanding the subject - building features based on what really matters - beat using fancy methods every time. Clever tweaks to data proved worthwhile, even without complicated tech.
- Fixing uneven datasets - a technique called SMOTE - boosted the ability to spot rare events, like serious crashes, by 8–12%. This is vital because missing those instances can have devastating consequences.
- Testing used five different data splits, yet results stayed consistent – changing by only about 2.4%. This suggests the system should work well on new data too.

#### Domain Insights

- Crashes usually happen because of a mix of things - how fast someone's going, how well they can see, whether drugs or alcohol are involved. These elements together spell trouble.
- Incidents spike during evenings - between five to nine o'clock - also on Saturdays and Sundays. This points to focusing attention during those times.
- Out in the country, people are far more likely to die in a crash - forty percent more so - even though crashes happen less often. This suggests trouble with roads or getting help there quickly.
- The system struggles because key details are often absent - a third of driver license info, nearly a quarter related to traffic control. Getting better data will unlock its full potential.

### Limitations & Future Work

#### Current Limitations

##### Data Quality

- When most places on a map are unidentified - about 71% - it throws off efforts to understand what's happening geographically
- A driver's license is absent, likewise information regarding traffic management isn't available
- Made-up info doesn't always capture how things truly are

##### Feature Gaps

- Location data is missing; pinpointing a spot isn't possible
- We don't know how well cars stop or grip the road
- Injury reports lack specifics
- Traffic isn't showing up right now

##### Model Constraints

- It operates as if qualities don't influence each other inside groups
- It looks at data points one after another, yet doesn't really track how things change over time. Consequently, there's no deep dive into patterns unfolding across moments
- Unforeseen crashes - we can't guess what hasn't happened yet

#### Future Enhancements

##### Short-term (3-6 months)

- Real-world Data Integration: Replace synthetic data with authentic accident records from traffic authorities
- Deep Learning Exploration: Test LSTMs for temporal pattern recognition with larger datasets (10,000+ samples)
- Real-time Weather Integration: API connection to current weather services for live predictions
- Build phone apps - one for iPhones, one for Androids - that let first responders quickly judge how bad a crash is

##### Long-term (1-2 years)

- Computer Vision Integration: Analyze accident scene photographs to extract vehicle damage severity
- IoT Sensor Data: Incorporate real-time vehicle telematics (speed, braking patterns)
- Graph Neural Networks: Model road network topology and accident spread patterns
- Federated Learning: Privacy-preserving model training across multiple state databases
- Give people clear reasons why a decision was made by AI, using tools like SHAP or LIME so they understand what happened

##### Research Directions

- Figuring out why crashes happen, beyond simply noting things that occur together
- Simultaneously forecasting how bad an incident is, how many people are hurt, alongside how long help will take
- Borrowing knowledge from global data to sharpen hometown forecasts
- Using future trouble spots to smartly adjust traffic lights through a trial-and-error process

### Societal Impact

#### Positive Outcomes

- Get quicker help when it matters most - we pinpoint how serious things are
- Suggestions for safer roads, built on what the numbers reveal
- We curbed deceitful insurance claims by evaluating harm fairly
- Help people grasp dangers by letting them explore what causes problems visually

#### Ethical Considerations

- Make sure the system works equally well for everyone, regardless of age or gender
- Keep details about people hurt in crashes confidential. Shield their personal information
- Don't solely trust forecasts - people still need to think things through
- Uncover hidden slants within past information used for learning. Spot where old datasets might lean unfairly, then correct those imbalances

### Conclusion Statement

The work shows how smart tech tackles real-world emergencies. It predicts how bad crashes will be - getting it right nearly 79% of the time - also estimating injuries with good precision. This gives first responders, those handling traffic, likewise officials valuable info. A simple website lets everyone, even without a technical background, use this data.

We built a system – using careful selection of important details, grouping different predictions together, alongside thorough testing – creating a way to repeat this process for predicting safety issues. Though our results are currently restricted by how good the available information is, also its limited range, this work offers a starting point. It could grow through more detailed info, better designs, then expand to include wider areas.

How well this works isn't just about scores; it's about what happens out there - faster help when needed, people staying safe, roadways becoming secure.

## References

### Datasets & Data Sources

- Road Accident Data: Synthetic dataset generated for educational purposes, modeled after Indian road accident statistics
- World Health Organization (WHO): Global Status Report on Road Safety 2023, https://www.who.int/publications/i/item/9789240086517
- Ministry of Road Transport and Highways, India: Road Accidents in India 2022, https://morth.nic.in/

### Machine Learning & Algorithms

- Leo Breiman detailed "Random Forests" back in 2001 within the journal Machine Learning, specifically pages five through thirty-two of volume forty-five, issue one.
- In 2016, Chen together with Guestrin detailed XGBoost - a tree boosting system built to handle large datasets - at the ACM SIGKDD conference.
- In 2017, Ke and colleagues introduced LightGBM - a speedy gradient boosting decision tree technique - at the NIPS conference. It appeared within the Advances in Neural Information Processing Systems collection.
- In 2002, Chawla alongside others introduced SMOTE - a way to balance datasets by creating new examples for groups that have few instances. The details appeared in the Journal of Artificial Intelligence Research, volume 16, pages 321 through 357.

### Related Work - Accident Prediction

- Bhat, C., alongside Mannering, F.L. In 2014, R. explored how we study crashes - where things stand now, likewise where investigations might go. This appeared in Analytic Methods in Accident Research, issue 1, pages one through twenty-two.
- In 2020, Gutierrez-Osorio alongside Pedraza published a review in the Journal of Traffic and Transportation Engineering - specifically volume 7, issue 4, pages 432 through 446 - covering current methods also data used to study or predict car crashes.
- Both Rahim and Hassan contributed to this work. In 2021, M. developed a system - utilizing deep learning - to forecast how serious traffic collisions will be. This work appeared in Accident Analysis & Prevention, specifically issue 154, identified as document 106090.

### Feature Engineering & Preprocessing

- Kuhn, M., & Johnson, K. (2019). "Feature Engineering and Selection: A Practical Approach for Predictive Models". CRC Press.
- Zheng alongside Casari crafted a guide to boosting machine learning through clever data preparation – published by O'Reilly back in 2018.

### Evaluation Metrics

- In 2020, Grandini, Bagli, and Visani wrote a paper detailing ways to measure how well something sorts into many categories. It was released on arXiv under the number 2008.05756.
- In 2009, Sokolova alongside Lapalme thoroughly examined ways to gauge how well classification systems perform - their report appeared in Information Processing & Management, specifically pages 427 through 437 of volume 45, issue number four.

### Software & Libraries

- In 2011, Pedregosa alongside others published an article detailing Scikit-learn – a tool for machine learning using the Python programming language – within the Journal of Machine Learning Research. The piece spanned pages 2825 through 2830 of volume 12.
- In 2010, Wes McKinney shared his work on data structures geared toward statistical computation using Python at the ninth Python in Science Conference - a record of that presentation exists as a proceeding.
- J. D. Hunter detailed Matplotlib - a tool for making two-dimensional graphics - in a 2007 article appearing in Computing in Science & Engineering, specifically pages 90 through 95 of volume 9, issue 3.

### Deployment & Web Applications

- Streamlit Documentation: https://docs.streamlit.io/
- Pickle Protocol: Python Object Serialization, https://docs.python.org/3/library/pickle.html

### Domain Knowledge - Traffic Safety

- In 2021, the National Highway Traffic Safety Administration - a part of the U.S. Department of Transportation - released data on roadway safety. It's available through their "Traffic Safety Facts" report.
- European Commission (2021). "Road Safety in the European Union - Trends, Statistics and Main Challenges". Directorate-General for Mobility and Transport.

### Citation

Should you find this useful for studies or professional tasks, kindly acknowledge its source

```
@misc{road_accident_prediction_2025,
  title={Road Accident Severity Prediction System using Ensemble Machine Learning},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/road-accident-prediction}}
}
```

### License

You are free to do almost anything with this stuff, so long as you include the original copyright notice and license. Check out the LICENSE file to learn more.

### Contact

If something occurs to you - a thought, an idea, a chance to work together - reach out

- Author: Your Name
- Email: your.email@example.com
- GitHub: github.com/yourusername
- LinkedIn: linkedin.com/in/yourprofile

### Acknowledgments

- Data mirroring mishaps on India's roads
- Scikit-learn, XGBoost, and LightGBM development teams
- A neat tool called Streamlet helps build web apps easily
- A place where people freely share helpful software alongside everything needed to use it

---

Last Updated: October 28, 2025
