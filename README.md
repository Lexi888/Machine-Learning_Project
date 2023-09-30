# Decision tree, Random forest and XGBoost of customer satisfation

# Project overview:
The purpose of this project is to explore which features are most important to customer satisfaction. Whether a customer is satisfied or dissatisfied belongs to a binary classification task. The dataset comprises survey responses from 129,880 customers. It includes a range of data points such as class, flight distance, and inflight entertainment. Since it's a classification task, I will use decision trees, random forest and XGBoost to accomplish this goal.

# Business understanding:
The term "customer experience" in the airline industry refers to how a customer feels and perceives their journey through the various stages of departure and airport arrival. In this highly competitive industry, every company tries to stand out and win customers, therefore, understanding what customers need and want is really crucial. In order to better understand what customers want the most, we use three different models to analyse the survey responses from the 129,880 customers.

# Data understanding:
The dataset has 22 columns, including diverse aspects of the passengers' experiences, including their ratings for seat comfort, food and beverage services, timeliness, and various other pivotal catagories. However, a noticeable emphasis within the dataset seems to revolve primarily around the interactions with airline personnel, spotlighting the significance of onboard service and customer relations. While this is undoubtedly crucial, it is important to recognize that in this era of evolving travel dynamics, the technological dimension of the passenger journey should not be overlooked. A noteworthy trend, as indicated by recent research conducted by Deloitte, underscores the increasing appetite among travelers for digitization in their travel experiences. By 2023, there is a big share of travelers of all ages becoming more comfortable with blended human/digital touchpoints. 

<img width="544" alt="Screenshot 2023-09-30 at 18 16 21" src="https://github.com/Lexi888/Machine-Learning_Project/assets/98598719/f8c01058-0422-433d-8b6e-b6d79b4fcaa0">


# Modeling and Evaluation
Decision trees require no assumptions regarding the distribution of underlying data, and don't require scaling of features. However, they are susceptible to overfitting. To mitigate this risk, I will also leverage the capabilities of random forests as a second model. This approach is designed to reduce variance, bias and overfitting. GBMs is a boosting methodology where each base learner in the sequences is built to predict residual errors of the model that preceded it. It provides high accuracy and is robust to outliers, but it can be time consuming due to many hyperparameters. Bad extrapolation is another issue, which means that it may have difficulty in predicting new values that fall outside of the range of values in the training data. In order to select a champion model, 'F1' is the major metric I will look at as it is a trade-off between precison and recall. Other than the metrics, we also need to consider the runtime of a model.

<img width="422" alt="Screenshot 2023-09-30 at 18 41 45" src="https://github.com/Lexi888/Machine-Learning_Project/assets/98598719/bee3eaac-2a75-4a06-a975-2115172df3c4">

# Conclusion
Since the champion model is Tuned Random Forest, we will use this model to solve the business problem here.<img width="564" alt="Screenshot 2023-09-30 at 19 17 39" src="https://github.com/Lexi888/Machine-Learning_Project/assets/98598719/27c8cf85-fa3c-48e0-8e84-d1c1a880190b">

By extracting feature importance, we can see that the most four important factors that should be focused on are; inflight entertainment, seat comfort, online booking process and support. The last two parts I believe can be improved by technology, but it needs more research about current online  booking processes and finding out the real problems customers may face.

