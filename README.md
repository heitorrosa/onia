# ONIA
A repository dedicated to study for the 2nd ONIA (Olimpíada Nacional de Inteligência Artificial) and the IOAI (International Olympiad in Artificial Intelligence). This repository contains all documents and resolutions made by me during the iteration of this olympiad.

* Fase 3 Etapa 1: *20/20*
* Fase 3 Etapa 2: ~

---

### Syllabus
* [IOAI Kazakhstan Syllabus](https://ioai-official.org/wp-content/uploads/2025/10/Syllabus.pdf) 
* [2nd ONIA Syllabus](https://www.oniabrasil.com.br/assets/files/Syllabus_da_2_ONIA.pdf) 

### Related repositories and content
* [IOAI-official/IOAI-2024](https://github.com/IOAI-official/IOAI-2024)
* [IOAI-official/IOAI-2025](https://github.com/IOAI-official/IOAI-2025)
* [ioai-writeup/ioai-writeup.github.io](https://github.com/ioai-writeup/ioai-writeup.github.io)
* [open-cu/awesome-ioia-tasks](https://github.com/open-cu/awesome-ioai-tasks)
* [NOIC-IA/Problem-Solutions](https://github.com/NOIC-IA/Problem-Solutions)
* [mgcvale/onia](https://github.com/mgcvale/onia)
* [stefanasandei/roai-solved](https://github.com/stefanasandei/roai-solved)
* [zHary27/machine-learning-problems](https://github.com/zHary27/machine-learning-problems)
* [jaredliw/ioai-tsp-2025](https://github.com/jaredliw/ioai-tsp-2025)
* [babidisrc/introducao-a-ML](https://github.com/babidisrc/introducao-a-ML)

---
* [ONIA 2025 4 Fase Etapa 1](https://www.oniabrasil.com.br/assets/files/2025_05_20_ONIA_4a_fase_1a_etapa_gabarito.pdf)
* [ONIA 2025 4 Fase Etapa 2](https://www.oniabrasil.com.br/assets/files/Gabarito_Prova_2a_etapa_4a_fase_13jun2025.pdf)

### Compiled Study Playlist
[Machine Learning](https://www.youtube.com/playlist?list=PLjn45pXnqU0DY8RDF2FL8PKrAHZlALK2m)

---

## Exercises

<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Category</th>
      <th>Projects & Exercises</th>
      <th>Dataset / Reference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>01</td>
      <td rowspan="26"><b>1. Foundational Skills & Classical Machine Learning</b></td>
      <td><b>01. Temporal Anomaly Detection:</b> As an energy analyst, parse a decade of hourly power consumption data using Pandas and NumPy. Construct a feature pipeline encoding cyclic time variables (sin/cos), followed by an integrated Scikit-Learn pipeline that trains an Isolation Forest or One-Class SVM to robustly detect blackout anomalies.</td>
      <td><a href="https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption">Hourly Energy Consumption</a></td>
    </tr>
    <tr>
      <td>02</td>
      <td><b>02. High-Cardinality Target Encoding:</b> Given the Adult Income dataset featuring heavy categorical variables, implement a highly customized <code>ColumnTransformer</code>. You must seamlessly handle unseen categories during inference, group sparse nominals into an 'other' bucket with Pandas, and construct a robust Logistic Regression baseline optimized exhaustively via <code>GridSearchCV</code>.</td>
      <td><a href="https://www.kaggle.com/datasets/uciml/adult-census-income">Adult Income</a></td>
    </tr>
    <tr>
      <td>03</td>
      <td><b>03. Missing Data & Regularized Recovery:</b> Using the House Prices dataset, architect an iterative imputer employing Ridge Regression to cross-estimate missing property traits. Once repaired, apply a Lasso (L1) regression to dynamically collapse irrelevant features to zero, validating your sparsity constraints across an automated 10-fold cross-validation scheme.</td>
      <td><a href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques">House Prices</a></td>
    </tr>
    <tr>
      <td>04</td>
      <td><b>04. Dimensionality Synthesis & Constraint Modeling:</b> Operating on the high-dimensional Breast Cancer dataset, synthesize non-linear interaction features utilizing <code>PolynomialFeatures</code>. Apply PCA to constrain the dimensions while retaining 95% variance, and establish an <code>SVC</code> pipeline that mathematically optimizes the hyperplane margin under strict recall requirements.</td>
      <td><a href="https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset">Breast Cancer (sklearn)</a></td>
    </tr>
    <tr>
      <td>05</td>
      <td><b>05. Deep Fraud Detection with Autoencoders:</b> Utilizing PyTorch, architect an Autoencoder neural network to detect anomalies in highly imbalanced credit card transaction data. Leverage high-level PyTorch modules to construct the encoder-decoder topology and compute reconstruction loss via MSE, efficiently separating fraudulent anomalous transactions from legitimate distributions.</td>
      <td><a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud">Credit Card Fraud</a></td>
    </tr>
    <tr>
      <td>06</td>
      <td><b>06. Unsupervised Density-Based Clustering (DBSCAN):</b> Provided an unstructured GPS coordinate array, you must engineer a spatial clustering solution from scratch. You cannot rely on pre-defined distance thresholds; instead, you must analytically determine <code>eps</code> using a K-Distance graph and implement <code>DBSCAN</code> leveraging a BallTree or KDTree to isolate arbitrary-shaped noise-heavy noise distributions.</td>
      <td><a href="https://www.kaggle.com/datasets/ahmedmohammad2003/uber-trip-data">Uber Coordinates</a></td>
    </tr>
    <tr>
      <td>07</td>
      <td><b>07. Centroid-Based Manifold Discovery (K-Means++):</b> Tasked with segmenting a high-dimensional customer behavior matrix, implement a K-Means strategy that explicitly optimizes the 'Elbow' and 'Silhouette' metrics. You must proactively handle multi-collinearity and scale variances using iterative <code>StandardScaler</code> applications before projecting the results onto a 2D PCA plane.</td>
      <td><a href="https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python">Mall Customers</a></td>
    </tr>
    <tr>
      <td>08</td>
      <td><b>08. Elastic Net Feature Shrinkage & K-Fold Stratification:</b> Tasked with regressing explicit medical parameters, completely bypass <code>cross_val_score</code>. Construct a native K-Fold stratification index mapping iteratively over Elastic Net equations mathematically bounding exactly L1 vs L2 regularization shrinkage thresholds mapping explicit coefficient paths decaying visually over loop trajectories.</td>
      <td><a href="https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset">Diabetes (sklearn)</a></td>
    </tr>
    <tr>
      <td>09</td>
      <td><b>09. Bayesian Probabilities & Textual Priors:</b> Given a corpus of Spam SMS text, transform raw vocabulary via <code>TfidfVectorizer</code> into sparse matrices. Architect a <code>MultinomialNB</code> model mapping explicit Bayesian priors systematically to counteract severe class imbalances, validating specific probability scores outputted iteratively per class.</td>
      <td><a href="https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset">SMS Spam Collection</a></td>
    </tr>
    <tr>
      <td>10</td>
      <td><b>10. Advanced Decision Tree Pruning & Ensembling:</b> Leveraging the Wine dataset, implement a rigorous Scikit-Learn <code>DecisionTreeClassifier</code>. Rigorously deploy Scikit-Learn's structural constraints, validating <code>ccp_alpha</code> pruning parameters and integrating randomized Forest architectures to analyze structural feature importances efficiently, bypassing entirely from-scratch split manual calculations.</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html">Wine Dataset</a></td>
    </tr>
    <tr>
      <td>11</td>
      <td><b>11. Ensemble Architectures (Voting Classifiers):</b> Working on complex tabular multi-class arrays, construct a heterogeneous <code>VotingClassifier</code> combining Logistic Regression, KNN constraints, and SVM architectures. Rigorously map predicted probability distributions natively when switching between "hard" voting constraints versus uncalibrated "soft" integrations.</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html">Synthetic Classification</a></td>
    </tr>
    <tr>
      <td>12</td>
      <td><b>12. Polynomial Dimensionality & Bias-Variance Validation:</b> Map explicit multi-degree relationships on an auto-mpg scalar array. Systematically loop polynomial expansions scaling linearly through degree arrays using <code>PolynomialFeatures</code> combined with Ridge regression, graphing explicitly validation vs training error to mathematically isolate optimal Bias-Variance constraints.</td>
      <td><a href="https://archive.ics.uci.edu/ml/datasets/auto+mpg">Auto MPG</a></td>
    </tr>
    <tr>
      <td>13</td>
      <td><b>13. Multi-Layered Stacking Regressors:</b> Given explicit dimensional configurations of real estate valuation, project heterogeneous base meta-models mapping explicit Ridge constraints and Tree-based leaf optimizations. Aggregate the subsequent scalar outputs into a final <code>StackingRegressor</code> terminating using Lasso L1 sparsity matrices.</td>
      <td><a href="https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction">Real Estate Valuation</a></td>
    </tr>
    <tr>
      <td>14</td>
      <td><b>14. Support Vector Tube Extrapolations:</b> Using dense sequential historical data mapping temperature variations, architect sequential <code>SVR</code> pipelines predicting non-linear extrapolations. Radically configure the structural epsilon-tube values mapping exact matrix thresholds separating zero-penalty errors from structural margin boundary violations natively in Scikit-Learn.</td>
      <td><a href="https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data">Daily Climate</a></td>
    </tr>
    <tr>
      <td>15</td>
      <td><b>15. Iterative Gradient Boosting & Custom Losses:</b> For a medical diagnostics pipeline using Heart Disease data, the cost of false negatives is extreme. Construct a <code>GradientBoostingClassifier</code> utilizing <code>staged_predict()</code>. You must formulate an approach to wrap the objective function to penalize false negatives severely, combined with an epoch-based early stopping loop.</td>
      <td><a href="https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset">Heart Disease</a></td>
    </tr>
    <tr>
      <td>16</td>
      <td><b>16. XGBoost Hardware Awareness & Sparsity:</b> Deployed in a low-latency anti-fraud environment on imbalanced Credit Card Fraud data. Assemble an XGBoost pipeline natively exploiting memory sparsity. Calibrate <code>scale_pos_weight</code> and extensively tune the tree depth and sub-sampling parameters continuously via <code>RandomizedSearchCV</code>.</td>
      <td><a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud">Credit Card Fraud</a></td>
    </tr>
    <tr>
      <td>17</td>
      <td><b>17. Staged Ensemble Aggregation:</b> Utilizing the Teleco Customer Churn data, structurally design an ensemble. Your objective is to forumlate a Scikit-Learn estimator that leverages <code>warm_start=True</code> on a Random Forest to incrementally append estimators, actively halting when the validation log-loss stabilizes computed iteratively via NumPy.</td>
      <td><a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn">Telco Customer Churn</a></td>
    </tr>
    <tr>
      <td>18</td>
      <td><b>18. Native Categorical Handling Challenge:</b> Tasked with forecasting Flight Delays, directly process string and mixed-type categorical columns without manual One-Hot Encoding. Implement a sophisticated pipeline deploying <code>HistGradientBoostingClassifier</code> or XGBoost's native categorical support, benchmarking memory footprints and execution latency.</td>
      <td><a href="https://www.kaggle.com/datasets/usdot/flight-delays">Flight Delays</a></td>
    </tr>
    <tr>
      <td>19</td>
      <td><b>19. SVM Kernel Subspace Projections:</b> Working with complex spatial distribution clusters, architect parallel Support Vector Machines matrices. Compare native boundary mappings between Polynomial Kernel shifts and deep Radial Basis Function (RBF) projections. You must clean the raw dataset, which contains missing values and non-standardized feature scales, and engineer a <code>ColumnTransformer</code> to handle these quirks before projection.</td>
      <td><a href="https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data">Breast Cancer Wisconsin</a></td>
    </tr>
    <tr>
      <td>20</td>
      <td><b>20. Lazy Learning Matrix Validations (KNN):</b> Predict classification mappings across a densely noisy biological feature array. Construct a <code>KNeighborsClassifier</code> isolating explicit mathematical validation through an active BallTree algorithm optimization sequence. You must perform extensive data cleaning on the raw input, handling outliers and inconsistent labels before tracking inference decay.</td>
      <td><a href="https://www.kaggle.com/datasets/uciml/glass">Glass Classification</a></td>
    </tr>
    <tr>
      <td>21</td>
      <td><b>21. Manifold Discovery in Consumer Data:</b> As a marketing proxy analyzing Mall Customers, you must isolate hidden consumer sub-segments. Discard naive K-Means; construct a pipeline channeling data through t-SNE for 2D topological mapping, heavily utilizing NumPy distance matrices, followed by HDBSCAN to capture core dense clusters accurately.</td>
      <td><a href="https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python">Mall Customers</a></td>
    </tr>
    <tr>
      <td>22</td>
      <td><b>22. Dimensionality-Constrained Dictionary Learning:</b> Using the MNIST visual dataset, bypass deep layer reliance by synthesizing a sparse PCA pipeline intersecting with a Dictionary Learning reconstructor defined in Scikit-Learn. Optimize the dictionary size explicitly to reconstruct and denoise digits corrupted by heavy simulated Gaussian noise.</td>
      <td><a href="https://www.kaggle.com/competitions/digit-recognizer">MNIST</a></td>
    </tr>
    <tr>
      <td>23</td>
      <td><b>23. Probabilistic GMM Density Estimation:</b> Given the Wine Quality array, map the multidimensional chemical distributions applying Gaussian Mixture Models (GMM) with variable covariance types. Generate synthetic wine profiles querying the learned continuous probability distributions natively, bounding outliers using probability thresholds.</td>
      <td><a href="https://www.kaggle.com/datasets/yasserh/wine-quality-dataset">Wine Quality</a></td>
    </tr>
    <tr>
      <td>24</td>
      <td><b>24. Hierarchical Feature Agglomeration:</b> Operating on highly multicollinear sensory data, compute the Spearman rank-order correlations systematically in Pandas/SciPy. Apply Scikit-Learn's Feature Agglomeration to hierarchically fuse tightly correlated features from the raw, uncleaned sensor signals. You must engineer a pipeline to handle noisy transients and missing sensor packets before initiating training.</td>
      <td><a href="https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones">Human Activity Recognition</a></td>
    </tr>
    <tr>
      <td>25</td>
      <td><b>25. Spatial Density Isolation Validation:</b> Process geolocation coordination matrices combining structural DBSCAN logic heavily mapped over a pre-processing UMAP structural reduction layer. Enforce parameters that reject strict spherical topology structures to isolate arbitrary shaped geographical routing matrices.</td>
      <td><a href="https://www.kaggle.com/datasets/ahmedmohammad2003/uber-trip-data">Uber Coordinates</a></td>
    </tr>
    <tr>
      <td>26</td>
      <td><b>26. High-Dimensional PCA & Feature Pipelines:</b> Working directly within the MNIST visual matrices, natively scale a raw pixel feature array. Construct entirely customized pipeline systems deploying Scikit-Learn's <code>PCA</code>, extracting optimal dimensions retaining 98% variances directly into an optimal downstream classifier rather than manual Eigendecomposition loops.</td>
      <td><a href="https://www.kaggle.com/datasets/oddrationale/mnist-in-csv">MNIST</a></td>
    </tr>
    <tr>
      <td>27</td>
      <td rowspan="15"><b>2. Neural Networks & Deep Learning</b></td>
      <td><b>27. Multi-Dimensional Housing MLP:</b> Implement a 3-layer MLP to predict housing prices. Focus on standardizing heterogeneous input features using <code>StandardScaler</code> and optimizing with <code>MSELoss</code>.</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html">CA Housing</a></td>
    </tr>
    <tr>
      <td>28</td>
      <td><b>28. Binary Health Risk MLP:</b> Build a classifier to predict heart disease risk. Implement <code>BCELoss</code> and <code>Sigmoid</code> output activation, handling binary classification thresholds.</td>
      <td><a href="https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset">Heart Disease</a></td>
    </tr>
    <tr>
      <td>29</td>
      <td><b>29. Digit Recognition MLP:</b> Use the MNIST dataset to build a multi-class classifier. Manage 784-pixel input flattening and <code>CrossEntropyLoss</code> for 10-way classification.</td>
      <td><a href="https://www.kaggle.com/competitions/digit-recognizer">MNIST</a></td>
    </tr>
    <tr>
      <td>30</td>
      <td><b>30. Dropout & Overfitting Control:</b> Integrate <code>nn.Dropout</code> layers into a deep MLP. Perform a comparative study on training vs validation accuracy with and without dropout active.</td>
      <td><a href="https://www.kaggle.com/datasets/zalando-research/fashionmnist">Fashion MNIST</a></td>
    </tr>
    <tr>
      <td>31</td>
      <td><b>31. Ablation Study (Activation Functions):</b> Construct parallel models using <code>ReLU</code>, <code>Tanh</code>, and <code>LeakyReLU</code>. Map their respective loss surfaces and convergence speeds.</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html">Synthetic Moons</a></td>
    </tr>
    <tr>
      <td>32</td>
      <td><b>32. Weight Initialization Impact:</b> Compare <code>Xavier/Glorot</code> vs <code>Zero</code> vs <code>Random</code> initialization. Visualize how gradients vanish or explode based on starting weights.</td>
      <td><a href="https://pytorch.org/docs/stable/nn.init.html">PyTorch Init</a></td>
    </tr>
    <tr>
      <td>33</td>
      <td><b>33. Learning Rate Scheduling:</b> Implement <code>StepLR</code> and <code>ReduceLROnPlateau</code>. Document how adaptive scheduling prevents local minima stagnation during MLP training.</td>
      <td><a href="https://www.kaggle.com/c/titanic">Titanic</a></td>
    </tr>
    <tr>
      <td>34</td>
      <td><b>34. Batch Normalization MLP Speed Trial:</b> Insert <code>BatchNorm1d</code> layers between Linear and ReLU. Measure the reduction in epochs required to reach 90% accuracy on tabular data.</td>
      <td><a href="https://www.kaggle.com/datasets/vuppalaadithyasairam/heart-disease-dataset">Health Risk</a></td>
    </tr>
    <tr>
      <td>35</td>
      <td><b>35. L1 vs L2 Sparsity MLP:</b> Compare weight decay (L2) with manual L1 penalty implementation. Visualize the resulting weight histograms to see sparsity effects.</td>
      <td><a href="https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset">Heart Disease</a></td>
    </tr>
    <tr>
      <td>36</td>
      <td><b>36. Early Stopping MLP Integration:</b> Program a custom validation loop that halts training when <code>val_loss</code> stops improving for a "patience" of 10 epochs.</td>
      <td><a href="https://www.kaggle.com/datasets/vuppalaadithyasairam/heart-disease-dataset">Health Risk</a></td>
    </tr>
    <tr>
      <td>37</td>
      <td><b>37. LeNet-5 Standard Digit Classifier:</b> Implement the classic 1998 LeNet-5 architecture (AvgPool, 5x5 kernels) to classify handwritten digits, documenting tensor shape transformations.</td>
      <td><a href="https://www.kaggle.com/competitions/digit-recognizer">MNIST</a></td>
    </tr>
    <tr>
      <td>38</td>
      <td><b>38. AlexNet Feature Extraction Layering:</b> Build the 2012 ImageNet winner with 11x11 kernels and ReLU, classifying RGB dog breeds to demonstrate deep feature hierarchy.</td>
      <td><a href="https://www.kaggle.com/datasets/miljan/stanford-dogs-dataset-tensors">Stanford Dogs</a></td>
    </tr>
    <tr>
      <td>39</td>
      <td><b>39. VGG-16 Deep Block Sequencing:</b> Construct a 16-layer network using small 3x3 filters and modular "VGG Blocks" to classify CIFAR-10 objects with extreme depth.</td>
      <td><a href="https://www.kaggle.com/c/cifar-10">CIFAR-10</a></td>
    </tr>
    <tr>
      <td>40</td>
      <td><b>40. Data Augmentation & Generalization:</b> Integrate a <code>torchvision</code> pipeline (Rotation, Crop, Jitter) to reduce the generalization gap in limited-category food classification.</td>
      <td><a href="https://www.kaggle.com/datasets/chrisfilo/fruit-recognition">Kaggle Fruits</a></td>
    </tr>
    <tr>
      <td>41</td>
      <td><b>41. Transfer Learning Standard Fine-Tuning:</b> Freeze a pre-trained ResNet-18 backbone and replace the FC head to detect Pneumonia from medical X-rays with low data samples.</td>
      <td><a href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia">Chest X-Ray</a></td>
    </tr>
    <tr>
      <td>42</td>
      <td rowspan="18"><b>3. Computer Vision & Advanced Architecture</b></td>
      <td><b>42. Tensor Broadcasting Constraints:</b> Orchestrating an attention routing mechanism for raw embedded indices, exclusively leverage PyTorch tensor properties (via <code>torch.einsum</code> or <code>matmul</code>) to output the scaled dot-product attention mapping. Forbid <code>for</code> loops actively, pushing aggressive computational broadcasting constraints across the GPU.</td>
      <td><a href="https://pytorch.org/docs/stable/tensors.html">PyTorch Tensor Docs</a></td>
    </tr>
    <tr>
      <td>43</td>
      <td><b>43. Activation Gradients & Loss Surface Mapping:</b> Rigorously swap PyTorch network topology internal mapping equations transitioning ReLU, Sigmoid, and Tanh constraint networks tracking performance across identical classification boundaries mapping structural Mean Squared Error (MSE) integrations against Binary Cross Entropy (BCE) constraints visually.</td>
      <td><a href="https://www.kaggle.com/c/titanic">Titanic</a></td>
    </tr>
    <tr>
      <td>44</td>
      <td><b>44. ELBO Matrix Optimization (VAE):</b> Assemble fundamental Variational Autoencoder matrices. Systematically derive PyTorch loss blocks mapping explicit mathematical properties evaluating explicit Evidence Lower Bound (ELBO) integration separating reconstruction boundaries actively off heavily restricted Kullback-Leibler divergence calculations natively.</td>
      <td><a href="https://www.kaggle.com/competitions/digit-recognizer">MNIST</a></td>
    </tr>
    <tr>
      <td>45</td>
      <td><b>45. Low-Rank Tensor Approximations (PEFT):</b> Process heavy generative pre-trained architectures mapping explicitly a customized LoRA layer configuration matrices structuring explicit frozen topological states appending strictly updated low-rank structures significantly controlling validation loss gradients preventing destructive matrix catastrophic failures dynamically.</td>
      <td><a href="https://www.kaggle.com/datasets/nimitmak/medical-qa-dataset">Medical Q&A</a></td>
    </tr>
    <tr>
      <td>46</td>
      <td><b>46. Bellman Equation State Traversal:</b> Formulate sophisticated array matrices mapping dynamically across Markov framework states organizing strict tabular Q-Learning routing frameworks mapping explicitly reward mappings actively scaling dimensional matrices into a generalized PyTorch deep Neural Network processing explicit pixel data constraints evaluating explicitly temporal discounting algorithms.</td>
      <td>OpenAI Gym (CartPole)</td>
    </tr>
    <tr>
      <td>47</td>
      <td><b>47. Proximal Policy Human Alignment:</b> Establish rigid structural optimization parameters mapping RLHF processes tracking explicit mathematical frameworks combining reward mappings systematically mapping policy gradient bounding scaling explicitly restricting structural catastrophic drift mapping complex loss bounding limits validating dynamic output parameters rigorously testing mathematical optimization boundaries.</td>
      <td>Theoretical Exercise</td>
    </tr>
    <tr>
      <td>48</td>
      <td><b>48. Generative Entropy Stochastic Mapping:</b> Manipulate generative frameworks tracking generative configurations adjusting explicitly mapped logit parameters parsing outputs recursively applying structural constraint formulas evaluating parameter stability dynamically tracking Top-K limitations vs Top-P boundaries establishing strict mapping behaviors via explicit Python looping thresholds natively extracting stochastic variations mathematically tracking outputs continuously.</td>
      <td>API / Local LLM</td>
    </tr>
    <tr>
      <td>49</td>
      <td><b>49. Prompt Engineering Optimization Pipeline:</b> Build an explicit automated testing suite iterating LLM prompt variations mathematically. Script structural benchmarks assessing Zero-Shot, Few-Shot, Chain-of-Thought, and Meta-Prompting accuracy variances executing dynamic validations comparing contextual bounds natively isolating exactly which methodology retrieves optimal parameter stability across complex text arrays.</td>
      <td>HuggingFace API / Local LLM</td>
    </tr>
    <tr>
      <td>50</td>
      <td><b>50. Transfer Learning Freezing Limits:</b> Given massive network limits traversing pre-trained arrays isolating exact weight bindings. Extract structural fine-tuning bounds freezing explicit mathematical arrays tracking strictly appended customized mapping modules iterating backpropagation strictly targeting exclusively novel neural mappings analyzing specifically computational parameter variances avoiding topological collapse.</td>
      <td><a href="https://huggingface.co/models">HuggingFace Models</a></td>
    </tr>
    <tr>
      <td>51</td>
      <td><b>51. Architectural Bypass via Skip Connections:</b> Confronting the CIFAR-10 challenge matrix, program a customized ResNet topography from bare PyTorch modules. Project custom residual blocking structures while handling raw, unnormalized image tensors and performing complex data augmentation strategies to prevent over-fitting.</td>
      <td><a href="https://www.kaggle.com/c/cifar-10">CIFAR-10</a></td>
    </tr>
    <tr>
      <td>52</td>
      <td><b>52. Real-Time Object Detection (YOLO Custom Head):</b> Targeting high-speed maritime navigation, implement a YOLO-style (You Only Look Once) detection head. You must process raw image frames, formulate a multi-part loss function (localization, confidence, and class loss), and implement non-maximum suppression (NMS) to eliminate overlapping bounding box proposals from scratch.</td>
      <td><a href="https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataset">Maritime Objects</a></td>
    </tr>
    <tr>
      <td>53</td>
      <td><b>53. Zero-Shot Visual Reasoning (CLIP):</b> Leveraging Contrastive Language-Image Pre-training (CLIP) principles, architect a dual-encoder system. You must align image embeddings with textual label embeddings in a shared latent space, enabling the model to classify unseen objects without explicit categorical training, mapped via cosine similarity.</td>
      <td><a href="https://www.kaggle.com/c/imagenet-object-localization-challenge">ImageNet (Subset)</a></td>
    </tr>
    <tr>
      <td>54</td>
      <td><b>54. ViT - Vision Transformer Architectures:</b> Bypass traditional Convolutional layers to build a Vision Transformer (ViT). Implement patch embedding, positional encoding, and a multi-head self-attention backbone to process images as sequences of tokens, validating performance on high-resolution medical imagery.</td>
      <td><a href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia">Chest X-Ray (Pneumonia)</a></td>
    </tr>
    <tr>
      <td>55</td>
      <td><b>55. Semantic Masking & Region Isolation:</b> Targeting industrial navigation on Cityscapes sequences, define a structural U-Net layout. Enforce expanding and contracting paths alongside matching symmetrical skip linkages. Compute a customized Intersection-over-Union (IoU) differentiable framework guiding the spatial optimization loop.</td>
      <td><a href="https://www.kaggle.com/datasets/dansbecker/cityscapes-image-pairs">Cityscapes</a></td>
    </tr>
    <tr>
      <td>56</td>
      <td><b>56. Self-Supervised Contrastive Formulations:</b> Provided unlabelled patches sourced from ImageNet subsets, formalize a fundamental SimCLR logic layer in PyTorch. Combine randomized geometrical transformations via <code>torchvision</code> translating directly towards an InfoNCE theoretical contrastive loss gradient formulation.</td>
      <td><a href="https://www.kaggle.com/c/imagenet-object-localization-challenge">ImageNet (Subset)</a></td>
    </tr>
    <tr>
      <td>57</td>
      <td><b>57. Multi-Scale Structural Proposals (mAP):</b> Evaluate complex structural arrays contrasting YOLO localized detection regressions dynamically opposed to Mask R-CNN topological instance logic mappings. Actively formulate an analytical evaluation script tracing mathematical Mean Average Precision (mAP) metrics processing explicitly under spatial intersection thresholds.</td>
      <td><a href="https://cocodataset.org/">COCO Dataset</a></td>
    </tr>
    <tr>
      <td>58</td>
      <td><b>58. Bipartite Matching Structural Detections:</b> Overhauling traditional anchoring methods map natively a complete DETR layout mechanism structure implementing direct array matching functions deriving exact Hungarian algorithm matching algorithms across fixed multi-label token mappings mapping bounding parameter regressions efficiently.</td>
      <td><a href="https://www.kaggle.com/datasets/robikscube/pascal-voc-2012">PASCAL VOC</a></td>
    </tr>
    <tr>
      <td>59</td>
      <td><b>59. Generative Adversarial Regularizations:</b> Provided the CelebA landscape, organize a DCGAN integration. Combat persistent mode collapse by designing single-sided label smoothing algorithms paired dynamically with Gaussian noise interference fed aggressively into the localized Discriminator topologies evaluating Inception Scores manually.</td>
      <td><a href="https://www.kaggle.com/datasets/jessicali9530/celeba-dataset">CelebA</a></td>
    </tr>
    <tr>
      <td>60</td>
      <td rowspan="17"><b>4. NLP & Sequence Modeling</b></td>
      <td><b>60. Recurrent State Memory Matrices:</b> For processing variable-length financial sentiments, transform text arrays into tightly packed sequential data batches. Enact an LSTM model processing directly utilizing <code>pack_padded_sequence</code> in PyTorch, dynamically stripping explicit padding tokens, feeding strictly final embedded memory vectors outward to a classification head.</td>
      <td><a href="https://www.kaggle.com/datasets/sbatti/financial-sentiment-analysis">Financial Sentiment</a></td>
    </tr>
    <tr>
      <td>61</td>
      <td><b>61. Attention Mechanism Decoding Frameworks:</b> Intersecting bilingual text translations across Europarl strings, align a raw Seq2Seq framework isolating transformer blocks completely. Execute mathematical blueprints for multiplicative (Luong) alignment weights iteratively computing the temporal focus matrix mappings via Numpy indexing during real-time generation.</td>
      <td><a href="https://www.statmt.org/europarl/">Europarl</a></td>
    </tr>
    <tr>
      <td>62</td>
      <td><b>62. Transformer Encoder Block Distillation:</b> Instructed to forge a solo bidirectional BERT-style computational layer from base PyTorch operations. Formulate explicit queries, keys, and values matrices to project Multi-Head dimensions identically, cascading matrices strictly down layer normalizations probing parameter weight capacity stability directly.</td>
      <td><a href="https://huggingface.co/models">HuggingFace Models</a></td>
    </tr>
    <tr>
      <td>63</td>
      <td><b>63. Tokenization & Byte-Pair Analytics:</b> Completely eliminating tokenizer library APIs, intake an unstructured sequence text via Pandas operations. Architect a definitive Byte-Pair Encoding (BPE) process operating entirely in naive Python loops recursively mapping consecutive pairing frequencies up towards an explicit maximum token dimension restraint.</td>
      <td><a href="https://www.kaggle.com/datasets/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones">Amazon Reviews Corpus</a></td>
    </tr>
    <tr>
      <td>64</td>
      <td><b>64. Continuous Autoregressive Decoding Protocols:</b> Configure a mathematically rigorous Decoder-only GPT sub-block framework routing causal mappings sequentially. Enforce zero attention leakage enforcing masking mechanisms recursively, testing output variability mapping Top-P and Temperature scaling algorithms natively via logits.</td>
      <td><a href="https://www.kaggle.com/datasets/Cornell-University/arxiv">arXiv Summaries</a></td>
    </tr>
    <tr>
      <td>65</td>
      <td><b>65. Continuous Bag-of-Words (CBOW) Spatial Mappings:</b> Tasked with mapping vast unstructured tokens, reconstruct a native Word2Vec architecture purely iterating inside PyTorch neural boundaries. Calculate localized vocabulary contexts defining explicitly embedded lookup properties computing Cosine Similarity matrix projections mapping word boundaries algebraically avoiding GenSim pipelines.</td>
      <td>Any Text Corpus</td>
    </tr>
    <tr>
      <td>66</td>
      <td><b>66. Frequency Domain Topologies:</b> Transcribe unstructured raw waveform signals compiling structurally mapped Mel Spectrogram arrays utilizing librosa/SciPy configurations mapping strictly optimized frame layers. Vectorize outputs cascading deeply across Conv2D mapping structures computing explicit multi-class categorical arrays precisely.</td>
      <td><a href="https://www.kaggle.com/datasets/joserzapata/free-spoken-digit-dataset-fsdd">FSDD</a></td>
    </tr>
    <tr>
      <td>67</td>
      <td><b>67. Contrastive Speech Quantization Masks:</b> Incorporating massive unstructured recording structures, build a Wav2Vec2 mapping structure formulating contrastive topological representations structurally resolving masked hidden boundaries parsing raw wave embeddings mapping mathematically towards optimal signal loss convergences.</td>
      <td><a href="https://www.kaggle.com/datasets/pypiahmad/librispeech-asr-corpus">LibriSpeech</a></td>
    </tr>
    <tr>
      <td>68</td>
      <td><b>68. Weakly Supervised Acoustic Mapping:</b> Investigate weak spatial parameters structurally leveraging Whisper transcription logic configurations tracking dynamic contextual text representations generating multi-dimensional attention states natively binding noise reduction filtering parameters recursively mapping language translation.</td>
      <td>HuggingFace API</td>
    </tr>
    <tr>
      <td>69</td>
      <td><b>69. Multimodal Natural Sound Topologies:</b> Organize structural Qwen-Audio analytical scripts configuring explicitly complex prompt inputs compiling structural language instructions aligning heavily with acoustic spatial vectors extracting non-verbal classification representations rigorously formatting categorical mapping matrices.</td>
      <td>HuggingFace Models</td>
    </tr>
    <tr>
      <td>70</td>
      <td><b>70. Temporal Sequence Alignment (DTW):</b> Operating entirely under raw sequential Numpy structures computing localized auditory matrices tracking Dynamic Time Warping operations. Align differing frequency lengths evaluating pure boundary distances optimizing global alignment bounds natively strictly bounded algorithmically offline.</td>
      <td><a href="https://www.kaggle.com/datasets/joserzapata/free-spoken-digit-dataset-fsdd">FSDD</a></td>
    </tr>
    <tr>
      <td>71</td>
      <td><b>71. Recurrent Acoustic Isolation Sequences:</b> Synthesize basic auditory mapping inputs projecting dimensional matrices traversing Recurrent Neural Network sequences directly. Implement pure linear transformations calculating sequential signal variances matching classification loss functions minimizing noise distributions heavily via mapped states tracking explicitly frame by frame.</td>
      <td><a href="https://www.kaggle.com/datasets/pypiahmad/librispeech-asr-corpus">LibriSpeech</a></td>
    </tr>
    <tr>
      <td>72</td>
      <td><b>72. Physio-Graph: Neural Temporal Graphs:</b> Structure a Graph Neural Network (GNN) mapping patient vital signs as nodes in a temporal relationship graph. Optimize the message-passing layers to predict critical health events from sparse, non-uniformly sampled clinical time-series.</td>
      <td><a href="https://www.kaggle.com/datasets/koki25/physionet-2012-challenge-dataset">PhysioNet 2012</a></td>
    </tr>
    <tr>
      <td>73</td>
      <td><b>73. GAN-Powered Anomaly Synthesis:</b> Deploy a Generative Adversarial Network to synthesize realistic defect patterns in steel manufacturing images. Use the synthetic data to augment a primary segmentation head, significantly improving the Mean Average Precision (mAP) for rare anomaly classes.</td>
      <td><a href="https://www.kaggle.com/c/severstal-steel-defect-detection">Steel Defect Detection</a></td>
    </tr>
    <tr>
      <td>74</td>
      <td><b>74. Hyper-Network Matrix Controllers:</b> Architect a Hyper-Network that dynamically generates the weights for a smaller task-specific network. Evaluate the system's ability to generalize across one-shot character recognition tasks using the Omniglot dataset.</td>
      <td><a href="https://www.kaggle.com/datasets/sainikhileswar/omniglot-dataset">Omniglot (Few-Shot)</a></td>
    </tr>
    <tr>
      <td>75</td>
      <td><b>75. Differentiable Sorting & Ranking:</b> Bypass non-differentiable sorting operations with a Soft-Sort layer. Implement a Learning-to-Rank (LTR) model on web search query indices, optimizing for Normalized Discounted Cumulative Gain (NDCG) using pure gradient descent.</td>
      <td><a href="https://www.kaggle.com/datasets/petezhishuo/mslr-web10k">MSLR Web10K</a></td>
    </tr>
    <tr>
      <td>76</td>
      <td><b>76. Implicit Neural SDF Reconstruction:</b> Design a Multi-Layer Perceptron (MLP) to learn the Signed Distance Function (SDF) of 3D objects. Train the network to reconstruct high-fidelity meshes from point clouds, leveraging implicit neural representations for spatial geometry.</td>
      <td><a href="https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset">ModelNet40</a></td>
    </tr>
  </tbody>
</table>

## License
This repository is under the GPL 3.0 License from June, 29th 2007.