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
      <th>Category</th>
      <th>Projects & Exercises</th>
      <th>Dataset / Reference</th>
    </tr>
  </thead>
  <tbody>
        <tr>
      <td rowspan="28"><b>1. Foundational Skills & Classical Machine Learning</b></td>
      <td><b>Temporal Anomaly Detection:</b> As an energy analyst, parse a decade of hourly power consumption data using Pandas and NumPy. Construct a feature pipeline encoding cyclic time variables (sin/cos), followed by an integrated Scikit-Learn pipeline that trains an Isolation Forest or One-Class SVM to robustly detect blackout anomalies.</td>
      <td><a href="https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption">Hourly Energy Consumption</a></td>
    </tr>
    <tr>
      <td><b>High-Cardinality Target Encoding:</b> Given the Adult Income dataset featuring heavy categorical variables, implement a highly customized <code>ColumnTransformer</code>. You must seamlessly handle unseen categories during inference, group sparse nominals into an 'other' bucket with Pandas, and construct a robust Logistic Regression baseline optimized exhaustively via <code>GridSearchCV</code>.</td>
      <td><a href="https://www.kaggle.com/datasets/uciml/adult-census-income">Adult Income</a></td>
    </tr>
    <tr>
      <td><b>Missing Data & Regularized Recovery:</b> Using the House Prices dataset, architect an iterative imputer employing Ridge Regression to cross-estimate missing property traits. Once repaired, apply a Lasso (L1) regression to dynamically collapse irrelevant features to zero, validating your sparsity constraints across an automated 10-fold cross-validation scheme.</td>
      <td><a href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques">House Prices</a></td>
    </tr>
    <tr>
      <td><b>Dimensionality Synthesis & Constraint Modeling:</b> Operating on the high-dimensional Breast Cancer dataset, synthesize non-linear interaction features utilizing <code>PolynomialFeatures</code>. Apply PCA to constrain the dimensions while retaining 95% variance, and establish an <code>SVC</code> pipeline that mathematically optimizes the hyperplane margin under strict recall requirements.</td>
      <td><a href="https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset">Breast Cancer (sklearn)</a></td>
    </tr>
    <tr>
      <td><b>Deep Fraud Detection with Autoencoders:</b> Utilizing PyTorch, architect an Autoencoder neural network to detect anomalies in highly imbalanced credit card transaction data. Leverage high-level PyTorch modules to construct the encoder-decoder topology and compute reconstruction loss via MSE, efficiently separating fraudulent anomalous transactions from legitimate distributions.</td>
      <td><a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud">Credit Card Fraud</a></td>
    </tr>
    <tr>
      <td><b>Unsupervised Density-Based Clustering (DBSCAN):</b> Provided an unstructured GPS coordinate array, you must engineer a spatial clustering solution from scratch. You cannot rely on pre-defined distance thresholds; instead, you must analytically determine <code>eps</code> using a K-Distance graph and implement <code>DBSCAN</code> leveraging a BallTree or KDTree to isolate arbitrary-shaped noise-heavy noise distributions.</td>
      <td><a href="https://www.kaggle.com/datasets/ahmedmohammad2003/uber-trip-data">Uber Coordinates</a></td>
    </tr>
    <tr>
      <td><b>Centroid-Based Manifold Discovery (K-Means++):</b> Tasked with segmenting a high-dimensional customer behavior matrix, implement a K-Means strategy that explicitly optimizes the 'Elbow' and 'Silhouette' metrics. You must proactively handle multi-collinearity and scale variances using iterative <code>StandardScaler</code> applications before projecting the results onto a 2D PCA plane.</td>
      <td><a href="https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python">Mall Customers</a></td>
    </tr>
    <tr>
      <td><b>Elastic Net Feature Shrinkage & K-Fold Stratification:</b> Tasked with regressing explicit medical parameters, completely bypass <code>cross_val_score</code>. Construct a native K-Fold stratification index mapping iteratively over Elastic Net equations mathematically bounding exactly L1 vs L2 regularization shrinkage thresholds mapping explicit coefficient paths decaying visually over loop trajectories.</td>
      <td><a href="https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset">Diabetes (sklearn)</a></td>
    </tr>
    <tr>
      <td><b>Bayesian Probabilities & Textual Priors:</b> Given a corpus of Spam SMS text, transform raw vocabulary via <code>TfidfVectorizer</code> into sparse matrices. Architect a <code>MultinomialNB</code> model mapping explicit Bayesian priors systematically to counteract severe class imbalances, validating specific probability scores outputted iteratively per class.</td>
      <td><a href="https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset">SMS Spam Collection</a></td>
    </tr>
    <tr>
      <td><b>Advanced Decision Tree Pruning & Ensembling:</b> Leveraging the Wine dataset, implement a rigorous Scikit-Learn <code>DecisionTreeClassifier</code>. Rigorously deploy Scikit-Learn's structural constraints, validating <code>ccp_alpha</code> pruning parameters and integrating randomized Forest architectures to analyze structural feature importances efficiently, bypassing entirely from-scratch split manual calculations.</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html">Wine Dataset</a></td>
    </tr>
    <tr>
      <td><b>Ensemble Architectures (Voting Classifiers):</b> Working on complex tabular multi-class arrays, construct a heterogeneous <code>VotingClassifier</code> combining Logistic Regression, KNN constraints, and SVM architectures. Rigorously map predicted probability distributions natively when switching between "hard" voting constraints versus uncalibrated "soft" integrations.</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html">Synthetic Classification</a></td>
    </tr>
    <tr>
      <td><b>Polynomial Dimensionality & Bias-Variance Validation:</b> Map explicit multi-degree relationships on an auto-mpg scalar array. Systematically loop polynomial expansions scaling linearly through degree arrays using <code>PolynomialFeatures</code> combined with Ridge regression, graphing explicitly validation vs training error to mathematically isolate optimal Bias-Variance constraints.</td>
      <td><a href="https://archive.ics.uci.edu/ml/datasets/auto+mpg">Auto MPG</a></td>
    </tr>
    <tr>
      <td><b>Multi-Layered Stacking Regressors:</b> Given explicit dimensional configurations of real estate valuation, project heterogeneous base meta-models mapping explicit Ridge constraints and Tree-based leaf optimizations. Aggregate the subsequent scalar outputs into a final <code>StackingRegressor</code> terminating using Lasso L1 sparsity matrices.</td>
      <td><a href="https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction">Real Estate Valuation</a></td>
    </tr>
    <tr>
      <td><b>Support Vector Tube Extrapolations:</b> Using dense sequential historical data mapping temperature variations, architect sequential <code>SVR</code> pipelines predicting non-linear extrapolations. Radically configure the structural epsilon-tube values mapping exact matrix thresholds separating zero-penalty errors from structural margin boundary violations natively in Scikit-Learn.</td>
      <td><a href="https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data">Daily Climate</a></td>
    </tr>
    <tr>
      <td><b>Iterative Gradient Boosting & Custom Losses:</b> For a medical diagnostics pipeline using Heart Disease data, the cost of false negatives is extreme. Construct a <code>GradientBoostingClassifier</code> utilizing <code>staged_predict()</code>. You must formulate an approach to wrap the objective function to penalize false negatives severely, combined with an epoch-based early stopping loop.</td>
      <td><a href="https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset">Heart Disease</a></td>
    </tr>
    <tr>
      <td><b>XGBoost Hardware Awareness & Sparsity:</b> Deployed in a low-latency anti-fraud environment on imbalanced Credit Card Fraud data. Assemble an XGBoost pipeline natively exploiting memory sparsity. Calibrate <code>scale_pos_weight</code> and extensively tune the tree depth and sub-sampling parameters continuously via <code>RandomizedSearchCV</code>.</td>
      <td><a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud">Credit Card Fraud</a></td>
    </tr>
    <tr>
      <td><b>Staged Ensemble Aggregation:</b> Utilizing the Teleco Customer Churn data, structurally design an ensemble. Your objective is to formulate a Scikit-Learn estimator that leverages <code>warm_start=True</code> on a Random Forest to incrementally append estimators, actively halting when the validation log-loss stabilizes computed iteratively via NumPy.</td>
      <td><a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn">Telco Customer Churn</a></td>
    </tr>
    <tr>
      <td><b>Native Categorical Handling Challenge:</b> Tasked with forecasting Flight Delays, directly process string and mixed-type categorical columns without manual One-Hot Encoding. Implement a sophisticated pipeline deploying <code>HistGradientBoostingClassifier</code> or XGBoost's native categorical support, benchmarking memory footprints and execution latency.</td>
      <td><a href="https://www.kaggle.com/datasets/usdot/flight-delays">Flight Delays</a></td>
    </tr>
    <tr>
      <td><b>SVM Kernel Subspace Projections:</b> Working with complex spatial distribution clusters, architect parallel Support Vector Machines matrices. Compare native boundary mappings between Polynomial Kernel shifts and deep Radial Basis Function (RBF) projections. You must clean the raw dataset, which contains missing values and non-standardized feature scales, and engineer a <code>ColumnTransformer</code> to handle these quirks before projection.</td>
      <td><a href="https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data">Breast Cancer Wisconsin</a></td>
    </tr>
    <tr>
      <td><b>Lazy Learning Matrix Validations (KNN):</b> Predict classification mappings across a densely noisy biological feature array. Construct a <code>KNeighborsClassifier</code> isolating explicit mathematical validation through an active `BallTree` algorithm optimization sequence. You must perform extensive data cleaning on the raw input, handling outliers and inconsistent labels before tracking inference decay.</td>
      <td><a href="https://www.kaggle.com/datasets/uciml/glass">Glass Classification</a></td>
    </tr>
    <tr>
      <td><b>Manifold Discovery in Consumer Data:</b> As a marketing proxy analyzing Mall Customers, you must isolate hidden consumer sub-segments. Discard naive K-Means; construct a pipeline channeling data through t-SNE for 2D topological mapping, heavily utilizing NumPy distance matrices, followed by HDBSCAN to capture core dense clusters accurately.</td>
      <td><a href="https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python">Mall Customers</a></td>
    </tr>
    <tr>
      <td><b>Dimensionality-Constrained Dictionary Learning:</b> Using the MNIST visual dataset, bypass deep layer reliance by synthesizing a sparse PCA pipeline intersecting with a Dictionary Learning reconstructor defined in Scikit-Learn. Optimize the dictionary size explicitly to reconstruct and denoise digits corrupted by heavy simulated Gaussian noise.</td>
      <td><a href="https://www.kaggle.com/competitions/digit-recognizer">MNIST</a></td>
    </tr>
    <tr>
      <td><b>Probabilistic GMM Density Estimation:</b> Given the Wine Quality array, map the multidimensional chemical distributions applying Gaussian Mixture Models (GMM) with variable covariance types. Generate synthetic wine profiles querying the learned continuous probability distributions natively, bounding outliers using probability thresholds.</td>
      <td><a href="https://www.kaggle.com/datasets/yasserh/wine-quality-dataset">Wine Quality</a></td>
    </tr>
    <tr>
      <td><b>Hierarchical Feature Agglomeration:</b> Operating on highly multicollinear sensory data, compute the Spearman rank-order correlations systematically in Pandas/SciPy. Apply Scikit-Learn's Feature Agglomeration to hierarchically fuse tightly correlated features from the raw, uncleaned sensor signals. You must engineer a pipeline to handle noisy transients and missing sensor packets before initiating training.</td>
      <td><a href="https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones">Human Activity Recognition</a></td>
    </tr>
    <tr>
      <td><b>Spatial Density Isolation Validation:</b> Process geolocation coordination matrices combining structural `DBSCAN` logic heavily mapped over a pre-processing `UMAP` structural reduction layer. Enforce parameters that reject strict spherical topology structures to isolate arbitrary shaped geographical routing matrices.</td>
      <td><a href="https://www.kaggle.com/datasets/ahmedmohammad2003/uber-trip-data">Uber Coordinates</a></td>
    </tr>
    <tr>
      <td><b>High-Dimensional PCA & Feature Pipelines:</b> Working directly within the MNIST visual matrices, natively scale a raw pixel feature array. Construct entirely customized pipeline systems deploying Scikit-Learn's <code>PCA</code>, extracting optimal dimensions retaining 98% variances directly into an optimal downstream classifier rather than manual Eigendecomposition loops.</td>
      <td><a href="https://www.kaggle.com/datasets/oddrationale/mnist-in-csv">MNIST</a></td>
    </tr>
    <tr>
      <td rowspan="40"><b>2. Neural Networks & Deep Learning</b></td>
      <td><b>Multi-Dimensional Housing MLP:</b> Implement a 3-layer MLP to predict housing prices. Focus on standardizing heterogeneous input features using <code>StandardScaler</code> and optimizing with <code>MSELoss</code>.</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html">CA Housing</a></td>
    </tr>
    <tr>
      <td><b>Binary Health Risk MLP:</b> Build a classifier to predict heart disease risk. Implement <code>BCELoss</code> and <code>Sigmoid</code> output activation, handling binary classification thresholds.</td>
      <td><a href="https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset">Heart Disease</a></td>
    </tr>
    <tr>
      <td><b>Digit Recognition MLP:</b> Use the MNIST dataset to build a multi-class classifier. Manage 784-pixel input flattening and <code>CrossEntropyLoss</code> for 10-way classification.</td>
      <td><a href="https://www.kaggle.com/competitions/digit-recognizer">MNIST</a></td>
    </tr>
    <tr>
      <td><b>Dropout & Overfitting Control:</b> Integrate <code>nn.Dropout</code> layers into a deep MLP. Perform a comparative study on training vs validation accuracy with and without dropout active.</td>
      <td><a href="https://www.kaggle.com/datasets/zalando-research/fashionmnist">Fashion MNIST</a></td>
    </tr>
    <tr>
      <td><b>Ablation Study (Activation Functions):</b> Construct parallel models using <code>ReLU</code>, <code>Tanh</code>, and <code>LeakyReLU</code>. Map their respective loss surfaces and convergence speeds.</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html">Synthetic Moons</a></td>
    </tr>
    <tr>
      <td><b>Weight Initialization Impact:</b> Compare <code>Xavier/Glorot</code> vs <code>Zero</code> vs <code>Random</code> initialization. Visualize how gradients vanish or explode based on starting weights.</td>
      <td><a href="https://pytorch.org/docs/stable/nn.init.html">PyTorch Init</a></td>
    </tr>
    <tr>
      <td><b>Learning Rate Scheduling:</b> Implement <code>StepLR</code> and <code>ReduceLROnPlateau</code>. Document how adaptive scheduling prevents local minima stagnation during MLP training.</td>
      <td><a href="https://www.kaggle.com/c/titanic">Titanic</a></td>
    </tr>
    <tr>
      <td><b>Batch Normalization MLP Speed Trial:</b> Insert <code>BatchNorm1d</code> layers between Linear and ReLU. Measure the reduction in epochs required to reach 90% accuracy on tabular data.</td>
      <td><a href="https://www.kaggle.com/datasets/vuppalaadithyasairam/heart-disease-dataset">Health Risk</a></td>
    </tr>
    <tr>
      <td><b>L1 vs L2 Sparsity MLP:</b> Compare weight decay (L2) with manual L1 penalty implementation. Visualize the resulting weight histograms to see sparsity effects.</td>
      <td><a href="https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset">Heart Disease</a></td>
    </tr>
    <tr>
      <td><b>Early Stopping MLP Integration:</b> Program a custom validation loop that halts training when <code>val_loss</code> stops improving for a "patience" of 10 epochs.</td>
      <td><a href="https://www.kaggle.com/datasets/vuppalaadithyasairam/heart-disease-dataset">Health Risk</a></td>
    </tr>
    <tr>
      <td><b>LeNet-5 Standard Digit Classifier:</b> Implement the classic 1998 LeNet-5 architecture (AvgPool, 5x5 kernels) to classify handwritten digits, documenting tensor shape transformations.</td>
      <td><a href="https://www.kaggle.com/competitions/digit-recognizer">MNIST</a></td>
    </tr>
    <tr>
      <td><b>AlexNet Feature Extraction Layering:</b> Build the 2012 ImageNet winner with 11x11 kernels and ReLU, classifying RGB dog breeds to demonstrate deep feature hierarchy.</td>
      <td><a href="https://www.kaggle.com/datasets/miljan/stanford-dogs-dataset-tensors">Stanford Dogs</a></td>
    </tr>
    <tr>
      <td><b>VGG-16 Deep Block Sequencing:</b> Construct a 16-layer network using small 3x3 filters and modular "VGG Blocks" to classify CIFAR-10 objects with extreme depth.</td>
      <td><a href="https://www.kaggle.com/c/cifar-10">CIFAR-10</a></td>
    </tr>
    <tr>
      <td><b>Data Augmentation & Generalization:</b> Integrate a <code>torchvision</code> pipeline (Rotation, Crop, Jitter) to reduce the generalization gap in limited-category food classification.</td>
      <td><a href="https://www.kaggle.com/datasets/chrisfilo/fruit-recognition">Kaggle Fruits</a></td>
    </tr>
    <tr>
      <td><b>Transfer Learning Standard Fine-Tuning:</b> Freeze a pre-trained ResNet-18 backbone and replace the FC head to detect Pneumonia from medical X-rays with low data samples.</td>
      <td><a href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia">Chest X-Ray</a></td>
    </tr>
    <tr>
      <td><b>Lung Segmentation with Deep CNNs:</b> Build a custom U-Net-like Encoder-Decoder to identify healthy vs diseased lung tissue in chest X-rays. Implement a differentiable Dice Loss to overcome severe foreground-background pixel imbalance.</td>
      <td><a href="https://www.kaggle.com/datasets/nikhilpandey31/lung-segmentation-from-chest-x-ray-dataset">Lung Segmentation</a></td>
    </tr>
    <tr>
      <td><b>Deep Multi-Layer Perceptron Architectures:</b> Operating over the Synthetic Moons dataset, design a 3-layer Multi-Layer Perceptron deploying PyTorch's high-level <code>nn.Module</code> arrays. Optimize non-linear decision boundaries through sequential <code>nn.ReLU</code> and <code>nn.CrossEntropyLoss</code> metrics natively executed on <code>autograd</code> rather than extracting manual derivations.</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html">Synthetic Moons</a></td>
    </tr>
    <tr>
      <td><b>Open Problem: End-to-End Solution Engineering:</b> Given a complex, raw dataset representing heterogeneous sensor signals, you must engineer a full solution from scratch. You are not allowed to use a predefined pipeline template; you must autonomously decide on feature engineering (e.g., FFT, wavelet transforms), architecture (CNN vs MLP), and validation strategy to solve the hidden classification objective.</td>
      <td><a href="https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones">Human Activity Recognition</a></td>
    </tr>
    <tr>
      <td><b>Deep Regressive Feature Extraction:</b> Working with the California Housing dataset, construct a deep neural network that not only predicts housing prices but also outputs an auxiliary latent feature representation. You must define a custom PyTorch <code>forward</code> pass that extracts these features into a separate Pandas DataFrame for downstream clustering analysis.</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html">CA Housing</a></td>
    </tr>
    <tr>
      <td><b>Dynamic Learning Rate Interventions:</b> Tasked with the continuous classification of Fashion MNIST tensors, architecture a rigid PyTorch training loop. Integrate AdamW decay alongside a customized OneCycleLR parameter scheduling, embedding dynamic early stopping condition evaluation stored iteratively inside a Pandas DataFrame tracker.</td>
      <td><a href="https://www.kaggle.com/datasets/zalando-research/fashionmnist">Fashion MNIST</a></td>
    </tr>
    <tr>
      <td><b>Custom Topologies & Gradient Scaling:</b> Leveraging California Housing regressions, design a PyTorch framework predicting absolute spatial costs. The business constraint demands logarithmic error penalization (RMSLE). Implement this loss natively ensuring zero gradient explosions, coupled directly with multi-head Dropout strategies stopping local topology collapse.</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html">CA Housing</a></td>
    </tr>
    <tr>
      <td><b>Tensor Broadcasting Constraints:</b> Orchestrating an attention routing mechanism for raw embedded indices, exclusively leverage PyTorch tensor properties (via <code>torch.einsum</code> or <code>matmul</code>) to output the scaled dot-product attention mapping. Forbid <code>for</code> loops actively, pushing aggressive computational broadcasting constraints across the GPU.</td>
      <td><a href="https://pytorch.org/docs/stable/tensors.html">PyTorch Tensor Docs</a></td>
    </tr>
    <tr>
      <td><b>Activation Gradients & Loss Surface Mapping:</b> Rigorously swap PyTorch network topology internal mapping equations transitioning ReLU, Sigmoid, and Tanh constraint networks tracking performance across identical classification boundaries mapping structural Mean Squared Error (MSE) integrations against Binary Cross Entropy (BCE) constraints visually.</td>
      <td><a href="https://www.kaggle.com/c/titanic">Titanic</a></td>
    </tr>
    <tr>
      <td><b>ELBO Matrix Optimization (VAE):</b> Assemble fundamental Variational Autoencoder matrices. Systematically derive PyTorch loss blocks mapping explicit mathematical properties evaluating explicit Evidence Lower Bound (ELBO) integration separating reconstruction boundaries actively off heavily restricted Kullback-Leibler divergence calculations natively.</td>
      <td><a href="https://www.kaggle.com/competitions/digit-recognizer">MNIST</a></td>
    </tr>
    <tr>
      <td><b>Low-Rank Tensor Approximations (PEFT):</b> Process heavy generative pre-trained architectures mapping explicitly a customized LoRA layer configuration matrices structuring explicit frozen topological states appending strictly updated low-rank structures significantly controlling validation loss gradients preventing destructive matrix catastrophic failures dynamically.</td>
      <td><a href="https://www.kaggle.com/datasets/nimitmak/medical-qa-dataset">Medical Q&A</a></td>
    </tr>
    <tr>
      <td><b>Bellman Equation State Traversal:</b> Formulate sophisticated array matrices mapping dynamically across Markov framework states organizing strict tabular Q-Learning routing frameworks mapping explicitly reward mappings actively scaling dimensional matrices into a generalized PyTorch deep Neural Network processing explicit pixel data constraints evaluating explicitly temporal discounting algorithms.</td>
      <td>OpenAI Gym (CartPole)</td>
    </tr>
    <tr>
      <td><b>Proximal Policy Human Alignment:</b> Establish rigid structural optimization parameters mapping RLHF processes tracking explicit mathematical frameworks combining reward mappings systematically mapping policy gradient bounding scaling explicitly restricting structural catastrophic drift mapping complex loss bounding limits validating dynamic output parameters rigorously testing mathematical optimization boundaries.</td>
      <td>Theoretical Exercise</td>
    </tr>
    <tr>
      <td><b>Generative Entropy Stochastic Mapping:</b> Manipulate generative frameworks tracking generative configurations adjusting explicitly mapped logit parameters parsing outputs recursively applying structural constraint formulas evaluating parameter stability dynamically tracking Top-K limitations vs Top-P boundaries establishing strict mapping behaviors via explicit Python looping thresholds natively extracting stochastic variations mathematically tracking outputs continuously.</td>
      <td>API / Local LLM</td>
    </tr>
    <tr>
      <td><b>Prompt Engineering Optimization Pipeline:</b> Build an explicit automated testing suite iterating LLM prompt variations mathematically. Script structural benchmarks assessing Zero-Shot, Few-Shot, Chain-of-Thought, and Meta-Prompting accuracy variances executing dynamic validations comparing contextual bounds natively isolating exactly which methodology retrieves optimal parameter stability across complex text arrays.</td>
      <td>HuggingFace API / Local LLM</td>
    </tr>
    <tr>
      <td><b>Transfer Learning Freezing Limits:</b> Given massive network limits traversing pre-trained arrays isolating exact weight bindings. Extract structural fine-tuning bounds freezing explicit mathematical arrays tracking strictly appended customized mapping modules iterating backpropagation strictly targeting exclusively novel neural mappings analyzing specifically computational parameter variances avoiding topological collapse.</td>
      <td><a href="https://huggingface.co/models">HuggingFace Models</a></td>
    </tr>
    <tr>
      <td rowspan="9"><b>3. Computer Vision</b></td>
      <td><b>Architectural Bypass via Skip Connections:</b> Confronting the CIFAR-10 challenge matrix, program a customized ResNet topography from bare PyTorch modules. Project custom residual blocking structures while handling raw, unnormalized image tensors and performing complex data augmentation strategies to prevent over-fitting.</td>
      <td><a href="https://www.kaggle.com/c/cifar-10">CIFAR-10</a></td>
    </tr>
    <tr>
      <td><b>Real-Time Object Detection (YOLO Custom Head):</b> Targeting high-speed maritime navigation, implement a YOLO-style (You Only Look Once) detection head. You must process raw image frames, formulate a multi-part loss function (localization, confidence, and class loss), and implement non-maximum suppression (NMS) to eliminate overlapping bounding box proposals from scratch.</td>
      <td><a href="https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataset">Maritime Objects</a></td>
    </tr>
    <tr>
      <td><b>Zero-Shot Visual Reasoning (CLIP):</b> Leveraging Contrastive Language-Image Pre-training (CLIP) principles, architect a dual-encoder system. You must align image embeddings with textual label embeddings in a shared latent space, enabling the model to classify unseen objects without explicit categorical training, mapped via cosine similarity.</td>
      <td><a href="https://www.kaggle.com/c/imagenet-object-localization-challenge">ImageNet (Subset)</a></td>
    </tr>
    <tr>
      <td><b>ViT - Vision Transformer Architectures:</b> Bypass traditional Convolutional layers to build a Vision Transformer (ViT). Implement patch embedding, positional encoding, and a multi-head self-attention backbone to process images as sequences of tokens, validating performance on high-resolution medical imagery.</td>
      <td><a href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia">Chest X-Ray (Pneumonia)</a></td>
    </tr>
    <tr>
      <td><b>Semantic Masking & Region Isolation:</b> Targeting industrial navigation on Cityscapes sequences, define a structural U-Net layout. Enforce expanding and contracting paths alongside matching symmetrical skip linkages. Compute a customized Intersection-over-Union (IoU) differentiable framework guiding the spatial optimization loop.</td>
      <td><a href="https://www.kaggle.com/datasets/dansbecker/cityscapes-image-pairs">Cityscapes</a></td>
    </tr>
    <tr>
      <td><b>Self-Supervised Contrastive Formulations:</b> Provided unlabelled patches sourced from ImageNet subsets, formalize a fundamental SimCLR logic layer in PyTorch. Combine randomized geometrical transformations via <code>torchvision</code> translating directly towards an InfoNCE theoretical contrastive loss gradient formulation.</td>
      <td><a href="https://www.kaggle.com/c/imagenet-object-localization-challenge">ImageNet (Subset)</a></td>
    </tr>
    <tr>
      <td><b>Multi-Scale Structural Proposals (mAP):</b> Evaluate complex structural arrays contrasting YOLO localized detection regressions dynamically opposed to Mask R-CNN topological instance logic mappings. Actively formulate an analytical evaluation script tracing mathematical Mean Average Precision (mAP) metrics processing explicitly under spatial intersection thresholds.</td>
      <td><a href="https://cocodataset.org/">COCO Dataset</a></td>
    </tr>
    <tr>
      <td><b>Bipartite Matching Structural Detections:</b> Overhauling traditional anchoring methods map natively a complete DETR layout mechanism structure implementing direct array matching functions deriving exact Hungarian algorithm matching algorithms across fixed multi-label token mappings mapping bounding parameter regressions efficiently.</td>
      <td><a href="https://www.kaggle.com/datasets/robikscube/pascal-voc-2012">PASCAL VOC</a></td>
    </tr>
    <tr>
      <td><b>Generative Adversarial Regularizations:</b> Provided the CelebA landscape, organize a DCGAN integration. Combat persistent mode collapse by designing single-sided label smoothing algorithms paired dynamically with Gaussian noise interference fed aggressively into the localized Discriminator topologies evaluating Inception Scores manually.</td>
      <td><a href="https://www.kaggle.com/datasets/jessicali9530/celeba-dataset">CelebA</a></td>
    </tr>
    <tr>
      <td rowspan="12"><b>4. Natural Language Processing & Audio</b></td>
      <td><b>Recurrent State Memory Matrices:</b> For processing variable-length financial sentiments, transform text arrays into tightly packed sequential data batches. Enact an LSTM model processing directly utilizing <code>pack_padded_sequence</code> in PyTorch, dynamically stripping explicit padding tokens, feeding strictly final embedded memory vectors outward to a classification head.</td>
      <td><a href="https://www.kaggle.com/datasets/sbatti/financial-sentiment-analysis">Financial Sentiment</a></td>
    </tr>
    <tr>
      <td><b>Attention Mechanism Decoding Frameworks:</b> Intersecting bilingual text translations across Europarl strings, align a raw Seq2Seq framework isolating transformer blocks completely. Execute mathematical blueprints for multiplicative (Luong) alignment weights iteratively computing the temporal focus matrix mappings via Numpy indexing during real-time generation.</td>
      <td><a href="https://www.statmt.org/europarl/">Europarl</a></td>
    </tr>
    <tr>
      <td><b>Transformer Encoder Block Distillation:</b> Instructed to forge a solo bidirectional BERT-style computational layer from base PyTorch operations. Formulate explicit queries, keys, and values matrices to project Multi-Head dimensions identically, cascading matrices strictly down layer normalizations probing parameter weight capacity stability directly.</td>
      <td><a href="https://huggingface.co/models">HuggingFace Models</a></td>
    </tr>
    <tr>
      <td><b>Tokenization & Byte-Pair Analytics:</b> Completely eliminating tokenizer library APIs, intake an unstructured sequence text via Pandas operations. Architect a definitive Byte-Pair Encoding (BPE) process operating entirely in naive Python loops recursively mapping consecutive pairing frequencies up towards an explicit maximum token dimension restraint.</td>
      <td><a href="https://www.kaggle.com/datasets/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones">Amazon Reviews Corpus</a></td>
    </tr>
    <tr>
      <td><b>Continuous Autoregressive Decoding Protocols:</b> Configure a mathematically rigorous Decoder-only GPT sub-block framework routing causal mappings sequentially. Enforce zero attention leakage enforcing masking mechanisms recursively, testing output variability mapping Top-P and Temperature scaling algorithms natively via logits.</td>
      <td><a href="https://www.kaggle.com/datasets/Cornell-University/arxiv">arXiv Summaries</a></td>
    </tr>
    <tr>
      <td><b>Continuous Bag-of-Words (CBOW) Spatial Mappings:</b> Tasked with mapping vast unstructured tokens, reconstruct a native Word2Vec architecture purely iterating inside PyTorch neural boundaries. Calculate localized vocabulary contexts defining explicitly embedded lookup properties computing Cosine Similarity matrix projections mapping word boundaries algebraically avoiding GenSim pipelines.</td>
      <td>Any Text Corpus</td>
    </tr>
    <tr>
      <td><b>Frequency Domain Topologies:</b> Transcribe unstructured raw waveform signals compiling structurally mapped Mel Spectrogram arrays utilizing librosa/SciPy configurations mapping strictly optimized frame layers. Vectorize outputs cascading deeply across Conv2D mapping structures computing explicit multi-class categorical arrays precisely.</td>
      <td><a href="https://www.kaggle.com/datasets/joserzapata/free-spoken-digit-dataset-fsdd">FSDD</a></td>
    </tr>
    <tr>
      <td><b>Contrastive Speech Quantization Masks:</b> Incorporating massive unstructured recording structures, build a Wav2Vec2 mapping structure formulating contrastive topological representations structurally resolving masked hidden boundaries parsing raw wave embeddings mapping mathematically towards optimal signal loss convergences.</td>
      <td><a href="https://www.kaggle.com/datasets/pypiahmad/librispeech-asr-corpus">LibriSpeech</a></td>
    </tr>
    <tr>
      <td><b>Weakly Supervised Acoustic Mapping:</b> Investigate weak spatial parameters structurally leveraging Whisper transcription logic configurations tracking dynamic contextual text representations generating multi-dimensional attention states natively binding noise reduction filtering parameters recursively mapping language translation.</td>
      <td>HuggingFace API</td>
    </tr>
    <tr>
      <td><b>Multimodal Natural Sound Topologies:</b> Organize structural Qwen-Audio analytical scripts configuring explicitly complex prompt inputs compiling structural language instructions aligning heavily with acoustic spatial vectors extracting non-verbal classification representations rigorously formatting categorical mapping matrices.</td>
      <td>HuggingFace Models</td>
    </tr>
    <tr>
      <td><b>Temporal Sequence Alignment (DTW):</b> Operating entirely under raw sequential Numpy structures computing localized auditory matrices tracking Dynamic Time Warping operations. Align differing frequency lengths evaluating pure boundary distances optimizing global alignment bounds natively strictly bounded algorithmically offline.</td>
      <td><a href="https://www.kaggle.com/datasets/joserzapata/free-spoken-digit-dataset-fsdd">FSDD</a></td>
    </tr>
    <tr>
      <td><b>Recurrent Acoustic Isolation Sequences:</b> Synthesize basic auditory mapping inputs projecting dimensional matrices traversing Recurrent Neural Network sequences directly. Implement pure linear transformations calculating sequential signal variances matching classification loss functions minimizing noise distributions heavily via mapped states tracking explicitly frame by frame.</td>
      <td><a href="https://www.kaggle.com/datasets/pypiahmad/librispeech-asr-corpus">LibriSpeech</a></td>
    </tr>
  </tbody>
</table>

* Output the following prompt in some LLM, preferably in your IDE, to create a detailed `.md` file for each exercise.

<pre>
You are an expert AI Educator and a Problem Setter for the International Olympiad in Artificial Intelligence (IOAI) and the Olimpíada Nacional de Inteligência Artificial (ONIA). Your task is to take a brief exercise description and expand it into a full, rigorous, "Olympiad-style" problem statement in Markdown (.md) format.

I will provide you with a brief [EXERCISE DESCRIPTION] and a [DATASET REFERENCE].

Your output must be a comprehensive Markdown file that strictly follows this structure:

# [Creative and Formal Problem Title]
**Difficulty Level:** [Estimate: Phase 3 (Intermediate) or Phase 4 (Advanced)] | **Time Limit:** [Estimate, e.g., 2 Hours]

## 1. Background / Scenario
Create a brief, engaging real-world scenario that contextualizes the problem structurally. Olympiad problems always have a complex dynamic narrative (e.g., "You are the chief data engineer handling structural failure predictions in deep-ocean pipelines...").

## 2. Problem Statement
Clearly and formally define the robust operational algorithms and modeling systems the student must construct. Replace generic phrasing with specific parameter expectations.

## 3. Dataset Description
Describe the dataset based on the provided link/context. Mention the explicit feature properties matrices, expected schema quirks, and statistical properties to be accommodated (e.g., "contains zero-inflated arrays in target columns").

## 4. Subtasks & Point Distribution
Break the underlying structure into 3 to 4 hyper-specific systematic subtasks. 
* **Task 4.1: [Name] (XX pts):** Define precise matrix manipulations / preprocessing pipelines. 
* **Task 4.2: [Name] (XX pts):** Detail the exact modeling layers / hyperparameter bounds.
* **Task 4.3: [Name] (XX pts):** Explain the rigid validation / scoring constraint integration.
(Ensure the tasks directly align with high-level optimization parameters referenced conceptually in the prompt).

## 5. Constraints & Technical Rules
List severe parameters and computational thresholds that scale complexity constraints.
* Allowed/Disallowed libraries (e.g., "You are strictly permitted to deploy Pandas, Scikit-Learn, and NumPy arrays for this constraint, isolating external PyTorch dependencies completely unless mapped mathematically backward").
* Execution constraints (e.g., "Must complete matrix transformations under 5 minutes utilizing strictly vectorized mathematical operators").
* **Scikit-Learn, XGBoost & Tooling Constraints (CRITICAL):** If the domain intersects tree-based or standard modeling implementations, mandate sophisticated tracking. The implementation MUST systematically feature arrays scaling with `Pipeline`, `ColumnTransformer`, custom loss overrides, or advanced structural metrics like `warm_start=True` or `staged_predict()`. Handling epochs iteration dynamically using partial fitting, tracked structurally using NumPy and Pandas properties.

## 6. Evaluation Criteria
Explain the rigid structural tests deployed. (e.g. penalized F1-weighted variance, bounded Mean Absolute Error via logarithm mapping, execution footprint). Specify how automated tests analyze validation leakage implicitly.

## 7. Deliverables
State exactly what elements construct the final package output structure natively (e.g., `pipeline.py`, `inference_script.ipynb`, and analytical vector mapping plots).

---
CRITICAL INSTRUCTIONS:
- DO NOT output direct model code or pure Python. This represents the explicit structural exam for an olympiad student to interpret and compile interactively.
- Leverage severe, professional mathematical Markdown.
- Design the problem mathematically robustly, reflecting a high-tier advanced programming competition matrix architecture limit.

Here is the input:
[EXERCISE DESCRIPTION]: 
[DATASET REFERENCE]: 
</pre>

## License
This repository is under the GPL 3.0 License from June, 29th 2007.| 62 | **Physio-Graph: Neural Temporal Graphs** | [PhysioNet Patient Data](https://www.kaggle.com/datasets/koki25/physionet-2012-challenge-dataset) | [instructions.md](exercises/62_Physio-Graph_Neural_Temporal_Graphs/instructions.md) |
| 63 | **GAN-Powered Anomaly Synthesis** | [Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection) | [instructions.md](exercises/63_GAN-Powered_Anomaly_Synthesis/instructions.md) |
| 64 | **Hyper-Network Matrix Controllers** | [Omniglot (Few-Shot)](https://www.kaggle.com/datasets/sainikhileswar/omniglot-dataset) | [instructions.md](exercises/64_Hyper-Network_Matrix_Controllers/instructions.md) |
| 65 | **Differentiable Sorting & Ranking** | [MSLR (Learning to Rank)](https://www.kaggle.com/datasets/petezhishuo/mslr-web10k) | [instructions.md](exercises/65_Differentiable_Sorting_&_Ranking/instructions.md) |
| 66 | **Implicit Neural SDF Reconstruction** | [ModelNet40 (3D shapes)](https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset) | [instructions.md](exercises/66_Implicit_Neural_SDF_Reconstruction/instructions.md) |
