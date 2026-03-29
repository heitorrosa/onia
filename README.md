# deeplearning
A repository dedicated to study for the 2nd Olimpíada Nacional de Inteligência Artificial (ONIA) and the International Olympiad in Artificial Intelligence (IOAI).

* [IOAI Kazakhstan Syllabus](https://ioai-official.org/wp-content/uploads/2025/10/Syllabus.pdf) 
* [2nd ONIA Syllabus](https://www.oniabrasil.com.br/assets/files/Syllabus_da_2_ONIA.pdf) 

### Related repositories
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

## Study Roadmap

**Phase 1: Neural Network Foundations**
* *Topics:* Perceptrons, Activation Functions (ReLU, Sigmoid), Loss Functions (MSE, Cross-Entropy), Gradient Descent, Backpropagation, and the Chain Rule, Adam and AdamW.
* *Goal:* Understand the "Theory (How it works)" of weights, biases, and learning rates.

**Phase 2: Classical Machine Learning (StatQuest)**
* *Topics:* Bias & Variance, Cross-Validation, Regularization (L1/L2), Evaluation Metrics (Confusion Matrix, ROC/AUC).
* *Models:* Logistic Regression, SVMs, Decision Trees, Random Forests, AdaBoost, XGBoost.
* *Unsupervised:* PCA, K-Means, Hierarchical Clustering, DBSCAN, t-SNE, UMAP.

**Phase 3: Deep Learning & PyTorch Architectures**
* *Vision:* CNNs (ResNet, U-Net), Object Detection (YOLO, R-CNN, DETR).
* *Sequences:* RNNs, LSTMs, Word2Vec, Seq2Seq.
* *Attention & Transformers:* Self-Attention, Vision Transformers (ViT), BERT (Encoder), GPT (Decoder).

**Phase 4: Generative AI, Audio & Advanced Tuning**
* *Generative:* Autoencoders, VAEs, GANs, Latent Diffusion Models (Stable Diffusion, ControlNet).
* *Audio:* Mel Spectrograms, HuBERT, Wav2vec2, Whisper, Qwen-Audio.
* *Optimization:* Transfer Learning, Fine-tuning, LoRA (PEFT), Reinforcement Learning (RLHF), Prompt Engineering.

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
      <td rowspan="6"><b>Programming Fundamentals</b></td>
      <td><b>Data Cleaning & EDA:</b> Use Pandas and Matplotlib to analyze a messy Titanic or Weather dataset. Handle missing values, filter outliers, and plot distributions.</td>
      <td><a href="https://www.kaggle.com/c/titanic">Titanic Dataset</a></td>
    </tr>
    <tr>
      <td><b>Scikit-Learn Pipelines:</b> Build a robust preprocessing pipeline (<code>SimpleImputer</code>, <code>StandardScaler</code>, <code>OneHotEncoder</code>) using the <code>ColumnTransformer</code> on the Adult Census Income dataset.</td>
      <td><a href="https://www.kaggle.com/datasets/uciml/adult-census-income">Adult Income</a></td>
    </tr>
    <tr>
      <td><b>Time-Series Manipulation:</b> Use Pandas to group data by month/year, handle datetime indices, and create rolling averages on a historical weather dataset.</td>
      <td><a href="https://www.kaggle.com/datasets/rtatman/did-it-rain-in-seattle-19482017">Seattle Weather</a></td>
    </tr>
    <tr>
      <td><b>NumPy Vectorization:</b> Write a script that calculates the Euclidean distance between thousands of points using standard Python <code>for</code> loops, and then rewrite it using NumPy broadcasting to compare execution times.</td>
      <td>Synthetic Array Data</td>
    </tr>
    <tr>
      <td><b>PyTorch Tensor Mastery:</b> Replicate standard NumPy operations (matrix multiplication, broadcasting, reshaping) using strictly PyTorch Tensors (<code>torch.matmul</code>, <code>torch.view</code>). Move tensors between CPU and GPU.</td>
      <td><a href="https://pytorch.org/docs/stable/tensors.html">PyTorch Docs</a></td>
    </tr>
    <tr>
      <td><b>Custom PyTorch DataLoaders:</b> Write a custom <code>torch.utils.data.Dataset</code> class to load images from a folder, apply basic transforms, and iterate through them using a <code>DataLoader</code>.</td>
      <td><a href="https://pytorch.org/tutorials/beginner/data_loading_tutorial.html">PyTorch Tutorial</a></td>
    </tr>
    <tr>
      <td rowspan="6"><b>Data Science Fundamentals</b></td>
      <td><b>The Evaluation Suite:</b> Evaluate a pre-trained classifier on a Breast Cancer dataset. Calculate Accuracy, Precision, Recall, F1-Score, and plot the ROC Curve and Confusion Matrix manually.</td>
      <td><a href="https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data">Breast Cancer</a></td>
    </tr>
    <tr>
      <td><b>Cross-Validation & Grid Search:</b> Implement K-Fold cross-validation using <code>GridSearchCV</code> to find the optimal hyperparameters (<code>max_depth</code>, <code>n_estimators</code>) for a model predicting Wine Quality.</td>
      <td><a href="https://www.kaggle.com/datasets/yasserh/wine-quality-dataset">Wine Quality</a></td>
    </tr>
    <tr>
      <td><b>Outlier Detection & Imputation:</b> Identify anomalies in a housing dataset using the IQR method and Z-scores. Test replacing them with the median vs. dropping the rows entirely.</td>
      <td><a href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques">House Prices</a></td>
    </tr>
    <tr>
      <td><b>Advanced Feature Selection:</b> Use Mutual Information (<code>mutual_info_classif</code>) and Recursive Feature Elimination (RFE) to reduce a dataset of 100+ variables down to the 10 most impactful ones.</td>
      <td><a href="https://scikit-learn.org/stable/modules/feature_selection.html">sklearn Datasets</a></td>
    </tr>
    <tr>
      <td><b>Feature Engineering:</b> Extract time-based features (day of week, month), rolling averages, and lag features from a Bike Sharing demand dataset to improve regression performance.</td>
      <td><a href="https://www.kaggle.com/c/bike-sharing-demand">Bike Sharing</a></td>
    </tr>
    <tr>
      <td><b>Handling Class Imbalance:</b> Train a model on highly imbalanced Credit Card Fraud data. Compare standard training against using SMOTE and modifying <code>class_weights</code> in PyTorch's <code>CrossEntropyLoss</code>.</td>
      <td><a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud">Credit Card Fraud</a></td>
    </tr>
    <tr>
      <td rowspan="6"><b>Supervised Learning</b></td>
      <td><b>Linear & Logistic Regressions:</b> Predict Diabetes disease progression using Linear Regression with L1 (Lasso) and L2 (Ridge) regularization to observe feature coefficient shrinkage.</td>
      <td><a href="https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset">Diabetes (sklearn)</a></td>
    </tr>
    <tr>
      <td><b>Elastic Net Regression:</b> Combine L1 and L2 penalties using Elastic Net. Perform a grid search to find the optimal <code>l1_ratio</code> on a dataset with highly correlated features.</td>
      <td><a href="https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset">CA Housing</a></td>
    </tr>
    <tr>
      <td><b>K-NN & SVM Decision Boundaries:</b> Classify Penguin species. Train SVMs with different kernels (Linear, RBF, Poly) and plot their 2D decision boundaries using PCA-reduced features.</td>
      <td><a href="https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data">Palmer Penguins</a></td>
    </tr>
    <tr>
      <td><b>Trees to Forests:</b> Predict Heart Disease risk. Compare a single overfitted Decision Tree against a Random Forest ensemble to visualize the reduction in variance.</td>
      <td><a href="https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset">Heart Disease</a></td>
    </tr>
    <tr>
      <td><b>Voting & Stacking Ensembles:</b> Combine a Logistic Regression, a Support Vector Machine, and a Random Forest using <code>VotingClassifier</code> and <code>StackingClassifier</code> to beat the performance of any single model.</td>
      <td><a href="https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data">Breast Cancer</a></td>
    </tr>
    <tr>
      <td><b>Gradient Boosting Mastery:</b> Use XGBoost or LightGBM to predict advanced House Prices. Focus on handling complex categorical variables and using early stopping.</td>
      <td><a href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques">House Prices</a></td>
    </tr>
    <tr>
      <td rowspan="6"><b>Unsupervised Learning</b></td>
      <td><b>K-Means Segmentation:</b> Perform customer segmentation on a Mall Customers dataset based on spending scores and income. Use the Elbow method to find the optimal 'K'.</td>
      <td><a href="https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python">Mall Customers</a></td>
    </tr>
    <tr>
      <td><b>Hierarchical Clustering & Dendrograms:</b> Apply Agglomerative Clustering on the same Mall dataset and plot a Dendrogram using SciPy to visually verify the optimal number of clusters.</td>
      <td><a href="https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python">Mall Customers</a></td>
    </tr>
    <tr>
      <td><b>Gaussian Mixture Models (GMM):</b> Fit a GMM to an overlapping dataset and extract the probabilities of each point belonging to a specific cluster (Soft Clustering).</td>
      <td>Synthetic Clusters</td>
    </tr>
    <tr>
      <td><b>PCA Dimension Reduction:</b> Apply PCA on the MNIST dataset. Plot the cumulative explained variance and train a Logistic Regression model on the reduced dimensions vs. the raw pixels.</td>
      <td><a href="https://www.kaggle.com/datasets/oddrationale/mnist-in-csv">MNIST</a></td>
    </tr>
    <tr>
      <td><b>DBSCAN Anomaly Detection:</b> Use DBSCAN to identify spatial clusters and noisy anomalies in a geospatial dataset (like NYC Taxi drop-offs).</td>
      <td><a href="https://www.kaggle.com/c/nyc-taxi-trip-duration">NYC Taxi Data</a></td>
    </tr>
    <tr>
      <td><b>Manifold Learning:</b> Compare t-SNE and UMAP on the Fashion-MNIST dataset. Observe how UMAP preserves global structure better than t-SNE while reducing dimensions to 2D.</td>
      <td><a href="https://www.kaggle.com/datasets/zalando-research/fashionmnist">Fashion MNIST</a></td>
    </tr>
    <tr>
      <td rowspan="6"><b>Neural Networks (PyTorch)</b></td>
      <td><b>Backprop from Scratch:</b> Build a simple 2-layer Perceptron in pure NumPy to solve a synthetic XOR dataset, implementing gradient descent and the chain rule manually.</td>
      <td>Synthetic (XOR)</td>
    </tr>
    <tr>
      <td><b>First PyTorch MLP:</b> Port the NumPy logic to PyTorch using <code>nn.Sequential</code>. Train a regression model on the California Housing dataset using <code>nn.MSELoss()</code>.</td>
      <td><a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html">CA Housing</a></td>
    </tr>
    <tr>
      <td><b>Custom Loss Functions:</b> Implement a custom Huber Loss function (Smooth L1 Loss) in PyTorch to train a model that is robust against extreme outliers in the target variable.</td>
      <td><a href="https://www.kaggle.com/c/house-prices-advanced-regression-techniques">House Prices</a></td>
    </tr>
    <tr>
      <td><b>Activations Exploration:</b> Build a multi-class PyTorch classifier. Experiment by swapping ReLU, LeakyReLU, Tanh, and Sigmoid activations to observe dead neurons and saturation.</td>
      <td><a href="https://www.kaggle.com/datasets/zalando-research/fashionmnist">Fashion MNIST</a></td>
    </tr>
    <tr>
      <td><b>The Optimizer Battle:</b> Train a deep network. Compare convergence speed and test accuracy over 50 epochs using SGD, SGD with Momentum, Adam, and AdamW.</td>
      <td><a href="https://www.kaggle.com/datasets/oddrationale/mnist-in-csv">MNIST</a></td>
    </tr>
    <tr>
      <td><b>Learning Rate Schedulers:</b> Train an MLP using <code>optim.lr_scheduler.StepLR</code> and <code>CosineAnnealingLR</code>. Plot the learning rate decay over epochs and observe how it helps escape local minima.</td>
      <td><a href="https://www.kaggle.com/datasets/oddrationale/mnist-in-csv">MNIST</a></td>
    </tr>
    <tr>
      <td rowspan="6"><b>Deep Learning & Regularization</b></td>
      <td><b>Overfitting Baseline:</b> Intentionally overfit a deep, unregularized PyTorch model on a small subset of training data (e.g., 500 images of MNIST) until training accuracy is 100% but validation is poor.</td>
      <td><a href="https://www.kaggle.com/datasets/oddrationale/mnist-in-csv">MNIST (Subset)</a></td>
    </tr>
    <tr>
      <td><b>Weight Initialization Study:</b> Initialize network weights using purely random normals vs. <code>nn.init.xavier_uniform_</code> and <code>nn.init.kaiming_normal_</code> (He init). Track the gradient norms in the first layer to visualize vanishing/exploding gradients.</td>
      <td><a href="https://www.kaggle.com/datasets/zalando-research/fashionmnist">Fashion MNIST</a></td>
    </tr>
    <tr>
      <td><b>Dropout Ablation:</b> Add <code>nn.Dropout</code> layers to your overfitted network. Experiment with dropout rates (0.2, 0.5, 0.8) and analyze the impact on the validation loss curve.</td>
      <td><a href="https://www.kaggle.com/datasets/oddrationale/mnist-in-csv">MNIST (Subset)</a></td>
    </tr>
    <tr>
      <td><b>L2 Regularization (Weight Decay):</b> Instead of Dropout, apply Weight Decay directly in the PyTorch Optimizer (e.g., <code>Adam(weight_decay=1e-4)</code>). Compare the final parameter histogram to a non-regularized model.</td>
      <td><a href="https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data">Breast Cancer</a></td>
    </tr>
    <tr>
      <td><b>Batch Norm vs Layer Norm:</b> Implement <code>nn.BatchNorm1d</code> and compare its training speed/stability against <code>nn.LayerNorm</code> on a dataset with highly varying feature scales.</td>
      <td><a href="https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset">Diabetes (sklearn)</a></td>
    </tr>
    <tr>
      <td><b>Early Stopping Callback:</b> Write a custom Python class that monitors <code>val_loss</code> during the PyTorch training loop and saves the <code>state_dict</code> (best model weights), stopping training if the metric doesn't improve for <i>N</i> epochs.</td>
      <td>Any Dataset</td>
    </tr>
    <tr>
      <td rowspan="6"><b>Computer Vision Fundamentals</b></td>
      <td><b>Hello World CV:</b> Build a Convolutional Neural Network (CNN) in PyTorch to identify hand-drawn digits (0-9).</td>
      <td><a href="https://www.kaggle.com/competitions/digit-recognizer">MNIST</a></td>
    </tr>
    <tr>
      <td><b>Natural Scene Classification:</b> Train a CNN to distinguish between 6 categories of landscapes (buildings, forests, glaciers, mountains, sea, streets).</td>
      <td><a href="https://www.kaggle.com/datasets/puneet6060/intel-image-classification">Intel Image</a></td>
    </tr>
    <tr>
      <td><b>Image Augmentation Pipeline:</b> Build a robust <code>torchvision.transforms</code> pipeline (rotation, color jitter, cropping) to drastically improve a baseline CNN's accuracy.</td>
      <td><a href="https://www.cs.toronto.edu/~kriz/cifar.html">CIFAR-10</a></td>
    </tr>
    <tr>
      <td><b>Satellite Poverty Map:</b> Train a CNN to classify satellite images of regions into different wealth brackets as a proxy for economic indicators.</td>
      <td><a href="https://www.kaggle.com/datasets/sandeshbhat/satellite-images-to-predict-povertyafrica">Satellite Images</a></td>
    </tr>
    <tr>
      <td><b>Transfer Learning Baseline:</b> Use a pre-trained ResNet18 model, freeze its early layers, and replace the final classifier head to classify Dogs vs Cats.</td>
      <td><a href="https://www.kaggle.com/c/dogs-vs-cats">Dogs vs Cats</a></td>
    </tr>
    <tr>
      <td><b>CNN Filter Visualization:</b> Extract the weights from the first convolutional layer of a trained network and plot them as images to visualize the learned "edge detectors".</td>
      <td><a href="https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html">PyTorch Models</a></td>
    </tr>
    <tr>
      <td rowspan="6"><b>Advanced Vision</b></td>
      <td><b>Traffic Sign Detection:</b> Use a pre-trained YOLO model (Ultralytics) to identify and locate various traffic signs in diverse street scenes with bounding boxes.</td>
      <td><a href="https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign">GTSRB</a></td>
    </tr>
    <tr>
      <td><b>Urban Scene Segmentation:</b> Implement a U-Net to perform pixel-level segmentation on city street images, separating "road," "sidewalk," and "car."</td>
      <td><a href="https://www.kaggle.com/datasets/dansbecker/cityscapes-image-pairs">Cityscapes</a></td>
    </tr>
    <tr>
      <td><b>Realistic Face Generation:</b> Code a Deep Convolutional GAN (DCGAN) to generate high-resolution synthetic faces from random noise vectors.</td>
      <td><a href="https://www.kaggle.com/datasets/jessicali9530/celeba-dataset">CelebA</a></td>
    </tr>
    <tr>
      <td><b>Self-Supervised Vision:</b> Implement a basic Contrastive Learning setup (SimCLR inspired) using augmentations to learn representations without any labels.</td>
      <td><a href="https://www.cs.toronto.edu/~kriz/cifar.html">CIFAR-10</a></td>
    </tr>
    <tr>
      <td><b>Instance Segmentation:</b> Use a pre-trained Mask R-CNN from <code>torchvision</code> to both detect objects with bounding boxes and generate pixel-perfect masks for each instance.</td>
      <td><a href="https://cocodataset.org/">COCO Dataset</a></td>
    </tr>
    <tr>
      <td><b>Neural Style Transfer:</b> Combine the content of one image with the artistic style of another by minimizing feature map differences across layers of a pre-trained VGG-19 network.</td>
      <td>Any Two Images</td>
    </tr>
    <tr>
      <td rowspan="6"><b>Vision-Text & Generative</b></td>
      <td><b>Descriptive Image Search:</b> Use OpenAI's CLIP model to find images in a large collection by typing natural language queries (e.g., "a dog catching a frisbee").</td>
      <td><a href="https://www.kaggle.com/datasets/adityajn105/flickr8k">Flickr8k</a></td>
    </tr>
    <tr>
      <td><b>Automatic Image Captioning:</b> Combine a vision encoder (like ResNet) with an RNN text decoder to generate a one-sentence description of an image.</td>
      <td><a href="https://www.kaggle.com/datasets/adityajn105/flickr8k">Flickr8k</a></td>
    </tr>
    <tr>
      <td><b>Weather-Adaptive Diffusion:</b> Use an Image-to-Image Diffusion Model to generate "rainy" or "foggy" versions of clear-weather driving images.</td>
      <td><a href="https://www.kaggle.com/datasets/yessicatuteja/foggy-cityscapes-image-dataset">Foggy Cityscapes</a></td>
    </tr>
    <tr>
      <td><b>Stable Diffusion Intro:</b> Use the HuggingFace <code>diffusers</code> library to generate images from text prompts, experimenting with different CFG scales and inference steps.</td>
      <td>HuggingFace API</td>
    </tr>
    <tr>
      <td><b>Visual Question Answering (VQA):</b> Use a pre-trained multimodal model (like BLIP) to pass an image and a text question (e.g., "What color is the car?") and output a text answer.</td>
      <td>HuggingFace API</td>
    </tr>
    <tr>
      <td><b>ControlNet Generation:</b> Use ControlNet alongside Stable Diffusion to generate an image that perfectly follows the pose of a human skeleton or the edges of a Canny map.</td>
      <td>HuggingFace API</td>
    </tr>
    <tr>
      <td rowspan="6"><b>Natural Language Processing</b></td>
      <td><b>Headline Sentiment Scorer:</b> Use standard Text Classification techniques (TF-IDF + Logistic Regression) to label news headlines as "Positive" or "Negative."</td>
      <td><a href="https://www.kaggle.com/datasets/sbatti/financial-sentiment-analysis">Financial Sentiment</a></td>
    </tr>
    <tr>
      <td><b>News Category Classifier:</b> Fine-tune a pre-trained BERT (Encoder) model to categorize news articles into categories like "Tech," "Sports," and "Politics."</td>
      <td><a href="https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification">BBC News</a></td>
    </tr>
    <tr>
      <td><b>Extractive Document Summarizer:</b> Build a Seq2Seq (Encoder-Decoder) model to take long articles and generate a concise summary of the key points.</td>
      <td><a href="https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail">CNN/DailyMail</a></td>
    </tr>
    <tr>
      <td><b>Prompt Engineering & RAG:</b> Build a Retrieval-Augmented Generation pipeline using an open-source LLM (Llama or Mistral) and LangChain to chat with a PDF document.</td>
      <td>Custom PDF Documents</td>
    </tr>
    <tr>
      <td><b>Word Embeddings Math:</b> Train Word2Vec on a custom text corpus. Write a script to visualize cosine similarity and perform vector math (e.g., King - Man + Woman = Queen).</td>
      <td>Any Text Corpus</td>
    </tr>
    <tr>
      <td><b>Named Entity Recognition (NER):</b> Fine-tune a lightweight transformer model (DistilBERT) to extract and classify person names, locations, and dates from raw paragraphs.</td>
      <td><a href="https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus">Kaggle NER</a></td>
    </tr>
    <tr>
      <td rowspan="6"><b>Audio Processing</b></td>
      <td><b>Spoken Command Recognition:</b> Convert audio <code>.wav</code> files into Mel Spectrograms and use a CNN to classify spoken digits (0-9).</td>
      <td><a href="https://www.kaggle.com/datasets/joserzapata/free-spoken-digit-dataset-fsdd">FSDD</a></td>
    </tr>
    <tr>
      <td><b>Automated Transcriber:</b> Use the pre-trained OpenAI Whisper model to transcribe audio clips of various speakers into accurate text.</td>
      <td><a href="https://www.kaggle.com/datasets/pypiahmad/librispeech-asr-corpus">LibriSpeech</a></td>
    </tr>
    <tr>
      <td><b>Multimodal Emotion Recognition:</b> Build a model that takes both audio (tone, pitch) and text (words used) to identify a speaker's emotional state (happy, angry, etc.).</td>
      <td><a href="https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio">RAVDESS</a></td>
    </tr>
    <tr>
      <td><b>Zero-Shot Audio Classification:</b> Use a pre-trained Audio-Text model (like CLAP) to classify environmental sounds without any fine-tuning.</td>
      <td><a href="https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50">ESC-50</a></td>
    </tr>
    <tr>
      <td><b>Music Genre Classification:</b> Extract Mel-Frequency Cepstral Coefficients (MFCCs) from raw audio tracks and train a Random Forest to classify them into 10 musical genres.</td>
      <td><a href="https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification">GTZAN Dataset</a></td>
    </tr>
    <tr>
      <td><b>Text-to-Speech (TTS) Pipeline:</b> Use an open-source HuggingFace model (like VITS or Bark) to build a pipeline that takes raw strings of text and generates human-like audio `.wav` files.</td>
      <td>HuggingFace API</td>
    </tr>
    <tr>
      <td rowspan="6"><b>Generative AI & LLMs</b></td>
      <td><b>Prompt Engineering & Jailbreaks:</b> Use a free API (like Groq or Gemini) to write a script that tests various Prompt Engineering techniques (Few-Shot, Chain-of-Thought). Then, attempt to safely "jailbreak" a local model to understand vulnerability testing.</td>
      <td>API / Local LLM</td>
    </tr>
    <tr>
      <td><b>RAG (Retrieval-Augmented Generation) from Scratch:</b> Build a local RAG pipeline without expensive vector databases. Use <code>sentence-transformers</code> to embed a PDF of the ONIA Syllabus, store it in a local FAISS index, and use an LLM API to answer questions about it.</td>
      <td><a href="https://www.oniabrasil.com.br/assets/files/Syllabus_da_2_ONIA.pdf">ONIA Syllabus PDF</a></td>
    </tr>
    <tr>
      <td><b>Toy VAE (Variational Autoencoder):</b> Build a VAE in PyTorch to generate new, synthetic handwritten digits. Plot the 2D latent space to see how the model transitions smoothly from a "1" to a "7". Runs in 5 minutes on a CPU.</td>
      <td><a href="https://www.kaggle.com/competitions/digit-recognizer">MNIST</a></td>
    </tr>
    <tr>
      <td><b>Micro-GAN for Synthetic Data:</b> Train a Deep Convolutional GAN (DCGAN) on the Fashion-MNIST dataset to generate pictures of synthetic clothes. This teaches the "Adversarial" generator/discriminator loop without needing heavy compute.</td>
      <td><a href="https://www.kaggle.com/datasets/zalando-research/fashionmnist">Fashion MNIST</a></td>
    </tr>
    <tr>
      <td><b>LoRA Fine-Tuning (PEFT):</b> Use the Hugging Face <code>peft</code> and <code>trl</code> libraries on a free Google Colab T4 GPU. Fine-tune a small model (like <code>GPT-2</code> or <code>TinyLlama 1.1B</code>) on a dataset of Medical Q&A using 4-bit quantization (QLoRA) so it fits in memory.</td>
      <td><a href="https://www.kaggle.com/datasets/nimitmak/medical-qa-dataset">Medical Q&A</a></td>
    </tr>
    <tr>
      <td><b>LLM Output Fact-Checker:</b> Create an automated evaluation pipeline. Use a small local LLM to extract factual claims from a generated text, and cross-reference them against a reliable Wikipedia search API to flag "hallucinations" (crucial for ONIA Eixo 8).</td>
      <td>Wikipedia API</td>
    </tr>
  </tbody>
</table>

## License
This repository is under the GPL 3.0 License from June, 29th 2007.