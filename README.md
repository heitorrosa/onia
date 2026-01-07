# deeplearning
A repository dedicated to study for the 2nd Olimpíada Nacional de Inteligência Artificial (ONIA)

* [IOIA UAE Syllabus](https://ioai-official.org/wp-content/uploads/2025/10/Syllabus.pdf)
* [2nd ONIA Syllabus](https://www.oniabrasil.com.br/assets/files/Syllabus_da_2_ONIA.pdf)

## Study Plan (3rd phase)

| Eixo Temático (Theme) | Ciclo Preparatório: What to Do | Recommended Exercises & Projects |
|---|---|---|
| **1. Letramento & Pensamento Computacional** | Focus on advanced logic, identifying errors (debugging), and complex structures in Python. | **Programming Fundamentals:** Build tools using Pandas and Matplotlib to plot stock moving averages. |
| **2. Fundamentos da IA** | Describe the full pipeline (data collection to monitoring) and recognize how IOAI topics connect to real problems. | **Data Science Fundamentals:** Build an "Evaluation Suite" to calculate Confusion Matrices and ROC Curves. |
| **3. Dados: O Combustível da IA** | Apply descriptive statistics (mean, SD), manipulate datasets with Pandas, and manage train/validation/test splits. | **Data Science Fundamentals:** Hyperparameter Grid Search and Automated Feature Engineering. |
| **4. IA no Cotidiano & Impacto** | Analyze deepfakes, labor market automation, and international policy guidelines like UNESCO's. | **NLP / Advanced Vision:** Use BERT for news categorization or YOLO for traffic sign detection. |
| **5. Machine Learning Clássico** | Implement Scikit-learn models (Linear/Logistic Regression, k-NN, SVM, Random Forests) and cross-validation. | **Supervised Learning:** "Bull or Bear" Classifier (Decision Trees/SVM) and Linear Regression with L1/L2. |
| **6. Deep Learning & Redes Neurais** | Implement MLPs and CNNs with PyTorch; study convolution filters, Transformers, and overfitting. | **Neural Network Mechanics:** "XOR" Trader from scratch and comparing Optimizers (Adam vs. SGD). |
| **7. IA Generativa & LLMs** | Develop clear prompts (zero-shot, chain-of-thought) and perform critical fact-checking of AI outputs. | **Vision-Text & Generative:** Automatic Image Captioning or using CLIP for descriptive image searches. |
| **8. Ética, Legislação & Futuro** | Study algorithmic bias, data protection (LGPD), and the impact of AI on career trajectories. | **Regularization & NLP:** Use LoRA (PEFT) to adapt BERT for sector-specific sentiment analysis. |

## Exercises (IOIA Syllabus)

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Projects</th>
      <th>Dataset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><b>Programming Fundamentals</b></td>
      <td>Build a tool using Pandas and Matplotlib to plot 50-day vs 200-day moving averages for any stock in the S&P 500.</td>
      <td><a href="https://www.kaggle.com/datasets/camnugent/sandp500">Link</a></td>
    </tr>
    <tr>
      <td>Use NumPy to calculate a correlation matrix of daily returns for 50 different stocks to see which sectors move together.</td>
      <td><a href="https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs">Link</a></td>
    </tr>
    <tr>
      <td>Real-time Market Dashboard. Create a Seaborn dashboard that visualizes "candlestick" patterns alongside trading volume spikes to identify high-volatility events.</td>
      <td><a href="https://www.kaggle.com/datasets/s3programmer/stock-market-dataset-for-financial-analysis">Link</a></td>
    </tr>
    <tr>
      <td rowspan="3"><b>Supervised Learning</b></td>
      <td>Use Linear Regression with L1/L2 Regularization to predict the next day's closing price based on the previous five days.</td>
      <td><a href="https://www.kaggle.com/datasets/ziya07/financial-market-forecasting-dataset">Link</a></td>
    </tr>
    <tr>
      <td>The "Bull or Bear" Classifier. Implement a Decision Tree and an SVM to classify if a stock will rise by >2% tomorrow. Compare their performance.</td>
      <td><a href="https://www.kaggle.com/datasets/s3programmer/stock-market-dataset-for-predictive-analysis">Link</a></td>
    </tr>
    <tr>
      <td>XGBoost Market Ensemble. Build a Random Forest and a Gradient Boosting model (XGBoost) to predict stock volatility during "Market Stress" events.</td>
      <td><a href="https://www.kaggle.com/datasets/ziya07/financial-market-forecasting-dataset">Link</a></td>
    </tr>
    <tr>
      <td rowspan="3"><b>Unsupervised Learning</b></td>
      <td>Sector Clustery. Use K-Means Clustering to group stocks based solely on their P/E ratios and dividend yields to see if they match their official industry sectors.</td>
      <td><a href="https://www.kaggle.com/datasets/samayashar/stock-market-simulation-dataset">Link</a></td>
    </tr>
    <tr>
      <td>Portfolio Dimension Reduction. Use PCA to reduce 30 different technical indicators into 3 principal components and visualize them in a 3D plot.</td>
      <td><a href="https://www.kaggle.com/datasets/s3programmer/stock-market-dataset-for-financial-analysis">Link</a></td>
    </tr>
    <tr>
      <td>Market Regime Mapping. Apply t-SNE or UMAP to 20 years of daily macroeconomic data to visualize different "economic regimes" (Recessions vs. Booms).</td>
      <td><a href="https://www.kaggle.com/datasets/sumedh1507/us-macro-data-25yrs">Link</a></td>
    </tr>
    <tr>
      <td rowspan="3"><b>Data Science Fundamentals</b></td>
      <td>The Evaluation Suite. Build a model and calculate its Confusion Matrix, F1-Score, and ROC Curve for predicting market "Event Flags."</td>
      <td><a href="https://www.kaggle.com/datasets/ziya07/financial-market-forecasting-dataset">Link</a></td>
    </tr>
    <tr>
      <td>Hyperparameter Grid Search. Perform a Cross-Validation grid search to find the optimal hyperparameters for a K-NN stock classifier.</td>
      <td><a href="https://www.kaggle.com/datasets/s3programmer/stock-market-dataset-for-predictive-analysis">Link</a></td>
    </tr>
    <tr>
      <td>Automated Feature Engineering. Create a pipeline that generates 100+ "Technical Indicator" features (RSI, MACD, etc.) and uses a feature-selection method to pick the top 10.</td>
      <td><a href="https://www.kaggle.com/datasets/s3programmer/stock-market-dataset-for-financial-analysis">Link</a></td>
    </tr>
    <tr>
      <td rowspan="3"><b>Neural Network Mechanics</b></td>
      <td>The "XOR" Trader. Code a 2-layer neural network from scratch in NumPy to solve a simple "AND/OR" trading rule (e.g., If Volume is High AND Price is Low).</td>
      <td><a href="https://www.kaggle.com/datasets/anitarostami/historical-stock-price-dataset">Link</a></td>
    </tr>
    <tr>
      <td>In PyTorch, build a model with a custom Loss Function that penalizes large losses (downside risk) more than it rewards gains.</td>
      <td><a href="https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs">Link</a></td>
    </tr>
    <tr>
      <td>Optimizer Battle. Compare Adam, AdamW, and SGD when training an MLP to predict "Market Stress Levels."</td>
      <td><a href="https://www.kaggle.com/datasets/ziya07/financial-market-forecasting-dataset">Link</a></td>
    </tr>
    <tr>
      <td rowspan="3"><b>Deep Learning Architectures</b></td>
      <td>Multi-Layer Economic Predictor. Build a deep MLP in PyTorch to predict GDP growth from 25 years of monthly indicators.</td>
      <td><a href="https://www.kaggle.com/datasets/sumedh1507/us-macro-data-25yrs">Link</a></td>
    </tr>
    <tr>
      <td>Ticker Embedding Space. Create a model that learns Embeddings for stock tickers. Use t-SNE to see if the model puts "AAPL" and "MSFT" near each other.</td>
      <td><a href="https://www.kaggle.com/datasets/samayashar/stock-market-simulation-dataset">Link</a></td>
    </tr>
    <tr>
      <td>Self-Attention Stock Scorer. Implement a basic Attention Mechanism that weighs which historical days are most important for today's prediction.</td>
      <td><a href="https://www.kaggle.com/datasets/ziya07/financial-market-forecasting-dataset">Link</a></td>
    </tr>
    <tr>
      <td rowspan="3"><b>Regularization & Optimization</b></td>
      <td>Dropout Experiment. Train two models—one with Dropout and one without—on volatile crypto data to see which generalizes better.</td>
      <td><a href="https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data">Link</a></td>
    </tr>
    <tr>
      <td>Batch Norm Stability. Test how Batch Normalization affects the training speed of a deep MLP used for global exchange rate prediction.</td>
      <td><a href="https://www.kaggle.com/datasets/sazidthe1/global-economic-monitor">Link</a></td>
    </tr>
    <tr>
      <td>Use Parameter-Efficient Fine-Tuning (LoRA) to adapt a pre-trained BERT model specifically to identify sentiment in "Green Energy" sector headlines.</td>
      <td><a href="https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests">Link</a></td>
    </tr>
    <tr>
      <td rowspan="3"><b>Computer Vision Fundamentals</b></td>
      <td>Hand-Drawn Digit Classifier. Build a CNN to identify digits (0-9). This is the classic "Hello World" of CV.</td>
      <td><a href="https://www.kaggle.com/competitions/digit-recognizer">Link</a></td>
    </tr>
    <tr>
      <td>Satellite Poverty Map. Train a CNN to classify satellite images of regions into different wealth brackets (proxy for GDP).</td>
      <td><a href="https://www.kaggle.com/datasets/sandeshbhat/satellite-images-to-predict-povertyafrica">Link</a></td>
    </tr>
    <tr>
      <td>Natural Scene Classification. Train a model to distinguish between 6 categories of landscapes (buildings, forests, glaciers, mountains, sea, streets).</td>
      <td><a href="https://www.kaggle.com/datasets/puneet6060/intel-image-classification">Link</a></td>
    </tr>
    <tr>
      <td rowspan="3"><b>Advanced Vision</b></td>
      <td>Traffic Sign Detection. Use a pre-trained YOLO model to identify and locate various traffic signs in diverse street scenes.</td>
      <td><a href="https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign">Link</a></td>
    </tr>
    <tr>
      <td>Urban Scene Segmentation. Implement a U-Net to perform pixel-level segmentation on city street images, separating "road," "sidewalk," and "car."</td>
      <td><a href="https://www.kaggle.com/datasets/dansbecker/cityscapes-image-pairs">Link</a></td>
    </tr>
    <tr>
      <td>Realistic Face Generation. Use a GAN (like DCGAN) to generate high-resolution synthetic faces that look realistic enough to pass basic visual checks.</td>
      <td><a href="https://www.kaggle.com/datasets/jessicali9530/celeba-dataset">Link</a></td>
    </tr>
    <tr>
      <td rowspan="3"><b>Vision-Text & Generative</b></td>
      <td>Descriptive Image Search. Use CLIP to find images in a large collection by typing general natural language queries like "a sunset over a city."</td>
      <td><a href="https://www.kaggle.com/datasets/adityajn105/flickr8k">Link</a></td>
    </tr>
    <tr>
      <td>Automatic Image Captioning. Combine a vision encoder with a text decoder to generate a one-sentence description of an image.</td>
      <td><a href="https://www.kaggle.com/datasets/adityajn105/flickr8k">Link</a></td>
    </tr>
    <tr>
      <td>Weather-Adaptive Augmentation. Use a Diffusion Model to generate "rainy" or "foggy" versions of clear-weather driving images to train a more robust autonomous vehicle classifier.</td>
      <td><a href="https://www.kaggle.com/datasets/yessicatuteja/foggy-cityscapes-image-dataset">Link</a></td>
    </tr>
    <tr>
      <td rowspan="3"><b>NLP</b></td>
      <td>Headline Sentiment Scorer. Use Text Classification to label financial news as "Positive" or "Negative."</td>
      <td><a href="https://www.google.com/search?q=https://www.kaggle.com/datasets/sbatti/financial-sentiment-analysis">Link</a></td>
    </tr>
    <tr>
      <td>News Category Classifier with BERT. Fine-tune a pre-trained BERT model to categorize news articles into "Tech," "Sports," "Politics," etc.</td>
      <td><a href="https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification">Link</a></td>
    </tr>
    <tr>
      <td>Extractive Document Summarizer. Build an Encoder-Decoder model that takes long news articles and generates a concise summary of the key points.</td>
      <td><a href="https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail">Link</a></td>
    </tr>
    <tr>
      <td rowspan="3"><b>Audio Processing</b></td>
      <td>Spoken Command Recognition. Use the audio equivalent of MNIST to classify spoken digits into their numerical values.</td>
      <td><a href="https://www.kaggle.com/datasets/joserzapata/free-spoken-digit-dataset-fsdd">Link</a></td>
    </tr>
    <tr>
      <td>Automated Podcast Transcriber. Use OpenAI Whisper to transcribe audio clips from various speakers into accurate text.</td>
      <td><a href="https://www.kaggle.com/datasets/pypiahmad/librispeech-asr-corpus">Link</a></td>
    </tr>
    <tr>
      <td>Multimodal Emotion Recognition. Build a model that takes both audio (tone, pitch) and text (words used) to identify the speaker's emotional state (e.g., happy, sad, angry).</td>
      <td><a href="https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio">Link</a></td>
    </tr>
  </tbody>
</table>

## License
This repository is under the GPL 3.0 License from June, 29th 2007.