# ThaiSarc_V1: A Comprehensive Dataset for Sarcasm Detection in Thai Political News Headlines

Welcome to **ThaiSarc V1**, a groundbreaking resource specifically designed for sarcasm detection in Thai political news headlines. This project represents a significant milestone in Natural Language Processing (NLP) for Thai-language tasks, combining state-of-the-art deep learning and generative AI models to address the unique challenges of sarcasm identification in the Thai context. ThaiSarcV1 not only contributes to the advancement of computational linguistics but also fosters media literacy by enabling nuanced understanding of sarcastic language in political discourse.

---

## 🌟 Key Features
- **Balanced Dataset**: ThaiSarcV1 consists of 1,128 carefully curated Thai political news headlines, evenly split into 564 sarcastic and 564 non-sarcastic samples to ensure unbiased analysis.
- **High Annotation Reliability**: A mutual agreement rate of 92% among annotators highlights the robustness of the dataset.
- **Novel Dataset**: The first publicly available dataset focusing on sarcasm detection in Thai political news headlines, offering a valuable resource for Thai NLP research.
- **Comprehensive Benchmarks**: Evaluation includes cutting-edge discriminative models (e.g., WangchanBERTa, BERT-base) and generative models (e.g., fine-tuned GPT-4o).
- **Broad Applications**: Designed for sentiment analysis, content moderation, and improving media literacy.

---

## 📋 Dataset Overview

| **Class**        | **Number of Items** | **Total Words** | **Average Length** | **Min Length** | **Max Length** |
|-------------------|---------------------|-----------------|---------------------|----------------|----------------|
| **Sarcastic**     | 564                 | 8,809           | 16.54              | 7              | 27             |
| **Non-Sarcastic** | 564                 | 8,250           | 15.47              | 8              | 28             |

The dataset was collected on **May 4, 2025**, using [WebScraper.io](https://webscraper.io) from 12 prominent Thai news agency websites. Each headline was reviewed, annotated, and validated for sarcasm using an adaptation of the sarcasm detection framework by Hiai and Shimada (2016). The careful preprocessing and annotation processes ensure high-quality data for training and testing models.

### 📰 News Agencies

ThaiSarc V1 headlines were sourced from the following news agencies:

| **News Agency**     | **Website URL**                                               |
|---------------------|-------------------------------------------------------------|
| ไทยรัฐ (Thairath)      | [https://www.thairath.co.th/news/politic](https://www.thairath.co.th/news/politic) |
| มติชน (Matichon)      | [https://www.matichon.co.th/politics](https://www.matichon.co.th/politics) |
| ประชาไท (Prachatai)    | [https://prachatai.com/category/politics](https://prachatai.com/category/politics) |
| The Matter           | [https://thematter.co/category/social/politics](https://thematter.co/category/social/politics) |
| ไทยพีบีเอส (Thai PBS)  | [https://www.thaipbs.or.th/news/politics](https://www.thaipbs.or.th/news/politics) |
| MCOT                | [https://tna.mcot.net/category/politics](https://tna.mcot.net/category/politics) |
| เดลินิวส์ (Daily News) | [https://www.dailynews.co.th/news_group/politics](https://www.dailynews.co.th/news_group/politics) |
| ข่าวสด (Khaosod)      | [https://www.khaosod.co.th/politics](https://www.khaosod.co.th/politics) |
| ไทยโพสต์ (Thai Post)  | [https://www.thaipost.net/politics](https://www.thaipost.net/politics) |
| แนวหน้า (Naewna)      | [https://www.naewna.com/politics](https://www.naewna.com/politics) |
| ผู้จัดการ (MGR Online)| [https://mgronline.com/politics](https://mgronline.com/politics) |
| อีจัน (Ejan)         | [https://www.ejan.co/category/politics](https://www.ejan.co/category/politics) |

---

## 📥 Download and Usage

Access ThaiSarc V1 on GitHub: [ThaiSarcV1 GitHub Repository](https://github.com/KunakornMart/ThaiSarcV1)

### Steps to Use the Dataset:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/KunakornMart/ThaiSarcV1.git
   ```

2. **Load the Dataset**:
   ```python
   import pandas as pd

   # Load the dataset
   data = pd.read_csv('path/to/ThaiSarcV1.csv')

   # Display the first 5 rows
   print(data.head())
   ```

3. **Preprocess and Tokenize**:
   Utilize libraries such as **PyThaiNLP** for tokenization and processing Thai text effectively.

---

## 🎯 Objectives

The primary goal of this project is to compare the efficiency of **discriminative deep learning models** (e.g., WangchanBERTa) and **generative AI models** (e.g., GPT-4o) in sarcasm detection, providing a comprehensive evaluation of their strengths and weaknesses.

This study also aims to answer key research questions:
1. How effectively do discriminative models perform in sarcasm detection for Thai political news headlines?
2. Can generative AI models, such as GPT-4o, surpass discriminative models in identifying sarcasm?
3. What are the implications of applying these models to real-world tasks, such as sentiment analysis and misinformation detection?
   
---

## 🔧 Methodology

### Data Preparation
- Tokenization: Implemented with **NewMM** (PyThaiNLP), achieving **93% accuracy** and processing 1,128 headlines in **3.28 seconds**.
- Preprocessing: Headlines were normalized and encoded for model training.

### Annotation Process
- Conducted by four trained annotators.
- Achieved high inter-annotator reliability (92%) through consensus discussions.

### Experimental Settings
- **Discriminative Models**: Implemented using TensorFlow 2.17.1 on Google Colab with GPU acceleration.
- **Generative Models**: Fine-tuned on GPT-4o and GPT-4.1 (latest version) for sarcasm detection.

---

## 🚀 Models and Benchmarks

### Evaluation Metrics
Key metrics included **Accuracy**, **Precision**, **Recall**, and **F1-Score**, ensuring a comprehensive analysis of model performance.

## 📊 Results

### Discriminative Models Performance (Encoder)
| Model            | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|-------------------|-------------|---------------|------------|--------------|
| WangchanBERTa (Monolingual)    | 80.59       | 80.69         | 80.59      | 80.59        |
| BERT-base (Multilingual)       | 65.88       | 66.63         | 65.88      | 65.50        |

### Generative  Models  Performance   (Decoder)
| Model                       | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|-----------------------------|-------------|---------------|------------|--------------|
| Fine-Tuned GPT-4o-mini      | 85.29       | 79.41         | 95.29      | 86.63        |
| Fine-Tuned GPT-4o           | 83.53       | 78.22         | 92.94      | 84.95        |
| Fine-Tuned GPT-4.1          | 82.94       | 78.00         | 91.76      | 84.32        |
| Fine-Tuned GPT-4.1-mini     | 82.35       | 76.70         | 92.94      | 84.04        |

---

## 📈 Applications

- **Media Literacy**: Enhances understanding of sarcasm in Thai political discourse.
- **NLP Research**: Serves as a benchmark for testing and improving sarcasm detection models.
- **Content Moderation**: Assists in filtering sarcastic or misleading content in online platforms.
- **Sentiment Analysis**: Supports improved sentiment categorization in Thai language processing.
- **Combatting Misinformation**: Provides tools to identify sarcasm in political narratives, reducing the spread of false information.
- **Future Work**: Potential areas include expanding the dataset, incorporating multimodal sarcasm detection (e.g., combining text and images), and applying models to broader domains beyond politics.

---

## 👥 Contributors
- **Kunakorn Pruksakorn**
- **Niwat Wuttisrisiriporn**
- **Narongkiat Phuengwong**
- **Advisor : Ohm Sornil**

---

## 📧 Contact
For inquiries, please contact [KunakornMart](https://github.com/KunakornMart).
