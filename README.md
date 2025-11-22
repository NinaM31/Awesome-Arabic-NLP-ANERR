<h1 align="center">
  ANERR – Arabic NLP Essential Resource Repository  
  <br>
  <strong> مصادر معالجة اللغة العربية – أَنِير</strong>
</h1>
<p align="center">
  <img src="https://github.com/user-attachments/assets/d5c4a90c-2a75-4cba-9577-c2423d28e50a?raw=true" width="260px" alt="ANERR logo">
</p>

<p align="center">
  <strong>أَنِير طريق بحثك في معالجة اللغة العربية </strong><br>
  Illuminate your Arabic NLP research with structured, curated resources
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Paper_Count-222-6A737D?style=for-the-badge">
  <img src="https://img.shields.io/badge/Tasks-10-6A737D?style=for-the-badge">
  <img src="https://img.shields.io/github/contributors/NinaM31/Awesome-Arabic-NLP-ANERR?style=for-the-badge&color=6A737D">
  <img src="https://img.shields.io/badge/License-MIT-6A737D?style=for-the-badge">
</p>

---

## 🔦 ما هو مستودع أَنِير (ANERR)؟
مستودع شامل ومنظم يضم أوراقاً بحثية، دراسات، مسوحات، مجموعات بيانات، أدوات، ونماذج خاصة بمعالجة اللغة العربية الطبيعية.  
تم بناء ANERR ليكون **خريطة بحثية مضيئة** تساعد الطلاب والباحثين والمهتمين بمجال الـNLP العربي على الوصول للمعلومات بسرعة ووضوح.

---

## Why ANERR?
- 200+ papers **with year, task, venue**
- Includes **datasets, corpora, tools, lexicons, and benchmarks**
- Covers modern topics: **RAG, Arabic LLMs, reasoning, multimodal, QA**
- Designed for **researchers, students, and practitioners**
- More structured and detailed than traditional "paper list" repos

Looking to write a survey paper yourself?  
👉 Check our guide: [How to Write a Systematic Literature Review (SLR)](How-to-Write-a-Survey-Paper/README.md)

---

## Table of Contents
- [Text Diacritization](#arabic-text-diacritization)
- [Text Summarization](#arabic-text-summarization)
- [Word Disambiguation](#arabic-word-disambiguation)
- [Sentiment Analysis](#arabic-sentiment-analysis)
- [Question-Answering Systems](#arabic-question-answering-systems)
- [Retrieval Augmented Generation](#arabic-retrieval-augmented-generation)
- [Information Retrieval](#arabic-information-retrieval)
- [Image Captioning](#arabic-image-captioning)
- [Named Entity Recognition](#arabic-named-entity-recognition)
- [LLM](#arabic-llm)

## Arabic Text Diacritization
![ATD:](https://img.shields.io/badge/Papers-25-6A737D?style=for-the-badge)

### Corpus
| Title                                 |Corpus Link                                                              |Paper Link                                                                | Type                  | Availability|Tokens|
| --------------------------------------| ------------------------------------------------------------------------| ------------------------------------------------------------------------ |-----------------------|-------------|-----|
|OpenITI                                |[github](https://github.com/OpenITI/RELEASE)                             | N/A                                                                      |Classical Arabic       |Free         |2.2B |
|Tashkeela                              |[kaggle](https://www.kaggle.com/datasets/linuxscout/tashkeela/data)      |[ELSEVIER](https://doi.org/10.1016/j.dib.2017.01.011)                     |Mostly Classical Arabic|Free         |75.6M|
|ATB                                    |[LDC](https://catalog.ldc.upenn.edu/LDC2003T06)                          | N/A                                                                      |Modern Standard Arabic |Commercial   |1M   |
|Tanzil Quran text                      |[tanzil](https://tanzil.net/download/)                                   | N/A                                                                      |Classical Arabic       |Free         |78K  |
|Wikinews                               |[github](https://github.com/kdarwish/Farasa)                             | N/A                                                                      |Modern Standard Arabic |Free         |18K  |

### Benchmark Dataset
| Title                               |Corpus Link                                                                                           |Paper Link                                                                | Type                  | Availability|Train |Valid |Test |
| ------------------------------------| ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |-----------------------|-------------|------|------|-----|
|Tashkeela variant by  Fadel et al.   |[github](https://github.com/AliOsm/arabic-text-diacritization/tree/master/dataset)                    |[IEEE](https://ieeexplore.ieee.org/document/8769512)                      |Mostly Classical Arabic|Free         |24.5M |102K  |107K |
|Tashkeela variant by  Abbad & Xiong  |[kaggle](https://www.kaggle.com/datasets/hamzaabbad/tashkeela-processed-fully-diacritized-arabic-text)|[Springer](https://link.springer.com/chapter/10.1007/978-3-030-45439-5_23)|Mostly Classical Arabic|Free         |31.7M |1.7M  |1.7M |
|CATT benchmark                       |[github](https://github.com/abjadai/catt/releases)                                                    |[ACL](https://doi.org/10.18653/v1/2024.arabicnlp-1.21)                    |Modern Standard Arabic |Free         |-     |120K  |125K |


### Studies & Surveys
| Title                                                                                        |Year| Link                                                |
| -------------------------------------------------------------------------------------------- |----|---------------------------------------------------  |
| Arabic text diacritization using transformers: a comparative study                           |2025|[IAES](http://doi.org/10.11591/ijai.v14.i1.pp702-711)|
| Literature Review of Automatic Restoration of Arabic Diacritics                              |2023|[IEEE](https://ieeexplore.ieee.org/document/10424191)|
| A Comparative Study of Some Automatic Arabic Text Diacritization Systems                     |2022|[WILEY](https://onlinelibrary.wiley.com/doi/10.1155/2022/3613710)|
| Automatic Methods and Neural Networks in Arabic Texts Diacritization: A Comprehensive Survey |2021|[IEEE](https://ieeexplore.ieee.org/document/9585619) |
| Arabic text diacritization: overview and solution                                            |2019|[ACM](https://dl.acm.org/doi/10.1145/3368756.3369088)|
| Arabic Text Diacritization Using Deep Neural Networks                                        |2019|[IEEE](https://ieeexplore.ieee.org/document/8769512) |
| A survey of automatic Arabic diacritization techniques                                       |2015|[CUP](https://www.cambridge.org/core/journals/natural-language-engineering/article/abs/survey-of-automatic-arabic-diacritization-techniques/8D869B09FE33D60F3C801A97AFD6C5F1) |


### Papers
| Title                                                                                                          | Year    |   Link                                                         |
| -------------------------------------------------------------------------------------------------------------- | ------- | -------------------------------------------------------------- |
| CATT: Character-based Arabic Tashkeel Transformer                                                              | 2024    | [ACL](https://aclanthology.org/2024.arabicnlp-1.21/)           |
| BERT-Based Arabic Diacritization: A state-of-the-art approach for improving text accuracy and pronunciation    | 2024    | [ELSEVIER](https://www.sciencedirect.com/science/article/abs/pii/S0957417424002811)|
| Arabic Text Diacritization In The Age Of Transfer Learning: Token Classification Is All You Need               | 2024    | [arXiv](https://arxiv.org/abs/2401.04848)                      |
| Neural Arabic Text Diacritization: State of the Art Results and a Novel Approach for Machine Translation       | 2021    | [arXiv](https://arxiv.org/abs/1911.03531)                      |
| Simple Extensible Deep Learning Model for Automatic Arabic Diacritization                                      | 2021    | [ACM](https://dl.acm.org/doi/abs/10.1145/3480938)              |
| A Deep Belief Network Classification Approach for Automatic Diacritization of Arabic Text                      | 2021    | [ApplSci](https://www.mdpi.com/2076-3417/11/11/5228)  |
| LAMAD: A Linguistic Attentional Model for Arabic Text Diacritization                                           | 2021    | [ACL](https://aclanthology.org/2021.findings-emnlp.317/)       |
| Effective Deep Learning Models for Automatic Diacritization of Arabic Text                                     | 2020    | [IEEE](https://ieeexplore.ieee.org/document/9274427)           |
| Multi-components System for Automatic Arabic Diacritization                                                    | 2020    | [Springer](https://link.springer.com/chapter/10.1007/978-3-030-45439-5_23) |
| Towards An Open Platform For Arabic Language Processing                                                        | 2020    | [RG](https://www.researchgate.net/publication/342673690_Towards_An_Open_Platform_For_Arabic_Language_Processing) |
| Diacritizing Arabic Text Using a Single Hidden Markov Model                                                    | 2018    | [IEEE](https://ieeexplore.ieee.org/document/8402214)            |
| SHAKKIL: An Automatic Diacritization System for Modern Standard Arabic Texts                                   | 2017    | [ACL](https://aclanthology.org/W17-1311/)                       |
| Arabic Diacritization with Recurrent Neural Networks                                                           | 2015    | [ACL](https://aclanthology.org/D15-1274/)                       |

### Tools
| Tool              | Open Source |   Tool Link                                                    | Paper Link                                                                |
| ----------------- | ----------- |--------------------------------------------------------------- |-------------------------------------------------------------------------- |
|CAMEL              |✔️          |[github](https://github.com/CAMeL-Lab/camel_tools)               |[ACL](https://www.aclweb.org/anthology/2020.lrec-1.868)                    |
|Mishkal            |✔️          |[github](https://github.com/linuxscout/mishkal)                  |[RG](http://dx.doi.org/10.13140/RG.2.2.29882.82881)                        |
|Shakkala           |✔️          |[github](https://github.com/Barqawiz/Shakkala)                   |N/A                                                                        |
|Tashkeela-model    |✔️          |[github](https://github.com/Anwarvic/Arabic-Tashkeela-Model)     |N/A                                                                        |
|Libtashkeel        |✔️          |[github](https://github.com/mush42/libtashkeel)                  |N/A                                                                        |
|Farasa             |✖️          |[farasa](https://farasa.qcri.org/)                               |[ACL](https://doi.org/10.18653/v1/N16-3003)                                |
|MADAMIRA           |✖️          |[columbia tech](https://inventions.techventures.columbia.edu/technologies/arabic-language--CU14012)|[ACL](https://aclanthology.org/L14-1479/)|

## Arabic Text Summarization
![ATS:](https://img.shields.io/badge/Papers-27-6A737D?style=for-the-badge)
### Studies & Surveys
| Title                                                                                                          |Year| Link                                                                        |
| ---------------------------------------------------------------------------------------------------------------|----|-----------------------------------------------------------------------------|
|A Review of Arabic Text Summarization: Methods, Datasets, and Evaluation Metrics, With a Proposed Solution      |2025|[IEEE](https://doi.org/10.1109/ACCESS.2025.3584855)                          |
|Abstractive text summarization using deep learning models: a survey                                             |2025|[Springer](https://link.springer.com/article/10.1007/s41060-025-00743-w)     |
|Current Trends and Advances in Extractive Text Summarization: A Comprehensive Review                            |2025|[IEEE](https://doi.org/10.1109/ACCESS.2025.3538886)                          |
|Exploring Advances in Arabic Long-Text Summarization Strategies: A Survey                                       |2024|[FCIHIB](https://doi.org/10.21608/fcihib.2024.258854.1103)                   |
|A survey of text summarization and Headline Generation methods in Arabic                                        |2024|[ACM](https://doi.org/10.1145/3674029.3674078)                               |
|Arabic Text Summarization Challenges using Deep Learning Techniques: A Review                                   |2023|[IJRITCC](http://dx.doi.org/10.17762/ijritcc.v11i11s.8079)                   |
|A Comprehensive Review of Arabic Text Summarization                                                             |2022|[IEEE](https://doi.org/10.1109/ACCESS.2022.3163292)                          |
|Deep Transformer Language Models for Arabic Text Summarization: A Comparison Study                              |2022|[ApplSci](https://doi.org/10.3390/app122311944)                              |
|Text Summarization: A Brief Review                                                                              |2019|[Springer](https://link.springer.com/chapter/10.1007/978-3-030-34614-0_1)    |
|Automatic Arabic Summarization: A survey of methodologies and systems                                           |2017|[ELSEVIER](https://doi.org/10.1016/j.procs.2017.10.088)                      |
|Automatic Arabic text summarization: a survey                                                                   |2016|[Springer](https://link.springer.com/article/10.1007/s10462-015-9442-x)      |

### Papers 
| Title                                                                                                                      |Year|   Link                                                                      |
| ---------------------------------------------------------------------------------------------------------------------------|----| ----------------------------------------------------------------------------|
|Arabic News Text Summarization: An Extractive Technique                                                                     |2025|[IEEE](https://doi.org/10.1109/ICIT64950.2025.11049137)                      |
|Enhanced model for abstractive Arabic text summarization using natural language generation and named entity recognition     |2025|[Springer](https://link.springer.com/article/10.1007/s00521-024-10949-x)     |
|Extractive Arabic Text Summarization-Graph-Based Approach                                                                   |2023|[ELECTRONICS](https://doi.org/10.3390/electronics12020437)                   |
|A Hybrid Arabic Text Summarization Approach based on Transformers                                                           |2022|[IEEE](https://doi.org/10.1109/MIUCC55081.2022.9781694)                      |
|Abstractive Arabic Text Summarization Based on Deep Learning                                                                |2022|[WILEY](https://doi.org/10.1155/2022/1566890)                                |
|Unsupervised neural networks for automatic Arabic text summarization using document clustering and topic modeling           |2021|[ELSEVIER](https://doi.org/10.1016/j.eswa.2021.114652)                       |
|An automatic arabic text summarization system based on genetic algorithms                                                   |2021|[ELSEVIER](https://doi.org/10.1016/j.procs.2021.05.083)                      |
|An efficient single document Arabic text summarization using a combination of statistical and semantic features             |2021|[JKSUCI](https://doi.org/10.1016/j.jksuci.2019.03.010)                       |
|BERT Fine-tuning For Arabic Text Summarization                                                                              |2020|[arXiv](https://doi.org/10.48550/arXiv.2004.14135)                           |
|Automatic Arabic Text Summarization Using Analogical Proportions                                                            |2020|[Springer](https://link.springer.com/article/10.1007/s12559-020-09748-y)     |
|Extractive Multi-Document Arabic Text Summarization Using Evolutionary Multi-Objective Optimization With K-Medoid Clustering|2020|[IEEE](https://doi.org/10.1109/ACCESS.2020.3046494)                          |
|Multidocument Arabic Text Summarization Based on Clustering and Word2Vec to Reduce Redundancy                               |2020|[INFO](https://doi.org/10.3390/info11020059)                                 |
|Extractive Arabic Text Summarization Using Modified PageRank Algorithm                                                      |2020|[JEIJ](https://doi.org/10.1016/j.eij.2019.11.001)                            |
|Arabic text summarization using deep learning approach                                                                      |2020|[Springer](https://link.springer.com/article/10.1186/s40537-020-00386-7)     |
|Arabic Text Summarization Using AraBERT Model Using Extractive Text Summarization Approach                                  |2020|[IJAISR](http://ijeais.org/wp-content/uploads/2020/8/IJAISR200802.pdf)       |
|An abstractive Arabic text summarizer with user controlled granularity                                                      |2018|[ELSEVIER](https://doi.org/10.1016/j.ipm.2018.06.002)                        |

## Arabic Word Disambiguation
![ATD:](https://img.shields.io/badge/Papers-23-6A737D?style=for-the-badge)
### Studies & Surveys
| Title                                                                                                  |Year| Link                                                                      |
| ------------------------------------------------------------------------------------------------------ |----|-------------------------------------------------------------------------  |
|Exploring the Impact of Stop Words and Particles on Arabic Word Sense Disambiguation                    |2025| [Springer](https://link.springer.com/chapter/10.1007/978-3-031-80438-0_3) |
|State-of-the-Art Approaches to Word Sense Disambiguation: A Multilingual Investigation                  |2024| [Springer](https://link.springer.com/chapter/10.1007/978-3-031-57624-9_10)|
|Evaluation of Stacked Embeddings for Arabic Word Sense Disambiguation                                   |2023| [PKP](https://doi.org/10.13053/cys-27-2-4281 )                            |
|Comparative Analysis of Recurrent Neural Network Architectures for Arabic Word Sense Disambiguation     |2022| [SciTePress](http://dx.doi.org/10.5220/0011527600003318)                  |
|A comprehensive review on Arabic word sense disambiguation for natural language processing applications |2022| [WIREs](https://doi.org/10.1002/widm.1447)                                |
|Arabic word sense disambiguation: a review                                                              |2019| [ACM](https://doi.org/10.1007/s10462-018-9622-6)                          |
|Arabic Word Sense Disambiguation - Survey                                                               |2017| [IEEE](https://ieeexplore.ieee.org/document/8250294)                      |
|An Introduction to Arabic Word Sense Disambiguation                                                     |2013| [Thesis](https://www.researchgate.net/publication/342927271_An_Introduction_to_Arabic_Word_Sense_Disambiguation) |

### Papers
| Title                                                                                                               |Year|   Link                                                                    |
| ------------------------------------------------------------------------------------------------------------------- |----| ------------------------------------------------------------------------- |
| Enhancing Arabic Word Sense Disambiguation with Ensemble BERT-Based Models                                          |2025| [Springer](https://link.springer.com/chapter/10.1007/978-3-031-79164-2_23)|
| ArabicNLU 2024: The First Arabic Natural Language Understanding Shared Task                                         |2024| [ACL](https://aclanthology.org/2024.arabicnlp-1.30/)                      |
| EnhancedBERT: A python software tailored for arabic word sense disambiguation                                       |2024| [ELSEVIER](https://doi.org/10.1016/j.simpa.2024.100715)                   |
| KSAA-CAD Shared Task: Contemporary Arabic Dictionary for Reverse Dictionary and Word Sense Disambiguation           |2024| [ACL](https://aclanthology.org/2024.arabicnlp-1.74/)                      |
| rematchka at ArabicNLU2024: Evaluating Large Language Models for Arabic Word Sense and Location Sense Disambiguation|2024| [ACL](https://aclanthology.org/2024.arabicnlp-1.33/)                      |
| Upaya at ArabicNLU Shared-Task: Arabic Lexical Disambiguation using Large Language Models                           |2024| [ACL](https://aclanthology.org/2024.arabicnlp-1.32/)                      |
| Pirates at ArabicNLU2024: Enhancing Arabic Word Sense Disambiguation using Transformer-Based Approaches             |2024| [ACL](https://aclanthology.org/2024.arabicnlp-1.31/)                      |
| Stacking of BERT and CNN Models for Arabic Word Sense Disambiguation                                                |2023| [ACM](https://dl.acm.org/doi/abs/10.1145/3623379)                         |
| Arabic word sense disambiguation using sense inventories                                                            |2023| [Springer](https://link.springer.com/article/10.1007/s41870-022-01147-w)  |
|Swarm optimization for Arabic word sense disambiguation based on English pre-trained word embeddings                 |2022| [IEEE](https://ieeexplore.ieee.org/abstract/document/9993494)             |
| Word Sense Disambiguation for Arabic Exploiting Arabic WordNet and Word Embedding                                   |2018| [ELSEVIER](https://doi.org/10.1016/j.procs.2018.10.460)                   |
| Word Sense Disambiguation Approach for Arabic Text                                                                  |2016| [SAI](http://dx.doi.org/10.14569/IJACSA.2016.070451)                      |
| Arabic Word Sense Disambiguation Using Wikipedia                                                                    |2016| [Springer](https://link.springer.com/article/10.1007/s10772-016-9376-y)   |
| A Semi-Supervised Method for Arabic Word Sense Disambiguation Using a Weighted Directed Graph                       |2013| [ACL](https://aclanthology.org/I13-1140/)                                 |
| A Hybrid Approach for Arabic Word Sense Disambiguation                                                              |2012| [ICJPL](https://doi.org/10.1142/S1793840612400090)                        |

## Arabic Sentiment Analysis
![ASA:](https://img.shields.io/badge/Papers-34-6A737D?style=for-the-badge)
### Studies & Surveys
| Title                                                                                                                     |Year| Link                                                                      |
| --------------------------------------------------------------------------------------------------------------------------|----|-------------------------------------------------------------------------  |
|The effect of emojis on arabic text in sentiment analysis using deep learning                                              |2025|[AIP](https://doi.org/10.1063/5.0255620)                                   |
|Benchmark Arabic news posts and analyzes Arabic sentiment through RMuBERT and SSL with AMCFFL technique                    |2025|[JEIJ](https://doi.org/10.1016/j.eij.2024.100601)                          |
|A systematic literature review on sentiment analysis techniques, challenges, and future trends                             |2025|[Springer](https://link.springer.com/article/10.1007/s10115-025-02365-x)   |
|Evolving techniques in sentiment analysis: a comprehensive review                                                          |2025|[PEERJ](https://doi.org/10.7717/peerj-cs.2592)                             |
|A Comprehensive Survey of Contemporary Arabic Sentiment Analysis: Methods, Challenges, and Future Directions               |2025|[ACL](https://aclanthology.org/2025.findings-naacl.208/)                   |
|Exploring Contemporary Arabic Sentiment Analysis: Methods, Challenges, and Future Trends                                   |2025|[PJMI](https://www.journals.airsd.org/index.php/pjmi/article/view/517)     |
|Evaluating Large Language Models for Arabic Sentiment Analysis: A Comparative Study Using Retrieval-Augmented Generation   |2024|[ELSEVIER](https://doi.org/10.1016/j.procs.2024.10.210)                    |
|Advancements and challenges in Arabic sentiment analysis: A decade of methodologies, applications, and resource development|2024|[Heliyon](https://doi.org/10.1016/j.heliyon.2024.e39786)                   |
|Natural Language Processing for Arabic Sentiment Analysis: A Systematic Literature Review                                  |2024|[IEEE](https://doi.org/10.1109/TBDATA.2024.3366083)                        |
|Towards a robust deep learning framework for Arabic sentiment analysis                                                     |2024|[CUP](https://doi.org/10.1017/nlp.2024.35)                                 |
|Comparative Analysis of Machine Learning Algorithms for Arabic Sentiment Analysis on Imbalanced Social Media Data          |2024|[IEEE](https://doi.org/10.1109/ICETSIS61505.2024.10459389)                 |
|Analyse the Enhancement of Sentiment Analysis in Arabic by doing a Comparative Study of Several Machine Learning Techniques|2024|[IJRASET](http://dx.doi.org/10.22214/ijraset.2024.60250)                   |
|Sentiment Analysis Methods for Arabic Content on Social Media: A Systematic Review                                         |2023|[IIETA](https://doi.org/10.18280/isi.290138)                               |
|Arabic sentiment analysis using recurrent neural networks: a review                                                        |2021|[Springer](https://link.springer.com/article/10.1007/s10462-021-09989-9)   |
|A review of sentiment analysis research in Arabic language                                                                 |2020|[ELSEVIER](https://doi.org/10.1016/j.future.2020.05.034)                   |
|Deep learning in Arabic sentiment analysis: An overview                                                                    |2019|[JInfSci](https://doi.org/10.1177/0165551519865)                           |

### Papers
| Title                                                                                                                         |Year|Link                                                                       |
| ----------------------------------------------------------------------------------------------------------------------------- |----|---------------------------------------------------------------------------|
|Selective Reading for Arabic Sentiment Analysis                                                                                |2025|[IEEE](https://doi.org/10.1109/ACCESS.2025.3556976)                        |
|Ar-MuSA: A Multimodal Benchmark Dataset and Evaluation Framework for Arabic Sentiment Analysis                                 |2025|[IJIES](http://dx.doi.org/10.22266/ijies2025.0531.03)                      |
|Large Language Models for Arabic Sentiment Analysis and Machine Translation                                                    |2025|[ETASR](https://doi.org/10.48084/etasr.9584)                               |
|Arabic sentiment analysis using ensemble learning and BERT                                                                     |2025|[AIP](https://doi.org/10.1063/5.0256396)                                   |
|Machine Learning Approaches for Sentiment Analysis on Social Media                                                             |2025|[Springer](https://link.springer.com/chapter/10.1007/978-3-031-80334-5_2)  |
|Computational intelligence approach to automated sentiment analysis of Arabic tweets                                           |2025|[Springer](https://link.springer.com/article/10.1007/s00521-024-10808-9)   |
|A combined AraBERT and Voting Ensemble classifier model for Arabic sentiment analysis                                          |2024|[ELSEVIER](https://doi.org/10.1016/j.nlp.2024.100100)                      |
|Sentiment Analysis of Arabic Tweets using ARABERT as a fine tuner and feature extractors                                       |2024|[IEEE](https://doi.org/10.1109/SDS60720.2024.00012)                        |
|FusionAraSA: Fusion-based Model for Accurate Arabic Sentiment Analysis                                                         |2024|[IEEE](https://doi.org/10.1109/eSmarTA62850.2024.10638882)                 |
|Emo-SL Framework: Emoji Sentiment Lexicon Using Text-Based Features and Machine Learning for Sentiment Analysis                |2024|[IEEE](https://doi.org/10.1109/ACCESS.2024.3382836)                        |
|CNN and LSTM based hybrid deep learning model for sentiment analysis on arabic text reviews                                    |2024|[MUET](http://dx.doi.org/10.22581/muet1982.3130)                           |
|ArabBert-LSTM: improving Arabic sentiment analysis based on transformer model and Long Short-Term Memory                       |2024|[FRAI](https://doi.org/10.3389/frai.2024.1408845)                          |
|Sentiment analysis of imbalanced Arabic data using sampling techniques and classification algorithms                           |2024|[IAES](https://doi.org/10.11591/eei.v13i1.5886)                            |
|Improving sentiment analysis using text network features within different machine learning algorithms                          |2024|[IAES](https://doi.org/10.11591/eei.v13i1.5576)                            |
|Sentiment analysis of Arabic social media texts: A machine learning approach to deciphering customer perceptions               |2024|[Heliyon](http://dx.doi.org/10.1016/j.heliyon.2024.e27863)                 |
|Improving Arabic sentiment analysis across context-aware attention deep model based on natural language processing             |2024|[Springer](https://link.springer.com/article/10.1007/s10579-024-09741-z)   |
|Arabic Sentiment Analysis for ChatGPT Using Machine Learning Classification Algorithms: A Hyperparameter Optimization Technique|2024|[ACM](https://doi.org/10.1145/3638285)                                     |
|Deep learning approaches for Arabic sentiment analysis                                                                         |2019|[Springer](https://link.springer.com/article/10.1007/s13278-019-0596-4)    |

## Arabic Question-Answering Systems
![AQAS:](https://img.shields.io/badge/Papers-28-6A737D?style=for-the-badge)
### Studies & Surveys
| Title                                                                                                                          |Year| Link                                                                                  |
| -------------------------------------------------------------------------------------------------------------------------------|----|-------------------------------------------------------------------------------------  |
|Deciphering Arabic question: a dedicated survey on Arabic question analysis methods, challenges, limitations and future pathways|2024|[Springer](https://link.springer.com/article/10.1007/s10462-024-10880-6)               |
|Techniques, datasets, evaluation metrics and future directions of a question answering system                                   |2024|[Springer](https://link.springer.com/article/10.1007/s10115-023-02019-w)               |
|The effect of clustering algorithms on question answering                                                                       |2024|[ELSEVIER](https://doi.org/10.1016/j.eswa.2023.122959)                                 |
|Interactive Question Answering Systems: Literature Review                                                                       |2024|[ACM](https://doi.org/10.1145/3657631)                                                 |
|Decoding Queries: An In-Depth Survey of Quality Techniques for Question Analysis in Arabic Question Answering Systems           |2024|[IEEE](https://doi.org/10.1109/ACCESS.2024.3458466)                                    |
|Challenges and opportunities for Arabic question-answering systems: current techniques and future directions                    |2023|[PEERJ](https://doi.org/10.7717/peerj-cs.1633)                                         |
|Arabic question answering system: a survey                                                                                      |2022|[Springer](https://link.springer.com/article/10.1007/S10462-021-10031-1)               |
|Arabic Question Answering Systems: Gap Analysis                                                                                 |2021|[IEEE](https://doi.org/10.1109/ACCESS.2021.3074950)                                    |
|Pre-trained Transformer-Based Approach for Arabic Question Answering : A Comparative Study                                      |2021|[arXiv](http://dx.doi.org/10.48550/arXiv.2111.05671)                                   |
|Arabic Question Answering: A Study on Challenges, Systems, and Techniques                                                       |2019|[IJCA](http://dx.doi.org/10.5120/ijca2019918524)                                       |
|A Review and Future Perspectives of Arabic Question Answering Systems                                                           |2016|[IEEE](https://doi.org/10.1109/TKDE.2016.2607201)                                      |
|Arabic Question Answering: Systems, Resources, Tools, and Future Trends                                                         |2014|[Springer](https://link.springer.com/article/10.1007/s13369-014-1062-2)                |

### Papers
| Title                                                                                                              |Year|   Link                                                                               |
| -------------------------------------------------------------------------------------------------------------------|----| -------------------------------------------------------------------------------------|
|CollabAS2: Enhancing Arabic Answer Sentence Selection Using Transformer-Based Collaborative Models                  |2025|[Springer](https://link.springer.com/article/10.1007/s13369-024-09345-3)              |
|Question-Answering (QA) Model for a Personalized Learning Assistant for Arabic Language                             |2025|[Springer](https://link.springer.com/chapter/10.1007/978-3-031-82377-0_30)            |
|Automated Question Generation for Science Tests in Arabic Language Using NLP Techniques                             |2025|[Springer](https://link.springer.com/chapter/10.1007/978-3-031-82377-0_24)            |
|ArabicQuest: Enhancing Arabic Visual Question Answering with LLM Fine-Tuning                                        |2024|[IEEE](https://doi.org/10.1109/IMSA61967.2024.10652693)                               |
|Arabic automatic question generation using transformer model                                                        |2024|[AIP](https://doi.org/10.1063/5.0199032)                                              |
|Deep learning model for arabic question-answering chatbot                                                           |2024|[AIP](https://doi.org/10.1063/5.0200612)                                              |
|AraMed: Arabic Medical Question Answering using Pretrained Transformer Language Models                              |2024|[ACL](https://aclanthology.org/2024.osact-1.6/)                                       |
|Exploring Sentence Embedding Representation For Arabic Question Answering                                           |2024|[IJCDS](http://dx.doi.org/10.12785/ijcds/150187)                                      |
|Stacked dynamic memory-coattention network for answering why-questions in Arabic                                    |2024|[Springer](https://link.springer.com/article/10.1007/s00521-024-09525-0)              |
|DAQAS: Deep Arabic Question Answering System based on duplicate question detection and machine reading comprehension|2023|[JKSUCI](https://doi.org/10.1016/j.jksuci.2023.101709)                                |
|Evaluation of an Arabic Chatbot Based on Extractive Question-Answering Transfer Learning and Language Transformers  |2023|[AI](https://doi.org/10.3390/ai4030035)                                               |
|Deep learning-based approach for Arabic open domain question answering                                              |2022|[PEERJ](https://doi.org/10.7717/peerj-cs.952)                                         |
|Neural Arabic Question Answering                                                                                    |2019|[ACL](https://doi.org/10.18653/v1/W19-4612)                                           |
|AlQuAnS – An Arabic Language Question Answering System                                                              |2017|[SciTePress](http://dx.doi.org/10.5220/0006602901440154)                              |
|Toward a New Arabic Question Answering System                                                                       |2018|[IAJIT](https://ccis2k.org/iajit/PDF/Special%20Issue%202018,%20No.%203A/17416.pdf)    |
|An Arabic question-answering system for factoid questions                                                           |2009|[IEEE](https://doi.org/10.1109/NLPKE.2009.5313730)                                    |

# Arabic Retrieval Augmented Generation
![ANER:](https://img.shields.io/badge/Papers-12-6A737D?style=for-the-badge)
### Studies & Surveys
|Title                                                                                                                    |Year| Link                                                                                            |
|-------------------------------------------------------------------------------------------------------------------------|----|-----------------------------------------------------------------------------------------------  |
|Evaluating RAG Pipelines for Arabic Lexical Information Retrieval: A Comparative Study of Embedding and Generation Models|2025|[ACL](https://aclanthology.org/2025.abjadnlp-1.16/)                                              |
|Optimizing RAG Pipelines for Arabic: A Systematic Analysis of Core Components                                            |2025|[arXiv](https://arxiv.org/abs/2506.06339)                                                        |
|Optimizing Arabic Information Retrieval: A Comprehensive Evaluation of Preprocessing Techniques                          |2024|[IEEE](https://doi.org/10.1109/ISIVC61350.2024.10577827)                                         |
|Evaluation of Semantic Search and its Role in Retrieved-Augmented-Generation (RAG) for Arabic Language                   |2024|[arXiv](https://arxiv.org/abs/2403.18350)                                                        |
|Exploring Retrieval Augmented Generation in Arabic                                                                       |2024|[ELSEVIER](https://doi.org/10.1016/j.procs.2024.10.203)                                          |
|A Comparative Evaluation of Retrieval-Augmented Generation For Arabic Documents                                          |2024|[IEEE](https://doi.org/10.1109/ACIT62805.2024.10877197)                                          |


### Papers
|Title                                                                                                               |Year|Link                                                                               |
|------------------------------------------------------------------------------------------------------------------- |----|---------------------------------------------------------------------------------- |
|Enhancing Arabic Retrieval Augmented Generation through Language Processing                                         |2025|[ACL](https://aclanthology.org/2025.icnlsp-1.45/)                                  |
|Advancing Arabic Medical Question Answering Systems with RAG and LLMs Integration                                   |2025|[IEEE](https://doi.org/10.1109/ICTCS65341.2025.10989446)                           |
|RFPG: Question-Answering from Low-Resource Language (Arabic) Texts using Factually Aware RAG                        |2025|[IEEE](https://doi.org/10.1109/CIC62241.2024.00023)                                |
|Enhanced Arabic Retrieval Augmented Generation Using Nested Embedding Models                                        |2025|[Springer](https://link.springer.com/chapter/10.1007/978-3-031-91235-1_11)         |
|Enhancing Arabic Information Retrieval for Question Answering                                                       |2024|[ACM](https://doi.org/10.1145/3644713.3644763)                                     |
|Semantic Embeddings for Arabic Retrieval Augmented Generation (ARAG)                                                |2024|[IJACSA](https://doi.org/10.14569/IJACSA.2023.01411135)                            |

# Arabic Information Retrieval
![ANER:](https://img.shields.io/badge/Papers-10-6A737D?style=for-the-badge)
### Studies & Surveys
|Title                                                                                                                    |Year| Link                                                                                            |
|-------------------------------------------------------------------------------------------------------------------------|----|-----------------------------------------------------------------------------------------------  |
|A Survey of Model Architectures in Information Retrieval                                                                 |2025|[arXiv](https://doi.org/10.48550/arXiv.2502.14822)                                               |
|Optimizing Arabic Information Retrieval: A Comprehensive Evaluation of Preprocessing Techniques                          |2024|[IEEE](https://doi.org/10.1109/ISIVC61350.2024.10577827)                                         |
|A Review on Recent Arabic Information Retrieval Techniques                                                               |2022|[EAI](https://doi.org/10.4108/eetiot.v8i3.2276)                                                  |
|Arabic Cross-Language Information Retrieval: A Review                                                                    |2016|[ACM](https://doi.org/10.1145/2789210)                                                           |


### Papers
|Title                                                                                                               |Year|Link                                                                               |
|------------------------------------------------------------------------------------------------------------------- |----|---------------------------------------------------------------------------------- |
|Enhanced Arabic Text Retrieval with Attentive Relevance Scoring                                                     |2025|[IEEE](https://doi.org/10.1109/MLSP62443.2025.11204239)                            |
|Incorporating Deep Median Networks for Arabic Document Retrieval Using Word Embeddings-Based Query Expansion        |2024|[JISTaP](https://doi.org/10.1633/JISTaP.2024.12.3.3)                               |
|Automating the search for legal information in Arabic: A novel approach to document retrieval                       |2024|[RTJ](https://doi.org/10.32362/2500-316X-2024-12-5-7-16)                           |
|Arabic Documents Information Retrieval for Printed, Handwritten, and Calligraphy Image                              |2021|[IEEE](https://doi.org/10.1109/ACCESS.2021.3066477)                                |
|NERWS: Towards Improving Information Retrieval of Digital Library Management System Using Named Entity Recognition and Word Sense|2021|[BDCC](https://doi.org/10.3390/bdcc5040059)                           |
|Improving Arabic information retrieval using word embedding similarities                                            |2018|[Springer](https://link.springer.com/article/10.1007/s10772-018-9492-y)                          |


# Arabic Image Captioning
![AIC:](https://img.shields.io/badge/Papers-20-6A737D?style=for-the-badge)
### Studies & Surveys
|Title                                                                                                                  |Year| Link                                                                                            |
|-----------------------------------------------------------------------------------------------------------------------|----|-----------------------------------------------------------------------------------------------  |
|A Systematic Literature Review on Using the Encoder-Decoder Models for Image Captioning in English and Arabic Languages|2023|[ApplSci](https://doi.org/10.3390/app131910894)                                                  |
|A performance analysis of transformer-based deep learning models for Arabic image captioning                           |2023|[JKSUCI](https://doi.org/10.1016/j.jksuci.2023.101750)                                           |
|An overview of the Image Description Systems in Arabic Language                                                        |2023|[IEEE](https://doi.org/10.1109/AICCSA59173.2023.10479235)                                        |
|Arabic Image Captioning using Pre-training of Deep Bidirectional Transformers                                          |2022|[Thesis](https://lup.lub.lu.se/luur/download?func=downloadFile&recordOId=9077401&fileOId=9077402)|
|Arabic Image Captioning: The Effect of Text Pre processing on the Attention Weights and the BLEU-N Scores              |2022|[SAI](http://dx.doi.org/10.14569/IJACSA.2022.0130751)                                            |
|Bench-Marking And Improving Arabic Automatic Image Captioning Through The Use Of Multi-Task Learning Paradigm          |2022|[arXiv](http://dx.doi.org/10.48550/arXiv.2202.05474)                                             |
|Deep Learning for Arabic Image Captioning: A Comparative Study of Main Factors and Preprocessing Recommendations       |2021|[SAI](https://dx.doi.org/10.14569/IJACSA.2021.0121105)                                           |
|A survey on Arabic Image Captioning Systems Using Deep Learning Models                                                 |2020|[IEEE](https://doi.org/10.1109/IIT50501.2020.9299027)                                            |

### Papers
|Title                                                                                                               |Year|Link                                                                               |
|------------------------------------------------------------------------------------------------------------------- |----|---------------------------------------------------------------------------------- |
|Attention-based transformer model for Arabic image captioning                                                       |2025|[Springer](https://link.springer.com/article/10.1007/s00521-025-11199-1)           |
|Utilizing Deep Learning Technique for Arabic Image Captioning                                                       |2024|[Springer](https://link.springer.com/chapter/10.1007/978-3-031-59707-7_17)         |
|Ar-CM-ViMETA: Arabic Image Captioning based on Concept Model and Vision-based Multi-Encoder Transformer Architecture|2024|[IAJIT](http://dx.doi.org/10.34028/iajit/21/3/9)                                   |
|Violet: A Vision-Language Model for Arabic Image Captioning with Gemini Decoder                                     |2023|[ACL](https://doi.org/10.18653/v1/2023.arabicnlp-1.1)                              |
|Towards Arabic Image Captioning: A Transformer-Based Approach                                                       |2023|[IEEE](https://doi.org/10.1109/NILES59815.2023.10296781)                           |
|Arabic Captioning for Images of Clothing Using Deep Learning                                                        |2023|[Sensors](https://doi.org/10.3390/s23083783)                                       |
|Improved Arabic image captioning model using feature concatenation with pre-trained word embedding                  |2023|[Springer](https://link.springer.com/article/10.1007/s00521-023-08744-1)           |
|Arabic Image Captioning (AIC): Utilizing Deep Learning and Main Factors Comparison and Prioritization               |2020|[BUID](https://bspace.buid.ac.ae/handle/1234/1995)                                 |
|AraCap: A hybrid deep learning architecture for Arabic Image Captioning                                             |2021|[ELSEVIER](https://doi.org/10.1016/j.procs.2021.05.108)                            |
|Arabic Image Captioning Using Deep Learning with Attention                                                          |2021|[UGA](https://www.ai.uga.edu/sites/default/files/inline-files/theses/sabri_sabri_m_202108_ms.pdf)|
|Automatic Arabic Image Captioning using RNN-LSTM-Based Language Model and CNN                                       |2018|[SAI](https://dx.doi.org/10.14569/IJACSA.2018.090610)                              |
|Generating Image Captions in Arabic Using Root-Word Based Recurrent Neural Networks and Deep Neural Networks        |2018|[AAAI](https://doi.org/10.1609/aaai.v32i1.12179)                                   |

# Arabic Named Entity Recognition
![ANER:](https://img.shields.io/badge/Papers-24-6A737D?style=for-the-badge)
### Studies & Surveys
|Title                                                                                                                  |Year| Link                                                                                            |
|-----------------------------------------------------------------------------------------------------------------------|----|-----------------------------------------------------------------------------------------------  |
|Advancements in Arabic Named Entity Recognition: A Comprehensive Review                                                |2024|[IEEE](https://doi.org/10.1109/ACCESS.2024.3491897)                                              |
|A Survey of Techniques and Challenges in Arabic Named Entity Recognition                                               |2024|[JSST](https://solidstatetechnology.us/index.php/JSST/article/view/11657)                        |
|A Survey on Arabic Named Entity Recognition: Past, Recent Advances, and Future Trends                                  |2023|[IEEE](https://doi.org/10.1109/TKDE.2023.3303136)                                                |
|A Recent Survey of Arabic Named Entity Recognition on Social Media                                                     |2020|[IIETA](http://dx.doi.org/10.18280/ria.340202)                                                   |
|A Comparative Review of Machine Learning for Arabic Named Entity Recognition                                           |2017|[IJASEIT](http://dx.doi.org/10.18517/ijaseit.7.2.1810)                                           |
|Statistical Arabic Name Entity Recognition Approaches: A Survey                                                        |2017|[JPROCS](https://doi.org/10.1016/j.procs.2017.08.288)                                            |
|Arabic Rule-Based Named Entity Recognition Systems Progress and Challenges                                             |2017|[IJASEIT](http://dx.doi.org/10.18517/ijaseit.7.3.1811)                                           |
|Arabic Named Entity Recognition—A Survey and Analysis                                                                  |2016|[Springer](https://link.springer.com/chapter/10.1007/978-3-319-39345-2_8)                        |
|A Survey of Arabic Named Entity Recognition and Classification                                                         |2014|[COLI](https://doi.org/10.1162/COLI_a_00178)                                                     |

### Papers
|Title                                                                                                               |Year|Link                                                                               |
|------------------------------------------------------------------------------------------------------------------- |----|---------------------------------------------------------------------------------- |
|An Advanced Natural Language Processing Framework for Arabic Named Entity Recognition                               |2025|[ApplSci](https://doi.org/10.3390/app15063073)                                     |
|ARDIAL-BERT: Advancing Multidialectal Arabic Named Entity Recognition Through Continual Pretraining                 |2025|[IEEE](https://doi.org/10.1109/TAI.2025.3562162)                                   |
|AMWAL: Named Entity Recognition for Arabic Financial News                                                           |2025|[ACL](https://aclanthology.org/2025.finnlp-1.20/)                                  |
|Flat and Nested Named Entity Recognition in Arabic Language                                                         |2024|[IEEE](https://doi.org/10.1109/IRASET60544.2024.10548518)                          |
|Arabic NER Evaluation: Pre-Trained Models via Contrastive Learning vs. LLM Few-Shot Prompting                       |2024|[ELSEVIER](https://doi.org/10.1016/j.procs.2024.10.196)                            |
|Bangor University at WojoodNER 2024: Advancing Arabic Named Entity Recognition with CAMeLBERT-Mix                   |2024|[ACL](https://doi.org/10.18653/v1/2024.arabicnlp-1.105)                            |
|Addax at WojoodNER 2024: Attention-Based Dual-Channel Neural Network for Arabic Named Entity Recognition            |2024|[ACL](https://doi.org/10.18653/v1/2024.arabicnlp-1.103)                            |
|Active Learning with AraGPT2 for Arabic Named Entity Recognition                                                    |2023|[Springer](https://link.springer.com/chapter/10.1007/978-3-031-41774-0_18)         |
|Arabic Fine-Grained Entity Recognition                                                                              |2023|[arXiv](https://doi.org/10.48550/arXiv.2310.17333)                                 |
|Arabic Named Entity Recognition in Arabic Tweets Using BERT-based Models                                            |2022|[JPROC](https://doi.org/10.1016/j.procs.2022.07.109)                               |
|Wojood: Nested Arabic Named Entity Corpus and Recognition using BERT                                                |2022|[arXiv](https://doi.org/10.48550/arXiv.2205.09651)                                 |
|Arabic Named Entity Recognition Using Variant Deep Neural Network Architectures and Combinatorial Feature Embedding Based on CNN, LSTMandBERT|2022|[ACL](https://aclanthology.org/2022.paclic-1.33/)         |
|Arabic Named Entity Recognition Using Transformer-based-CRF Model                                                   |2021|[ACL](https://aclanthology.org/2021.icnlsp-1.31/)                                  |
|Classical Arabic Named Entity Recognition Using Variant Deep Neural Network Architectures and BERT                  |2021|[IEEE](https://doi.org/10.1109/ACCESS.2021.3092261)                                |
|Arabic Named Entity Recognition: A BERT-BGRU Approach                                                               |2021|[TSP](http://dx.doi.org/10.32604/cmc.2021.016054)                                  |


# Arabic LLM
![ANER:](https://img.shields.io/badge/Papers-11-6A737D?style=for-the-badge)
### Studies & Surveys
|Title                                                                                                                                    |Year| Link                                                                                            |
|-----------------------------------------------------------------------------------------------------------------------------------------|----|-----------------------------------------------------------------------------------------------  |
|Evaluating Arabic Large Language Models: A Survey of Benchmarks, Methods, and Gaps                                                       |2025|[arXiv](https://doi.org/10.48550/arXiv.2510.13430)                                               |
|Evaluating Translation Quality: A Qualitative and Quantitative Assessment of Machine and LLM-Driven Arabic–English Translations          |2025|[INFO](https://doi.org/10.3390/info16060440)                                                     |
|AL-QASIDA: Analyzing LLM Quality and Accuracy Systematically in Dialectal Arabic|2024|[IEEE](https://doi.org/10.1109/ACCESS.2024.3491897)|2025|[arXiv](https://arxiv.org/abs/2412.04193)                                                        |
|Large Language Models and Arabic Content: A Review                                                                                       |2025|[arXiv](https://arxiv.org/abs/2505.08004)                                                        |
|A Survey of Large Language Models for Arabic Language and its Dialects                                                                   |2024|[arXiv](https://arxiv.org/abs/2410.20238)                                                        |


### Papers
|Title                                                                                                               |Year|Link                                                                               |
|------------------------------------------------------------------------------------------------------------------- |----|---------------------------------------------------------------------------------- |
|AraHalluEval: A Fine-grained Hallucination Evaluation Framework for Arabic LLMs                                     |2025|[ACL](https://doi.org/10.18653/v1/2025.arabicnlp-main.12)                          |
|Command R7B Arabic: A Small, Enterprise Focused, Multilingual, and Culturally Aware Arabic LLM                      |2025|[arXiv](https://arxiv.org/abs/2503.14603)                                          |
|Synthetic Arabic Medical Dialogues Using Advanced Multi-Agent LLM Techniques                                        |2024|[ACL](https://doi.org/10.18653/v1/2024.arabicnlp-1.2)                              |
|Creating Arabic LLM Prompts at Scale                                                                                |2024|[arXiv](https://arxiv.org/abs/2408.05882)                                          |
|ALLaM: Large Language Models for Arabic and English                                                                 |2024|[arXiv](https://arxiv.org/abs/2407.15390)                                          |
|Arabic Mini-ClimateGPT : A Climate Change and Sustainability Tailored Arabic LLM                                    |2023|[arXiv](https://arxiv.org/abs/2312.09366)                                          |

# Arabic LLM Reasoning
![ANER:](https://img.shields.io/badge/Papers-8-6A737D?style=for-the-badge)
### Studies & Surveys
|Title                                                                                                                                    |Year| Link                                                                                            |
|-----------------------------------------------------------------------------------------------------------------------------------------|----|-----------------------------------------------------------------------------------------------  |
|Saudi-Alignment Benchmark: Assessing LLMs Alignment with Cultural Norms and Domain Knowledge in the Saudi Context                        |2025|[ACL](https://doi.org/10.18653/v1/2025.arabicnlp-main.11)                                        |
|ArabicSense: A Benchmark for Evaluating Commonsense Reasoning in Arabic with Large Language Models                                       |2025|[ACL](https://aclanthology.org/2025.wacl-1.1/)                                                   |
|AraTable: Benchmarking LLMs' Reasoning and Understanding of Arabic Tabular Data                                                          |2025|[arXiv](https://doi.org/10.48550/arXiv.2507.18442)                                               |
|AraReasoner: Evaluating Reasoning-Based LLMs for Arabic NLP                                                                              |2025|[arXiv](https://doi.org/10.48550/arXiv.2506.08768)                                               |


### Papers
|Title                                                                                                               |Year|Link                                                                               |
|------------------------------------------------------------------------------------------------------------------- |----|---------------------------------------------------------------------------------- |
|In-Context Learning in Large Language Models (LLMs): Mechanisms, Capabilities, and Implications for Advanced Knowledge Representation and Reasoning|2025|[IEEE](https://doi.org/10.1109/ACCESS.2025.3575303) |
|Cross-Cultural Transfer of Commonsense Reasoning in LLMs: Evidence from the Arab World                              |2025|[ACL](https://doi.org/10.18653/v1/2025.findings-emnlp.247)                         |
|Fann or Flop: A Multigenre, Multiera Benchmark for Arabic Poetry Understanding in LLMs                              |2025|[ACL](https://doi.org/10.18653/v1/2025.emnlp-main.1023)                            |
|Multi-Hop Arabic LLM Reasoning in Complex QA                                                                        |2024|[ELSEVIER](https://www.sciencedirect.com/science/article/pii/S1877050924029806)    |

