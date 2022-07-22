# CheXray: Automatic Diagnosis of Chest X-Rays with AI

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

In the four teaching hospitals on Longwood Avenue in Boston, Massachusetts, there are more radiologists working there than there are in West Africa. In addition, it can take up to two years to train people to become radiologists. In those two years, one of the most important skills radiologists learn is how to write a diagnosis report for their patients and determining what diseases they need to treat their patients for. How do we give this service to populations who need it? Let's use AI to write and summarize these diagnostic reports so that everyone can start off on an equal playing field. 
Using PyTorch and fast.ai, I built two separate models, one based off of [this research paper](https://arxiv.org/abs/1502.03044) designed to generate radiologists reports and another based off of [this forum](https://forums.fast.ai/t/gradient-blending-for-multi-modal-models-in-progress/75645/12) and the [fast.ai courses](https://course.fast.ai/) to summarize the generated report, the images, and other clinical data into a list of diseases the patient most likely needs to be checked out for. 
To use the service I've created for the above stated purpose, visit the Binder website: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AndrewJHinh/CheXray/HEAD?urlpath=%2Fvoila%2Frender%2Fproduction.ipynb) 

### Built With

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [fast.ai](https://www.fast.ai/)
* [Jupyter Notebook](https://jupyter.org/)
* [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)
* [Binder](https://mybinder.org/)

<!-- GETTING STARTED -->
## Getting Started

* Go to the website link in the "About the Project" section if you just want to see the website. Otherwise, please wait: the files needed to reproduce the website need to be uploaded. In the meantime, the following instructions are not valid.

### Prerequisites
* Currently supported for Mac OS only
* Latest version of Python
* fastai==2.0.0
  ```sh
  pip install fastai==2.0.0
  ```
* fastcore==1.0.0
  ```sh
  pip install fastcore==1.0.0
  ```
* fastbook==0.0.8
  ```sh
  pip install fastbook==0.0.8
  ```
* torch==1.6.0  for Jupyter Notebook
	* torch==1.7.0 for Google Colab
  ```sh
  pip install torch==1.6.0, pip install torch==1.7.0
  ```
* spacy==2.2.4
  ```sh
  pip install spacy==2.2.4
  ```
* Latest version of FastText
  ```sh
  git clone https://github.com/facebookresearch/fastText.git
  cd fastText
  pip install . 
  ```
* Latest version of NLTK
  ```sh
  pip install nltk
  ```
* Latest version of rouge-score
  ```sh
  pip install rouge-score
  ```
  
### Installation: 

1. Follow the instructions on [physionet.org](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) to become a [credentialed user](https://physionet.org/login/?next=/settings/credentialing/) and [sign the data use agreement](https://physionet.org/login/?next=/sign-dua/mimic-cxr-jpg/2.0.0/) to get the MIMIC-CXR Dataset (make sure to have around 960GB of storage to store it)
2. Clone the repo
   ```sh
   git clone https://github.com/AndrewJHinh/CheXray.git
   ```
3. Go to local repo
   ```sh
   cd CheXray
   ```
4. Install Dependencies (as mentioned in Prerequisites section) 

<!-- USAGE EXAMPLES -->
## Usage

1. Open Jupyter Notebook
   ```sh
   jupyter notebook
   ```
2. Follow mimic-cxr.ipynb
    2a. For each "training model" section, go to the corresponding google colab notebook, change the hardware to GPU, and train the model (for google colab notebooks imgcapall.ipynb and sum.ipynb after changing the hardware, run "editing files" cells, restart the runtime, and continue to train the model)
    
3. Go to production.ipynb to test out your models and check if everything works on Viola (replace "notebooks" in url with "viola/render")
4. Make separate local repo within current directory containing everything in sample repo (named CheXray) within the repo
5. Push to GitHub (using [Git-LFS](https://forums.fast.ai/t/deploying-your-notebook-as-an-app-under-10-minutes/70621) for files larger than 25mb)
6. Copy GitHub link to Binder and deploy

<!-- ROADMAP -->
## Roadmap

* Support models that take in more than two views
* Make models better
* Use other datasets such as [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) and [Open-I](https://openi.nlm.nih.gov/)

<!-- CONTRIBUTING -->
## Contributing

Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- CONTACT -->
## Contact

Andrew Hinh - ajhinh@gmail.com - 4158103676

Project Link: [https://github.com/AndrewJHinh/CheXray](https://github.com/AndrewJHinh/CheXray)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* Rochelle Devault (Science Teacher at UHS)
* [fast.ai course](https://course.fast.ai/)
* [Skumarr53](https://github.com/Skumarr53/Image-Caption-Generation-using-Fastai/blob/master/main-Finalized.ipynb)